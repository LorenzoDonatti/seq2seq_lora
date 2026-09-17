"""Paired causal ARX/VARX and restricted ARIMAX/VARIMAX estimators.

For horizon H, w_s = weather[s-H]. Hence w_{t+1:t+H} is entirely observed
at origin t. The same delayed exogenous design is used by every statistical
family. ARIMAX/VARIMAX have d in {0,1} and a diagonal MA(1) term. Fitting is
regularized conditional sum of squares, not unrestricted VARMA maximum likelihood.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter


class UnstableModelError(ValueError):
    """A candidate violates the declared stationary dynamics constraint."""


def _ridge_solution(design, target, alpha):
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    if alpha == 0:
        return np.linalg.lstsq(design, target, rcond=None)[0]
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


class HistoricalWeatherARIMAX:
    """Vector model with diagonal MA or its independent per-node restriction."""
    def __init__(self, n_targets=8, pred_length=1, lags=3, alpha=1.0,
                 joint=False, difference=0, moving_average=False):
        if lags < 1 or pred_length < 1 or difference not in (0, 1) or alpha < 0:
            raise ValueError("Invalid statistical configuration")
        self.n_targets, self.pred_length = n_targets, pred_length
        self.lags, self.alpha = lags, alpha
        self.joint, self.difference = joint, difference
        self.moving_average = moving_average
        self.coefficients, self.ma = [], []
        self.training_summary = {}

    @property
    def burn_in(self):
        return max(self.pred_length, self.lags + self.difference)

    def _equation_data(self, segment, node):
        y = np.asarray(segment[:, :self.n_targets], dtype=np.float64)
        weather = np.asarray(segment[:, self.n_targets:], dtype=np.float64)
        z = y if self.difference == 0 else np.vstack([np.zeros((1, self.n_targets)), np.diff(y, axis=0)])
        times = np.arange(self.burn_in, len(y))
        channels = np.arange(self.n_targets) if self.joint else np.array([node])
        ar = np.stack([z[times-lag][:, channels] for lag in range(1, self.lags+1)], axis=1)
        design = np.column_stack([np.ones(len(times)), ar.reshape(len(times), -1),
                                  weather[times-self.pred_length]])
        return design, z[times, node]

    def fit(self, X_train, y_train, *, training_segments):
        if X_train.shape[1] <= self.burn_in:
            raise ValueError(f"History must exceed {self.burn_in} hours for horizon "
                             f"{self.pred_length}; use a history at least as long as H.")
        segments = [np.asarray(s, dtype=np.float64) for s in training_segments
                    if len(s) > self.burn_in]
        if not segments:
            raise ValueError("No contiguous training segments support this model")
        self.coefficients, self.ma = [], []
        convergence = []
        for node in range(self.n_targets):
            equations = [self._equation_data(s, node) for s in segments]

            def profile(theta, return_beta=False):
                # e_s + theta*e_{s-1} = z_s - design_s @ beta.
                # Each segment starts with zero conditional innovation state.
                fd = np.concatenate([lfilter([1.0], [1.0, theta], design, axis=0)
                                     for design, target in equations])
                fy = np.concatenate([lfilter([1.0], [1.0, theta], target)
                                     for design, target in equations])
                beta = _ridge_solution(fd, fy, self.alpha)
                residual = fy - fd @ beta
                objective = residual @ residual + self.alpha * (beta[1:] @ beta[1:])
                return beta if return_beta else float(objective)

            if self.moving_average:
                optimum = minimize_scalar(profile, bounds=(-0.95, 0.95), method="bounded",
                                          options={"xatol": 1e-5, "maxiter": 100})
                if not optimum.success or not np.isfinite(optimum.fun):
                    raise RuntimeError("Conditional MA optimization did not converge")
                theta = float(optimum.x)
                convergence.append(bool(optimum.success))
            else:
                theta = 0.0
                convergence.append(True)
            self.coefficients.append(profile(theta, return_beta=True))
            self.ma.append(theta)
        radius = self.spectral_radius()
        if radius >= 1.0:
            raise UnstableModelError(f"AR dynamics not stationary: spectral radius={radius:.6f}")
        self.training_summary = {
            "estimator": "ridge conditional sum of squares; profiled diagonal MA(1)",
            "joint": self.joint, "p": self.lags, "d": self.difference,
            "q": int(self.moving_average), "ma_structure": "diagonal",
            "weather_lag_hours": self.pred_length,
            "training_segment_lengths": [len(s) for s in segments],
            "conditional_observations": sum(len(s)-self.burn_in for s in segments),
            "excluded_short_segment_rows": sum(len(s) for s in training_segments if len(s)<=self.burn_in),
            "spectral_radius": radius, "converged_per_node": convergence,
            "innovation_initialization": "zero independently at each segment/window",
            "ma_coefficients": self.ma,
        }
        return self

    def spectral_radius(self):
        n, p = self.n_targets, self.lags
        companion = np.zeros((n*p, n*p))
        for node, beta in enumerate(self.coefficients):
            if self.joint:
                companion[node, :] = beta[1:1+n*p]
            else:
                for lag in range(p):
                    companion[node, lag*n+node] = beta[1+lag]
        if p > 1:
            companion[n:, :-n] = np.eye(n*(p-1))
        return float(np.max(np.abs(np.linalg.eigvals(companion))))

    def predict(self, X, pred_length=None):
        horizon = self.pred_length if pred_length is None else pred_length
        if horizon != self.pred_length:
            raise ValueError("Horizon must match fitted delayed-exogenous model")
        if X.shape[1] < self.burn_in:
            raise ValueError("Insufficient observed history")
        batch, length, _ = X.shape
        y = np.asarray(X[:, :, :self.n_targets], dtype=np.float64)
        z = y.copy() if self.difference == 0 else np.concatenate(
            [np.zeros((batch,1,self.n_targets)), np.diff(y,axis=1)], axis=1)
        weather = np.asarray(X[:, :, self.n_targets:], dtype=np.float64)
        error = np.zeros((batch,self.n_targets))
        for step in range(self.burn_in, length):
            for node, beta in enumerate(self.coefficients):
                own = z[:, step-self.lags:step, :][:, ::-1]
                ar = own.reshape(batch,-1) if self.joint else own[:,:,node]
                design = np.column_stack([np.ones(batch), ar, weather[:,step-horizon]])
                error[:,node] = z[:,step,node] - design @ beta - self.ma[node]*error[:,node]
        history = [z[:,i].copy() for i in range(length)]
        levels = y[:,-1].copy()
        forecasts = []
        for lead in range(horizon):
            step = length + lead
            ar_all = np.stack(history[-self.lags:][::-1],axis=1)
            next_z = np.empty((batch,self.n_targets))
            for node,beta in enumerate(self.coefficients):
                ar = ar_all.reshape(batch,-1) if self.joint else ar_all[:,:,node]
                # step-H never exceeds length-1: no future weather is accessed.
                design = np.column_stack([np.ones(batch), ar, weather[:,step-horizon]])
                next_z[:,node] = design @ beta + self.ma[node]*error[:,node]
            history.append(next_z)
            levels = levels + next_z if self.difference else next_z
            forecasts.append(levels.copy())
            error.fill(0.0)  # conditional expectation of future innovations
        return np.stack(forecasts,axis=1).astype(np.float32)

    def total_parameters(self):
        return sum(len(beta) for beta in self.coefficients) + self.n_targets*int(self.moving_average)
