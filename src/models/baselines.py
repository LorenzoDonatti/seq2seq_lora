"""Causal Box--Jenkins ARIMAX and multivariate VARX estimators.

For horizon H, w_s = weather[s-H]. Hence w_{t+1:t+H} is entirely observed
at origin t. The same delayed exogenous design is used by every statistical
family. ARIMAX is estimated per node by exact Gaussian state-space likelihood over
independent contiguous segments. VARX is a regularized joint linear model.
"""
import numpy as np
import warnings
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


class UnstableModelError(ValueError):
    """A candidate violates the declared stationary dynamics constraint."""


def _ridge_solution(design, target, alpha):
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    if alpha == 0:
        return np.linalg.lstsq(design, target, rcond=None)[0]
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def _segment_equations(segment, n_targets, horizon, node):
    """Align y[t] with weather[t-H] without using future meteorology."""
    segment = np.asarray(segment, dtype=np.float64)
    return segment[horizon:, node], segment[:-horizon, n_targets:]


def _fit_segmented_sarimax(segments, n_targets, horizon, node, order, maxiter=200):
    """Maximize the sum of independent-segment SARIMAX log likelihoods."""
    models = []
    lengths = []
    for segment in segments:
        endog, exog = _segment_equations(segment, n_targets, horizon, node)
        if len(endog) <= sum(order) + 2:
            continue
        models.append(SARIMAX(endog, exog=exog, order=order, trend="c",
                              enforce_stationarity=True, enforce_invertibility=True))
        lengths.append(len(endog))
    if not models:
        raise ValueError(f"No segment supports ARIMAX{order}")
    reference = models[int(np.argmax(lengths))]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            initial = reference.untransform_params(reference.start_params)
    except (ValueError, np.linalg.LinAlgError):
        raise RuntimeError(f"ARIMAX{order} could not initialize state-space parameters")

    def objective(unconstrained):
        values = [-model.loglike(unconstrained, transformed=False) for model in models]
        total = float(np.sum(values))
        return total if np.isfinite(total) else 1e100

    optimum = minimize(objective, initial, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-8})
    if not optimum.success:
        # Expensive recovery path only for candidates whose joint optimization
        # cannot converge from the deterministic state-space starting values.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preliminary = reference.fit(disp=False, maxiter=min(75, maxiter))
            retry_initial = reference.untransform_params(preliminary.params)
            retry = minimize(objective, retry_initial, method="L-BFGS-B",
                             options={"maxiter": maxiter, "ftol": 1e-8})
            if retry.success or retry.fun < optimum.fun:
                optimum = retry
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass
    if not np.isfinite(optimum.fun):
        raise RuntimeError(f"ARIMAX{order} likelihood is not finite")
    params = reference.transform_params(optimum.x)
    loglike = -objective(optimum.x)
    nobs = int(sum(lengths))
    k = int(len(params))
    aic = 2 * k - 2 * loglike
    aicc = aic + (2 * k * (k + 1) / (nobs - k - 1) if nobs > k + 1 else np.inf)
    bic = np.log(nobs) * k - 2 * loglike
    diagnostic_pvalues = []
    residual_count = 0
    for model in models:
        result = model.filter(params)
        burn = max(result.loglikelihood_burn, max(order) + 1)
        residual = np.asarray(result.resid[burn:], dtype=np.float64)
        residual = residual[np.isfinite(residual)]
        residual_count += len(residual)
        if len(residual) >= 12:
            lag = min(24, max(1, len(residual) // 5))
            diagnostic_pvalues.append(float(acorr_ljungbox(
                residual, lags=[lag], return_df=True)["lb_pvalue"].iloc[0]))
    return {
        "params": params, "order": tuple(int(value) for value in order),
        "log_likelihood": float(loglike), "aic": float(aic),
        "aicc": float(aicc), "bic": float(bic), "nobs": nobs,
        "parameters": k, "converged": bool(optimum.success),
        "optimizer_message": str(optimum.message),
        "ljung_box_min_pvalue": min(diagnostic_pvalues) if diagnostic_pvalues else None,
        "diagnostic_residuals": residual_count,
        "segment_lengths": lengths,
    }


class BoxJenkinsARIMAX:
    """One independently identified Box--Jenkins ARIMAX model per RSSI node."""
    def __init__(self, n_targets=8, pred_length=1, orders=None):
        self.n_targets = n_targets
        self.pred_length = pred_length
        self.orders = ([tuple(order) for order in orders] if orders is not None
                       else [(1, 0, 1)] * n_targets)
        if len(self.orders) != n_targets:
            raise ValueError("ARIMAX requires one (p,d,q) order per node")
        self.fits = []
        self.training_summary = {}

    def fit(self, X_train, y_train, *, training_segments):
        segments = [np.asarray(segment, dtype=np.float64) for segment in training_segments]
        self.fits = [_fit_segmented_sarimax(
            segments, self.n_targets, self.pred_length, node, self.orders[node])
            for node in range(self.n_targets)]
        self.training_summary = {
            "estimator": "exact Gaussian state-space likelihood summed over independent contiguous segments",
            "selection": "training-only AICc per node; ACF/PACF bound candidate orders",
            "joint": False, "orders": [list(order) for order in self.orders],
            "weather_lag_hours": self.pred_length,
            "diagnostics": [{key: value for key, value in fit.items() if key != "params"}
                            for fit in self.fits],
            "innovation_initialization": "stationary initialization independently per segment/window",
        }
        return self

    def predict(self, X, pred_length=None):
        horizon = self.pred_length if pred_length is None else pred_length
        if horizon != self.pred_length:
            raise ValueError("Horizon must match fitted delayed-exogenous model")
        X = np.asarray(X, dtype=np.float64)
        batch, length, _ = X.shape
        predictions = np.empty((batch, horizon, self.n_targets), dtype=np.float32)
        for sample in range(batch):
            weather = X[sample, :, self.n_targets:]
            for node, fit in enumerate(self.fits):
                endog = X[sample, horizon:, node]
                observed_exog = weather[:-horizon]
                model = SARIMAX(endog, exog=observed_exog, order=fit["order"], trend="c",
                                enforce_stationarity=True, enforce_invertibility=True)
                result = model.filter(fit["params"])
                future_exog = weather[length-horizon:length]
                predictions[sample, :, node] = result.forecast(
                    horizon, exog=future_exog).astype(np.float32)
        return predictions

    def total_parameters(self):
        return sum(fit["parameters"] for fit in self.fits)


class HistoricalWeatherVARX:
    """Regularized joint VARX with dense or explicitly sparse lag sets."""
    def __init__(self, n_targets=8, pred_length=1, lags=3, alpha=1.0):
        lag_indices = (list(range(1, int(lags) + 1))
                       if np.isscalar(lags) else sorted({int(lag) for lag in lags}))
        if not lag_indices or lag_indices[0] < 1 or pred_length < 1 or alpha < 0:
            raise ValueError("Invalid statistical configuration")
        self.n_targets, self.pred_length = n_targets, pred_length
        self.lag_indices, self.alpha = lag_indices, alpha
        self.lags = max(lag_indices)  # Backward-compatible maximum-lag attribute.
        self.joint = True
        self.coefficients = []
        self.training_summary = {}

    @property
    def burn_in(self):
        return max(self.pred_length, max(self.lag_indices))

    def _equation_data(self, segment, node):
        y = np.asarray(segment[:, :self.n_targets], dtype=np.float64)
        weather = np.asarray(segment[:, self.n_targets:], dtype=np.float64)
        z = y
        times = np.arange(self.burn_in, len(y))
        channels = np.arange(self.n_targets)
        ar = np.stack([z[times-lag][:, channels] for lag in self.lag_indices], axis=1)
        design = np.column_stack([np.ones(len(times)), ar.reshape(len(times), -1),
                                  weather[times-self.pred_length]])
        return design, z[times, node]

    def fit(self, X_train, y_train, *, training_segments):
        if X_train.shape[1] < self.burn_in:
            raise ValueError(f"History must contain at least {self.burn_in} hours for "
                             f"the requested lag structure and horizon {self.pred_length}.")
        segments = [np.asarray(s, dtype=np.float64) for s in training_segments
                    if len(s) > self.burn_in]
        if not segments:
            raise ValueError("No contiguous training segments support this model")
        self.coefficients = []
        for node in range(self.n_targets):
            equations = [self._equation_data(s, node) for s in segments]
            design = np.concatenate([item[0] for item in equations])
            target = np.concatenate([item[1] for item in equations])
            self.coefficients.append(_ridge_solution(design, target, self.alpha))
        radius = self.spectral_radius()
        if radius >= 1.0:
            raise UnstableModelError(f"AR dynamics not stationary: spectral radius={radius:.6f}")
        self.training_summary = {
            "estimator": "regularized conditional sum of squares",
            "joint": True, "p_max": max(self.lag_indices),
            "lag_indices": self.lag_indices, "d": 0,
            "q": 0,
            "weather_lag_hours": self.pred_length,
            "training_segment_lengths": [len(s) for s in segments],
            "conditional_observations": sum(len(s)-self.burn_in for s in segments),
            "excluded_short_segment_rows": sum(len(s) for s in training_segments if len(s)<=self.burn_in),
            "spectral_radius": radius,
        }
        residual_blocks = []
        for segment in segments:
            columns = []
            for node in range(self.n_targets):
                design, target = self._equation_data(segment, node)
                columns.append(target - design @ self.coefficients[node])
            residual_blocks.append(np.column_stack(columns))
        residuals = np.vstack(residual_blocks)
        covariance = residuals.T @ residuals / len(residuals)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(logdet):
            raise ValueError("Residual covariance is not positive definite")
        coefficient_parameters = self.total_parameters()
        covariance_parameters = self.n_targets * (self.n_targets + 1) // 2
        bic_parameters = coefficient_parameters + covariance_parameters
        self.training_summary["conditional_sse"] = float(np.square(residuals).sum())
        self.training_summary["residual_logdet"] = float(logdet)
        self.training_summary["bic_parameter_count"] = bic_parameters
        self.training_summary["bic"] = float(
            len(residuals) * logdet + bic_parameters * np.log(len(residuals)))
        return self

    def spectral_radius(self):
        n, p = self.n_targets, max(self.lag_indices)
        companion = np.zeros((n*p, n*p))
        for node, beta in enumerate(self.coefficients):
            for position, lag in enumerate(self.lag_indices):
                start = 1 + position * n
                companion[node, (lag-1)*n:lag*n] = beta[start:start+n]
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
        z = y.copy()
        weather = np.asarray(X[:, :, self.n_targets:], dtype=np.float64)
        history = [z[:,i].copy() for i in range(length)]
        levels = y[:,-1].copy()
        forecasts = []
        for lead in range(horizon):
            step = length + lead
            ar_all = np.stack([history[-lag] for lag in self.lag_indices], axis=1)
            next_z = np.empty((batch,self.n_targets))
            for node,beta in enumerate(self.coefficients):
                ar = ar_all.reshape(batch,-1)
                # step-H never exceeds length-1: no future weather is accessed.
                design = np.column_stack([np.ones(batch), ar, weather[:,step-horizon]])
                next_z[:,node] = design @ beta
            history.append(next_z)
            levels = next_z
            forecasts.append(levels.copy())
        return np.stack(forecasts,axis=1).astype(np.float32)

    def total_parameters(self):
        return sum(len(beta) for beta in self.coefficients)
