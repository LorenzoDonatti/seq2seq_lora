"""
Statistical baseline forecasting models for LoRaWAN RSSI.

1. Naive Persistence Baseline:
   Repeats the last known observation of each node across the forecast horizon.

2. AutoRegressive (AR / ARIMA) Baseline:
   Fits an AR(p) autoregressive model per node with recursive multi-step forecasting.
   Standard statistical benchmark in time series literature.

3. Joint VAR / direct VARX:
   Fits one multivariate model for all nodes, with optional historical weather.
"""

from typing import List
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
class PersistenceModel:
    """Predicts future steps using the last observed RSSI for each node."""
    def __init__(self, n_targets: int):
        self.n_targets = n_targets

    def predict(self, X: np.ndarray, pred_length: int) -> np.ndarray:
        """
        X: (batch_size, seq_length, n_features)
        Returns: (batch_size, pred_length, n_targets)
        """
        last_step_targets = X[:, -1, :self.n_targets]
        return np.repeat(last_step_targets[:, np.newaxis, :], pred_length, axis=1)


class AutoRegressiveModel:
    """
    Fits an AR(p) model for each node on the training sequence,
    and produces recursive multi-horizon forecasts for test windows.
    """
    def __init__(self, n_targets: int = 8, lags: int = 12):
        self.n_targets = n_targets
        self.lags = lags
        self.ar_params: List[np.ndarray] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fits AR(lags) by least squares over valid forecast origins. This avoids
        joining independent contiguous segments after missing-data filtering.
        """
        self.ar_params = []
        for i in range(self.n_targets):
            effective_lags = min(self.lags, X_train.shape[1])
            histories = X_train[:, -effective_lags:, i][:, ::-1]
            design = np.column_stack([np.ones(len(histories)), histories])
            target = y_train[:, 0, i]
            params, *_ = np.linalg.lstsq(design, target, rcond=None)
            self.ar_params.append(params)
        return self

    def predict(self, X: np.ndarray, pred_length: int) -> np.ndarray:
        """
        Recursive multi-step forecast for each window in X.
        X shape: (batch_size, seq_length, n_features)
        Returns shape: (batch_size, pred_length, n_targets)
        """
        batch_size = len(X)
        out = np.zeros((batch_size, pred_length, self.n_targets), dtype=np.float32)

        for i in range(self.n_targets):
            params = self.ar_params[i]
            intercept = params[0]
            coeffs = params[1:]
            p_lags = len(coeffs)

            for b in range(batch_size):
                history = list(X[b, -p_lags:, i])
                for h in range(pred_length):
                    # val = c + sum(phi_k * y_{t-k})
                    val = intercept + np.dot(coeffs, history[::-1])
                    out[b, h, i] = val
                    history.pop(0)
                    history.append(val)

        return out

    def total_parameters(self) -> int:
        return sum(len(p) for p in self.ar_params)


def _ridge_solution(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Solve multi-output ridge regression without penalizing the intercept."""
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if alpha == 0:
        return np.linalg.lstsq(design, target, rcond=None)[0]
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


class JointVARModel:
    """One ridge-regularized VAR for all RSSI nodes.

    Each node at t depends on every node over the previous ``lags`` steps.
    Multi-step forecasts are recursive and never use future observations.
    """

    def __init__(self, n_targets: int = 8, lags: int = 12, alpha: float = 0.0):
        self.n_targets = n_targets
        self.lags = lags
        self.alpha = alpha
        self.params: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        effective_lags = min(self.lags, X_train.shape[1])
        lagged = X_train[:, -effective_lags:, :self.n_targets][:, ::-1, :]
        design = np.column_stack([np.ones(len(lagged)), lagged.reshape(len(lagged), -1)])
        self.params = _ridge_solution(design, y_train[:, 0, :self.n_targets], self.alpha)
        self.lags = effective_lags
        return self

    def predict(self, X: np.ndarray, pred_length: int) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("JointVARModel must be fitted before prediction.")
        output = np.empty((len(X), pred_length, self.n_targets), dtype=np.float32)
        for batch_index in range(len(X)):
            history = [row.copy() for row in X[batch_index, :, :self.n_targets]]
            for lead in range(pred_length):
                features = np.concatenate(history[-self.lags:][::-1])
                forecast = np.concatenate([[1.0], features]) @ self.params
                output[batch_index, lead] = forecast
                history.append(forecast)
        return output

    def total_parameters(self) -> int:
        return 0 if self.params is None else int(self.params.size)


class JointDirectVARXModel:
    """Direct joint multi-horizon VARX using only historically available inputs.

    RSSI and weather lags are projected together to every node and lead time.
    It is called VARX because weather is exogenous, but no future weather values
    are required or exposed to the model.
    """

    def __init__(self, n_targets: int = 8, lags: int = 12, pred_length: int = 1,
                 alpha: float = 0.0):
        self.n_targets = n_targets
        self.lags = lags
        self.pred_length = pred_length
        self.alpha = alpha
        self.params: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        effective_lags = min(self.lags, X_train.shape[1])
        lagged = X_train[:, -effective_lags:, :][:, ::-1, :]
        design = np.column_stack([np.ones(len(lagged)), lagged.reshape(len(lagged), -1)])
        target = y_train[:, :self.pred_length, :self.n_targets].reshape(len(y_train), -1)
        self.params = _ridge_solution(design, target, self.alpha)
        self.lags = effective_lags
        return self

    def predict(self, X: np.ndarray, pred_length: int | None = None) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("JointDirectVARXModel must be fitted before prediction.")
        requested = self.pred_length if pred_length is None else pred_length
        if requested != self.pred_length:
            raise ValueError("Prediction horizon must match the fitted direct VARX horizon.")
        lagged = X[:, -self.lags:, :][:, ::-1, :]
        design = np.column_stack([np.ones(len(lagged)), lagged.reshape(len(lagged), -1)])
        return (design @ self.params).reshape(len(X), self.pred_length, self.n_targets).astype(np.float32)

    def total_parameters(self) -> int:
        return 0 if self.params is None else int(self.params.size)


class ARIMAModel:
    """Per-node ARIMA fitted by statsmodels with vectorized ARMA inference.

    ``statsmodels`` uses a state-space parameterization whose constant cannot be
    copied directly into an ARMA recursion.  During fit, the reported long-run
    mean is converted to the equivalent intercept.  Inference then reconstructs
    residual states for all windows in parallel, avoiding one expensive
    ``statsmodels.apply`` call per node and origin.
    """
    def __init__(self, n_targets: int = 8, order=(2, 0, 1)):
        self.n_targets = n_targets
        self.order = tuple(order)
        self.params = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.params = []
        for i in range(self.n_targets):
            # Fit once on the one-step training target sequence. Parameters are
            # never re-estimated during validation/test inference.
            series = y_train[:, 0, i].astype(np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(series, order=self.order, trend="c").fit()
            names = dict(zip(model.param_names, model.params))
            ar = np.array(
                [names.get(f"ar.L{k}", 0.0) for k in range(1, self.order[0] + 1)],
                dtype=np.float64,
            )
            ma = np.array(
                [names.get(f"ma.L{k}", 0.0) for k in range(1, self.order[2] + 1)],
                dtype=np.float64,
            )
            # For d=0, statsmodels' `const` is the unconditional process mean.
            mean = float(names.get("const", 0.0))
            intercept = mean * (1.0 - ar.sum())
            self.params.append((intercept, ar, ma))
        return self

    def predict(self, X: np.ndarray, pred_length: int = 1) -> np.ndarray:
        if self.order[1] != 0:
            raise NotImplementedError("Vectorized ARIMA inference currently supports d=0 only.")
        out = np.empty((len(X), pred_length, self.n_targets), dtype=np.float32)
        for node, (intercept, ar, ma) in enumerate(self.params):
            observed = X[:, :, node].astype(np.float64)
            residuals = np.zeros_like(observed)

            # Reconstruct conditional residuals for every origin simultaneously.
            for t in range(observed.shape[1]):
                fitted = np.full(len(X), intercept, dtype=np.float64)
                for lag, coefficient in enumerate(ar, start=1):
                    if t >= lag:
                        fitted += coefficient * observed[:, t - lag]
                for lag, coefficient in enumerate(ma, start=1):
                    if t >= lag:
                        fitted += coefficient * residuals[:, t - lag]
                residuals[:, t] = observed[:, t] - fitted

            value_history = [observed[:, t] for t in range(observed.shape[1])]
            error_history = [residuals[:, t] for t in range(residuals.shape[1])]
            for lead in range(pred_length):
                forecast = np.full(len(X), intercept, dtype=np.float64)
                for lag, coefficient in enumerate(ar, start=1):
                    forecast += coefficient * value_history[-lag]
                for lag, coefficient in enumerate(ma, start=1):
                    forecast += coefficient * error_history[-lag]
                out[:, lead, node] = forecast
                value_history.append(forecast)
                # Future innovations have conditional expectation zero.
                error_history.append(np.zeros(len(X), dtype=np.float64))
        return out

    def total_parameters(self) -> int:
        p, d, q = self.order
        return self.n_targets * (1 + p + q)
