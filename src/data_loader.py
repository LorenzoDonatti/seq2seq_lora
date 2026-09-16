"""
Data loading and preprocessing pipeline for LoRaWAN RSSI forecasting.

Ensures strict zero-leakage:
- Chronological train/val/test split.
- Windows containing missing values are discarded (no imputation).
- Windows must be strictly hourly and cannot cross acquisition gaps.
- Scalers fitted SOLELY on the training set.
"""

from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/emanueleg/lora-rssi/master/vineyard-2021_data/combined_hourly_data.csv"
TARGET_COLS = [f"RSSI_{i:02d}" for i in range(1, 9)]
EXOGENOUS_COLS = ["temp", "hum", "bar", "rain"]


def load_raw_data(
    file_path: Optional[str] = None,
    url: str = DEFAULT_DATA_URL
) -> pd.DataFrame:
    """Load dataset from local path or remote URL."""
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=";")
    else:
        df = pd.read_csv(url, sep=";")
        if file_path:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, sep=";", index=False)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series chronologically into train, validation, and test sets.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


class LoRaDataPipeline:
    """Handles training-only scaling while preserving missing observations."""
    def __init__(
        self,
        target_cols: List[str] = TARGET_COLS,
        exogenous_cols: List[str] = EXOGENOUS_COLS
    ):
        self.target_cols = target_cols
        self.exogenous_cols = exogenous_cols
        self.all_cols = target_cols + exogenous_cols
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame):
        """Fit one scaler per feature using observed training values only."""
        for col in self.all_cols:
            observed = train_df[[col]].dropna()
            if observed.empty:
                raise ValueError(f"Training column {col!r} has no observed values.")
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(observed.to_numpy(dtype=np.float64))
            self.scalers[col] = scaler

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale observed values and preserve missing values for window filtering."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on train data before transform.")

        df_scaled = pd.DataFrame(index=df.index)
        for col in self.all_cols:
            values = df[[col]].to_numpy(dtype=np.float64)
            valid = np.isfinite(values[:, 0])
            scaled = np.full(len(values), np.nan, dtype=np.float64)
            if valid.any():
                scaled[valid] = self.scalers[col].transform(values[valid]).ravel()
            df_scaled[col] = scaled
        return df_scaled

    def inverse_transform_targets(self, scaled_targets: np.ndarray) -> np.ndarray:
        """
        Inverse transform target predictions or ground truth back to physical dBm.
        scaled_targets shape: (batch, pred_len, n_targets) or (batch, n_targets)
        """
        orig_shape = scaled_targets.shape
        if len(orig_shape) == 2:
            scaled_targets = scaled_targets[:, np.newaxis, :]

        batch_size, pred_len, n_targets = scaled_targets.shape
        unscaled = np.zeros_like(scaled_targets, dtype=np.float32)

        for i, col in enumerate(self.target_cols):
            col_data = scaled_targets[:, :, i].reshape(-1, 1)
            unscaled[:, :, i] = self.scalers[col].inverse_transform(col_data).reshape(batch_size, pred_len)

        if len(orig_shape) == 2:
            return unscaled.squeeze(axis=1)
        return unscaled


def create_sliding_windows(
    data: np.ndarray,
    seq_length: int,
    pred_length: int,
    n_targets: int,
    timestamps: Optional[np.ndarray] = None,
    expected_frequency: np.timedelta64 = np.timedelta64(1, "h")
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates strictly causal sliding windows:
    X: [t - seq_length, ..., t - 1] (all features)
    y: [t, ..., t + pred_length - 1] (target RSSIs only)
    """
    X, y = [], []
    total_steps = len(data) - seq_length - pred_length + 1
    for i in range(max(0, total_steps)):
        window = data[i : i + seq_length + pred_length]
        if not np.isfinite(window).all():
            continue
        if timestamps is not None:
            time_window = timestamps[i : i + seq_length + pred_length]
            if not np.all(np.diff(time_window) == expected_frequency):
                continue
        X.append(window[:seq_length])
        y.append(window[seq_length:, :n_targets])

    if not X:
        return (
            np.empty((0, seq_length, data.shape[1]), dtype=np.float32),
            np.empty((0, pred_length, n_targets), dtype=np.float32),
        )
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _timestamps(df: pd.DataFrame) -> np.ndarray:
    """Parse Unix-second or textual timestamps to nanosecond datetime values."""
    values = df["timestamp"]
    if pd.api.types.is_numeric_dtype(values):
        parsed = pd.to_datetime(values, unit="s", utc=True)
    else:
        parsed = pd.to_datetime(values, utc=True)
    if parsed.isna().any():
        raise ValueError("Dataset contains invalid timestamps.")
    return parsed.to_numpy(dtype="datetime64[ns]")


def get_prepared_datasets(
    file_path: str = "data/combined_hourly_data.csv",
    seq_length: int = 24,
    pred_length: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
):
    """
    Build chronological datasets from complete, strictly hourly windows.
    """
    raw_df = load_raw_data(file_path=file_path)
    train_df, val_df, test_df = split_chronological(raw_df, train_ratio, val_ratio)

    pipeline = LoRaDataPipeline(TARGET_COLS, EXOGENOUS_COLS)
    pipeline.fit(train_df)

    train_scaled = pipeline.transform(train_df)
    val_scaled = pipeline.transform(val_df)
    test_scaled = pipeline.transform(test_df)

    n_targets = len(TARGET_COLS)
    X_train, y_train = create_sliding_windows(
        train_scaled.values, seq_length, pred_length, n_targets, _timestamps(train_df)
    )
    X_val, y_val = create_sliding_windows(
        val_scaled.values, seq_length, pred_length, n_targets, _timestamps(val_df)
    )
    X_test, y_test = create_sliding_windows(
        test_scaled.values, seq_length, pred_length, n_targets, _timestamps(test_df)
    )

    for split_name, X in (("train", X_train), ("validation", X_val), ("test", X_test)):
        if len(X) == 0:
            raise ValueError(
                f"No valid {split_name} windows remain after removing missing/non-hourly intervals."
            )

    return {
        "pipeline": pipeline,
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "target_names": TARGET_COLS,
        "feature_names": pipeline.all_cols,
        "quality_report": {
            "policy": "complete strictly-hourly windows; no imputation",
            "valid_windows": {
                "train": len(X_train), "validation": len(X_val), "test": len(X_test)
            },
            "discarded_windows": {
                "train": max(0, len(train_df) - seq_length - pred_length + 1) - len(X_train),
                "validation": max(0, len(val_df) - seq_length - pred_length + 1) - len(X_val),
                "test": max(0, len(test_df) - seq_length - pred_length + 1) - len(X_test),
            },
        }
    }
