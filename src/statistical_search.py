"""Validation-only model selection for AR and ARIMA baselines."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from src.data_loader import get_prepared_datasets
from src.metrics import calculate_metrics, measure_inference_speed
from src.models import (
    AutoRegressiveModel, ARIMAModel, JointVARModel, JointDirectVARXModel,
)


def search_statistical_models(
    data_file: str = "data/combined_hourly_data.csv",
    history: int = 24,
    horizon: int = 1,
) -> Dict[str, Any]:
    """Select univariate and joint statistical configurations on validation."""
    data = get_prepared_datasets(data_file, history, horizon)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    pipeline = data["pipeline"]
    y_true = pipeline.inverse_transform_targets(y_val)

    specs = [("AR", {"lags": lag}) for lag in (6, 8, 12, 16, 24)]
    specs += [
        ("ARIMA", {"order": order})
        for order in ((1, 0, 0), (2, 0, 0), (2, 0, 1), (3, 0, 1), (4, 0, 1))
    ]
    specs += [
        (name, {"lags": lags, "alpha": alpha})
        for name in ("Joint_VAR", "Joint_VARX")
        for lags in (6, 12, 24)
        for alpha in (0.0, 0.001, 0.01, 0.1, 1.0)
    ]

    rows = []
    for name, config in specs:
        started = time.perf_counter()
        if name == "AR":
            model = AutoRegressiveModel(len(data["target_names"]), config["lags"])
        elif name == "ARIMA":
            model = ARIMAModel(len(data["target_names"]), config["order"])
        elif name == "Joint_VAR":
            model = JointVARModel(len(data["target_names"]), config["lags"], config["alpha"])
        else:
            model = JointDirectVARXModel(
                len(data["target_names"]), config["lags"], horizon, config["alpha"]
            )
        model.fit(X_train, y_train)
        prediction = pipeline.inverse_transform_targets(model.predict(X_val, horizon))
        metrics = calculate_metrics(y_true, prediction, data["target_names"])
        latency = measure_inference_speed(lambda x: model.predict(x, horizon), X_val[:32])
        row = {
            "model": name,
            "config": config,
            "val_mae_dbm": metrics["global"]["mae_dbm"],
            "val_rmse_dbm": metrics["global"]["rmse_dbm"],
            "parameters": model.total_parameters(),
            "latency_ms_batch32_median": round(latency, 4),
            "fit_seconds": round(time.perf_counter() - started, 3),
        }
        rows.append(row)
        print(
            f"{name} {config}: MAE={row['val_mae_dbm']:.4f} | "
            f"latency={latency:.4f} ms",
            flush=True,
        )

    rows.sort(key=lambda row: row["val_mae_dbm"])
    return {
        "protocol": {
            "task": "rolling_origin_forecast",
            "horizon": horizon,
            "history": history,
            "selection": "validation MAE",
            "test_used": False,
            "latency_definition": (
                "median wall-clock milliseconds for 32 origins and all nodes on CPU; "
                "5 warmups and 30 runs; excludes training and preprocessing"
            ),
            "ar_lags": [6, 8, 12, 16, 24],
            "arima_orders": [list(config["order"]) for name, config in specs if name == "ARIMA"],
            "joint_lags": [6, 12, 24],
            "ridge_alphas": [0.0, 0.001, 0.01, 0.1, 1.0],
        },
        "best": rows[0],
        "trials": rows,
    }


def save_statistical_search(result: Dict[str, Any], path: str) -> None:
    """Persist a statistical search report as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
