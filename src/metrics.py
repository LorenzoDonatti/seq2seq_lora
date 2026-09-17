"""
Evaluation metrics tailored for LoRaWAN link-budget forecasting.

Focuses on physical telecommunication units (dBm) and system efficiency trade-offs:
- MAE (dBm)
- RMSE (dBm)
- Model parameter count
- Storage footprint
"""

from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str]
) -> Dict[str, Any]:
    """
    Computes per-node and global MAE and RMSE in physical units (dBm).

    y_true: shape (N, pred_length, n_nodes) or (N, n_nodes)
    y_pred: shape (N, pred_length, n_nodes) or (N, n_nodes)
    """
    if y_true.ndim == 2:
        y_true = y_true[:, np.newaxis, :]
        y_pred = y_pred[:, np.newaxis, :]

    n_nodes = y_true.shape[2]
    per_node = {}

    for i in range(n_nodes):
        node_name = target_names[i] if i < len(target_names) else f"Node_{i+1:02d}"
        node_true = y_true[:, :, i].flatten()
        node_pred = y_pred[:, :, i].flatten()

        mae = float(mean_absolute_error(node_true, node_pred))
        rmse = float(root_mean_squared_error(node_true, node_pred))
        per_node[node_name] = {
            "mae_db": round(mae, 4),
            "rmse_db": round(rmse, 4)
        }

    global_true = y_true.flatten()
    global_pred = y_pred.flatten()
    global_mae = float(mean_absolute_error(global_true, global_pred))
    global_rmse = float(root_mean_squared_error(global_true, global_pred))

    per_lead_time = {}
    for lead in range(y_true.shape[1]):
        lead_true = y_true[:, lead, :].flatten()
        lead_pred = y_pred[:, lead, :].flatten()
        per_lead_time[f"t+{lead + 1}"] = {
            "mae_db": round(float(mean_absolute_error(lead_true, lead_pred)), 4),
            "rmse_db": round(float(root_mean_squared_error(lead_true, lead_pred)), 4),
        }

    return {
        "global": {
            "mae_db": round(global_mae, 4),
            "rmse_db": round(global_rmse, 4)
        },
        "terminal_horizon": per_lead_time[f"t+{y_true.shape[1]}"],
        "per_lead_time": per_lead_time,
        "per_node": per_node
    }
