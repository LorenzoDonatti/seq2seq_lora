"""
Publication-ready visualization routines for LoRaWAN RSSI forecasts.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions_comparison(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    target_names: List[str],
    n_samples: int = 120,
    save_path: str = "benchmark_results/predictions_comparison.png"
):
    """
    Plots ground truth vs models' predictions for each of the 8 LoRa nodes.
    y_true: (N, pred_length, n_nodes)
    predictions_dict: {model_name: (N, pred_length, n_nodes)}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_nodes = len(target_names)
    rows = (n_nodes + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 3.5 * rows), sharex=True)
    axes = axes.flatten()

    time_idx = np.arange(n_samples)

    for i in range(n_nodes):
        ax = axes[i]
        node_name = target_names[i]
        ax.plot(time_idx, y_true[:n_samples, 0, i], label="Ground Truth", color="black", linewidth=1.8)

        for model_name, preds in predictions_dict.items():
            ax.plot(time_idx, preds[:n_samples, 0, i], label=model_name, linestyle="--", alpha=0.85)

        ax.set_title(f"{node_name} RSSI Forecast", fontsize=11, fontweight="bold")
        ax.set_ylabel("RSSI (dBm)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    for j in range(n_nodes, len(axes)):
        fig.delaxes(axes[j])

    axes[-1].set_xlabel("Test Time Step (Hours)", fontsize=11)
    axes[-2].set_xlabel("Test Time Step (Hours)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_benchmark_metrics(
    results: Dict[str, Any],
    save_path: str = "benchmark_results/metrics_comparison.png"
):
    """
    Resource-oriented summary. Accuracy by node is plotted separately.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    models = list(results.keys())
    params = [results[m].get("parameters", 0) for m in models]
    latencies = [results[m].get("latency_ms", 0.0) for m in models]
    instances = [results[m].get("model_instances", 1) for m in models]
    training = [results[m].get("training_seconds", 0.0) for m in models]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    bars1 = ax1.bar(models, instances, color="#1f77b4", edgecolor="black")
    ax1.set_title("Independently parameterized predictors", fontweight="bold")
    ax1.set_ylabel("Predictors needed for all nodes")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(axis="y", linestyle=":", alpha=0.7)
    for b in bars1:
        ax1.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(models, training, color="#ff7f0e", edgecolor="black")
    ax2.set_title("Complete training time", fontweight="bold")
    ax2.set_ylabel("Seconds")
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(axis="y", linestyle=":", alpha=0.7)
    for b in bars2:
        ax2.annotate(f"{b.get_height():.2f}s", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=9)

    # 3. Model Parameters
    bars3 = ax3.bar(models, params, color="#2ca02c", edgecolor="black")
    ax3.set_title("Trainable Parameters Count", fontweight="bold")
    ax3.set_ylabel("Parameter Count")
    ax3.tick_params(axis="x", rotation=25)
    ax3.grid(axis="y", linestyle=":", alpha=0.7)
    for b in bars3:
        ax3.annotate(f"{int(b.get_height()):,}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=8)

    # 4. Inference Latency (ms)
    bars4 = ax4.bar(models, latencies, color="#d62728", edgecolor="black")
    ax4.set_title("Inference Latency per Pass (ms) - Lower is better", fontweight="bold")
    ax4.set_ylabel("Latency (ms)")
    ax4.tick_params(axis="x", rotation=25)
    ax4.grid(axis="y", linestyle=":", alpha=0.7)
    for b in bars4:
        ax4.annotate(f"{b.get_height():.2f}ms", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_node_metrics(results: Dict[str, Any], target_names: List[str],
                      save_path: str = "benchmark_results/metrics_per_node.png"):
    """Heatmaps make node-level accuracy the primary benchmark view."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    models = list(results)
    mae = np.array([[results[m]["metrics"]["per_node"][n]["mae_dbm"]
                     for n in target_names] for m in models])
    rmse = np.array([[results[m]["metrics"]["per_node"][n]["rmse_dbm"]
                      for n in target_names] for m in models])
    fig, axes = plt.subplots(1, 2, figsize=(17, max(5, 0.65 * len(models))))
    for ax, values, title in zip(axes, (mae, rmse), ("MAE by node (dBm)", "RMSE by node (dBm)")):
        im = ax.imshow(values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(target_names)), target_names, rotation=45, ha="right")
        ax.set_yticks(range(len(models)), models)
        ax.set_title(title, fontweight="bold")
        for i in range(len(models)):
            for j in range(len(target_names)):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_multi_horizon_degradation(
    multi_horizon_results: Dict[int, Dict[str, Any]],
    save_path: str = "benchmark_results/multi_horizon_degradation.png"
):
    """
    Plots error degradation across forecasting horizons (H = 1, 6, 12, 24 hours).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    horizons = sorted(multi_horizon_results.keys())
    models = list(multi_horizon_results[horizons[0]].keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    markers = ["o", "s", "^", "D", "v", "x", "p"]

    for idx, m in enumerate(models):
        mae_curve = [multi_horizon_results[h][m]["metrics"]["terminal_horizon"]["mae_dbm"] for h in horizons]
        rmse_curve = [multi_horizon_results[h][m]["metrics"]["terminal_horizon"]["rmse_dbm"] for h in horizons]
        marker = markers[idx % len(markers)]

        ax1.plot(horizons, mae_curve, marker=marker, linewidth=2, label=m)
        ax2.plot(horizons, rmse_curve, marker=marker, linewidth=2, label=m)

    ax1.set_title("Terminal-step MAE across Forecasting Horizons", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Prediction Horizon H (Hours)", fontsize=11)
    ax1.set_ylabel("MAE (dBm)", fontsize=11)
    ax1.set_xticks(horizons)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(fontsize=9)

    ax2.set_title("Terminal-step RMSE across Forecasting Horizons", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Prediction Horizon H (Hours)", fontsize=11)
    ax2.set_ylabel("RMSE (dBm)", fontsize=11)
    ax2.set_xticks(horizons)
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_learned_adjacency_heatmap(
    adj_matrix: np.ndarray,
    node_names: List[str],
    save_path: str = "benchmark_results/physical_adaptive_graph.png"
):
    """
    Plots heatmap of the learned spatial adjacency matrix between LoRa nodes.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    im = plt.imshow(adj_matrix, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, label="Spatial Correlation Weight")
    plt.title("STGNN Physical-Adaptive Graph Topology", fontsize=12, fontweight="bold")
    plt.xticks(range(len(node_names)), node_names, rotation=45)
    plt.yticks(range(len(node_names)), node_names)

    for i in range(len(node_names)):
        for j in range(len(node_names)):
            plt.text(j, i, f"{adj_matrix[i, j]:.2f}",
                     ha="center", va="center", color="black" if adj_matrix[i, j] < 0.5 else "white", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
