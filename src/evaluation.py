"""All-node accuracy and resource evaluation; selection never uses test metrics."""
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from src.data_loader import get_prepared_datasets
from src.metrics import calculate_metrics
from src.model_registry import (DEFAULTS, NEURAL_MODELS, GRAPH_ABLATIONS,
                                make_neural, make_statistical, model_metadata)
from src.runtime import resolve_device, seed_everything, synchronize
from src.plotting import (plot_predictions_comparison, plot_benchmark_metrics,
                         plot_node_metrics, plot_learned_adjacency_heatmap)


def run_full_benchmark(data_file="data/combined_hourly_data.csv", seq_length=24,
                       pred_length=1, epochs=64, output_dir="benchmark_results",
                       seed=42, selected_configs=None, config_source="built_in_defaults",
                       device="auto", patience=10, graph_ablations=False):
    device = resolve_device(device)
    seed_everything(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"benchmark_summary_H{pred_length}.json").exists():
        raise FileExistsError(f"Results already exist in {out}; use a new output directory.")
    if config_source != "built_in_defaults":
        missing = set(DEFAULTS) - set(selected_configs or {})
        if missing:
            raise ValueError(f"Missing optimized models {sorted(missing)}; run --optimize.")
    configs = selected_configs if selected_configs is not None else DEFAULTS
    data = get_prepared_datasets(data_file, seq_length, pred_length)
    Xtr, ytr = data["train"]
    Xv, yv = data["val"]
    Xt, yt = data["test"]
    pipeline, names = data["pipeline"], data["target_names"]
    truth = pipeline.inverse_transform_targets(yt)
    results, predictions = {}, {}
    print(f"H={pred_length}: device={device}; windows={data['quality_report']['valid_windows']}",
          flush=True)
    models = list(DEFAULTS) + (list(GRAPH_ABLATIONS) if graph_ablations else [])
    for name in models:
        cfg = configs["PhysicalAdaptive_STGNN"] if name in GRAPH_ABLATIONS else configs[name]
        neural = name in NEURAL_MODELS or name in GRAPH_ABLATIONS
        actual_device = device if neural else "cpu"
        seed_everything(seed)
        if neural:
            model = make_neural(name, cfg, data, pred_length, device)
        else:
            model = make_statistical(name, cfg, data, pred_length)
        synchronize(actual_device)
        started = time.perf_counter()
        if neural:
            model.fit(Xtr, ytr, Xv, yv, epochs=epochs, batch_size=cfg["batch_size"],
                      patience=patience, target_scale=data["target_scale"])
        else:
            model.fit(Xtr, ytr, training_segments=data["training_segments"])
        synchronize(actual_device)
        training_seconds = time.perf_counter() - started
        predict = model.predict if neural else lambda x: model.predict(x, pred_length)
        prediction = pipeline.inverse_transform_targets(predict(Xt))
        predictions[name] = prediction
        parameters = model.total_parameters()
        metadata = model_metadata(name, len(names))
        entry = {
            "metrics": calculate_metrics(truth, prediction, names),
            "parameters": parameters, "parameters_per_node": parameters / len(names),
            **metadata,
            "parameter_storage_kib": parameters * (4 if neural else 8) / 1024,
            "training_seconds": training_seconds,
            "training": getattr(model, "training_summary", {}),
            "device": actual_device, "config": cfg,
            "ablation_control": "fixed hybrid-selected configuration, refitted" if name in GRAPH_ABLATIONS else None,
        }
        results[name] = entry
        print(f"{name}: MAE={entry['metrics']['global']['mae_db']:.4f} dB; "
              f"params={parameters}; fit={training_seconds:.1f}s", flush=True)
        if name == "PhysicalAdaptive_STGNN" or name in GRAPH_ABLATIONS:
            graph = model.graph_diagnostics()
            (out / f"{name}_graph_H{pred_length}.json").write_text(json.dumps(graph, indent=2))
            torch.save({k:v.detach().cpu() for k,v in model.model.state_dict().items()},
                       out / f"{name}_weights_H{pred_length}.pt")
            plot_learned_adjacency_heatmap(model.get_learned_adjacency(), names,
                                           str(out / f"{name}_graph_H{pred_length}.png"))
        del model

    np.savez_compressed(out / f"predictions_H{pred_length}.npz", y_true=truth,
                        origin_timestamps=data["origin_timestamps"]["test"],
                        target_names=np.asarray(names), **predictions)
    with (out / f"metrics_per_node_H{pred_length}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "node", "mae_db", "rmse_db"])
        for name, entry in results.items():
            for node, metric in entry["metrics"]["per_node"].items():
                writer.writerow([name, node, metric["mae_db"], metric["rmse_db"]])
    (out / f"benchmark_summary_H{pred_length}.json").write_text(json.dumps(results, indent=2))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    protocol = {
        "seed": seed,
        "history_hours": seq_length, "forecast_horizon_hours": pred_length,
        "epochs": epochs, "patience": patience, "graph_ablations": graph_ablations,
        "hyperparameter_source": config_source, "selected_hyperparameters": configs,
        "data_quality": data["quality_report"],
        "task": "rolling_origin_one_step_ahead",
        "forecast_strategies": {name: model_metadata(name)["forecast_strategy"] for name in models},
        "statistical_exogenous_policy": f"weather delayed by H={pred_length}; no future values or forecasts",
        "statistical_estimator": "ARIMAX: segmented exact state-space likelihood and training AICc; "
                                 "VARX: regularized conditional least squares and training BIC",
        "training_support": "statistical: complete contiguous train segments; neural: complete train forecast windows",
        "graph_ablation_policy": "fixed hybrid validation-selected hyperparameters, retrained per ablation",
        "scalers": {col: {"scale": float(sc.scale_[0]), "min": float(sc.min_[0])}
                    for col,sc in pipeline.scalers.items()},
        "training_objective": "neural models: macro MAE in dB, best validation checkpoint, early stopping; "
                              "statistical models: training-only information criteria",
        "metric_semantics": {"global": "mean across origins, leads and nodes",
                             "terminal_horizon": f"error at t+{pred_length}",
                             "error_unit": "dB"},
        "resource_metrics": "weight/coefficients bytes only, excluding buffers/framework/activations; "
                            "fit time excludes hyperparameter search; model-instance count is the primary "
                            "operational-consolidation measure",
        "data_sha256": hashlib.sha256(Path(data_file).read_bytes()).hexdigest(),
        "topology": {
            "source": data["topology"]["source"],
            "node_coordinates": data["topology"]["node_coordinates"].tolist(),
            "gateway_coordinates": data["topology"]["gateway_coordinates"].tolist(),
            "gateway_distances_m": data["topology"]["gateway_distances_m"].tolist(),
        },
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path("src").rglob("*.py"))},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform(),
                        "cuda_runtime": torch.version.cuda, "device": device,
                        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                        "torch_threads": torch.get_num_threads(), "git_revision": revision,
                        "git_dirty": dirty},
        "split_origin_ranges": {k: [str(v[0]), str(v[-1])] for k,v in data["origin_timestamps"].items()},
        "limitations": ["single deployment and seed; exploratory",
                        "complete common windows; no operational missing-node evaluation",
                        "no claim of network energy or ADR improvement"],
    }
    (out / f"benchmark_protocol_H{pred_length}.json").write_text(json.dumps(protocol, indent=2))
    plot_predictions_comparison(truth, predictions, names,
                                save_path=str(out / f"predictions_comparison_H{pred_length}.png"))
    plot_benchmark_metrics(results, str(out / f"metrics_comparison_H{pred_length}.png"))
    plot_node_metrics(results, names, str(out / f"metrics_per_node_H{pred_length}.png"))
    return results
