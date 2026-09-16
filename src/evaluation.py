"""
Benchmark evaluation orchestrator comparing statistical, single-node, and integrated models.
"""

from typing import Dict, Any, List
import json
import os
import random
import time
import csv
import numpy as np
import torch

from src.data_loader import get_prepared_datasets
from src.metrics import calculate_metrics, measure_inference_speed
from src.models import (
    PersistenceModel,
    AutoRegressiveModel,
    ARIMAModel,
    JointVARModel,
    JointDirectVARXModel,
    EnsembleSingleNodeLSTM,
    MultiNodeSeq2SeqTrainer,
    NLinearTrainer,
    AdaptiveSTGNNTrainer
)
from src.plotting import (
    plot_predictions_comparison,
    plot_benchmark_metrics,
    plot_node_metrics,
    plot_multi_horizon_degradation,
    plot_learned_adjacency_heatmap
)


def run_full_benchmark(
    data_file: str = "data/combined_hourly_data.csv",
    seq_length: int = 24,
    pred_length: int = 1,
    epochs: int = 128,
    output_dir: str = "benchmark_results",
    seed: int = 42,
    selected_configs: Dict[str, Any] = None,
    config_source: str = "built_in_defaults",
) -> Dict[str, Any]:
    """
    Runs benchmark for a specific forecast horizon:
    1. Naive Persistence (Last Known Value)
    2. AutoRegressive AR(12) Baseline
    3. Joint VAR and direct VARX (one model for all nodes)
    4. ARIMA per node
    5. Single-Node Independent LSTMs (8 independent models)
    6. Multi-Node Seq2Seq with Temporal Attention + Residual Link
    7. Modern linear baseline: NLinear (AAAI 2023)
    8. Spatio-Temporal: physical/adaptive STGNN
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"\n=======================================================")
    print(f" LoRaWAN RSSI Forecasting Benchmark")
    print(f" History Window (L): {seq_length}h | Horizon (H): {pred_length}h")
    print(f"=======================================================\n")

    print("[1/10] Loading causal, complete-window dataset...")
    data = get_prepared_datasets(
        file_path=data_file,
        seq_length=seq_length,
        pred_length=pred_length
    )
    pipeline = data["pipeline"]
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]
    target_names = data["target_names"]
    print(f"Valid windows after quality filtering: {data['quality_report']['valid_windows']}")

    y_test_dbm = pipeline.inverse_transform_targets(y_test)

    results: Dict[str, Any] = {}
    preds_dbm: Dict[str, np.ndarray] = {}
    selected_configs = selected_configs or {}
    ar_cfg = selected_configs.get("AR", {"lags": 24})
    arima_cfg = selected_configs.get("ARIMA", {"order": [3, 0, 1]})
    var_cfg = selected_configs.get("Joint_VAR", {"lags": 6, "alpha": 0.0})
    varx_cfg = selected_configs.get("Joint_VARX", {"lags": 6, "alpha": 0.01})

    def deployment_metadata(parameters: int, model_instances: int, training_seconds: float = 0.0,
                            training: Dict[str, Any] = None,
                            bytes_per_parameter: int = 4) -> Dict[str, Any]:
        """Comparable resource indicators for forecasting all nodes together."""
        return {
            "parameters": int(parameters),
            "model_instances": int(model_instances),
            "joint_multi_node_model": bool(model_instances == 1 and parameters > 0),
            "parameters_per_node": round(parameters / len(target_names), 2),
            "parameter_storage_kib": round(parameters * bytes_per_parameter / 1024, 3),
            "training_seconds": round(training_seconds, 3),
            "training": training or {},
        }

    def timed_fit(trainer, *args, **kwargs) -> float:
        started = time.perf_counter()
        trainer.fit(*args, **kwargs)
        return time.perf_counter() - started

    def reset_model_seed() -> None:
        """Give every neural family the declared benchmark seed independently."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 1. Baseline: Persistence (Naive)
    print("[2/10] Evaluating Baseline 1: Naive Persistence...")
    naive = PersistenceModel(n_targets=len(target_names))
    lat_naive = measure_inference_speed(lambda x: naive.predict(x, pred_length), X_test[:32])
    y_pred_naive_scaled = naive.predict(X_test, pred_length)
    y_pred_naive_dbm = pipeline.inverse_transform_targets(y_pred_naive_scaled)
    preds_dbm["Persistence"] = y_pred_naive_dbm
    results["Persistence"] = {
        "metrics": calculate_metrics(y_test_dbm, y_pred_naive_dbm, target_names),
        **deployment_metadata(0, 1),
        "latency_ms": round(lat_naive, 3),
        "type": "baseline_persistence"
    }

    # 2. Baseline: AutoRegressive AR(12)
    print(f"[3/10] Fitting Baseline 2: AutoRegressive AR({ar_cfg['lags']})...")
    ar_model = AutoRegressiveModel(n_targets=len(target_names), lags=ar_cfg["lags"])
    ar_started = time.perf_counter()
    ar_model.fit(X_train, y_train)
    ar_train_s = time.perf_counter() - ar_started
    lat_ar = measure_inference_speed(lambda x: ar_model.predict(x, pred_length), X_test[:32])
    y_pred_ar_scaled = ar_model.predict(X_test, pred_length)
    y_pred_ar_dbm = pipeline.inverse_transform_targets(y_pred_ar_scaled)
    preds_dbm["AR_Baseline"] = y_pred_ar_dbm
    results["AR_Baseline"] = {
        "metrics": calculate_metrics(y_test_dbm, y_pred_ar_dbm, target_names),
        **deployment_metadata(ar_model.total_parameters(), len(target_names), ar_train_s,
                              bytes_per_parameter=8),
        "latency_ms": round(lat_ar, 3),
        "type": "baseline_autoregressive"
    }

    # 3. Joint VAR: one statistical model with cross-node lag coefficients.
    print(f"[4/10] Fitting Joint VAR({var_cfg['lags']})...")
    var_model = JointVARModel(len(target_names), var_cfg["lags"], var_cfg["alpha"])
    var_started = time.perf_counter()
    var_model.fit(X_train, y_train)
    var_train_s = time.perf_counter() - var_started
    lat_var = measure_inference_speed(lambda x: var_model.predict(x, pred_length), X_test[:32])
    y_pred_var_dbm = pipeline.inverse_transform_targets(var_model.predict(X_test, pred_length))
    preds_dbm["Joint_VAR"] = y_pred_var_dbm
    results["Joint_VAR"] = {
        "metrics": calculate_metrics(y_test_dbm, y_pred_var_dbm, target_names),
        **deployment_metadata(var_model.total_parameters(), 1, var_train_s,
                              bytes_per_parameter=8),
        "latency_ms": round(lat_var, 3),
        "type": "joint_statistical_var",
        "config": var_cfg,
    }

    # 4. Direct VARX: one joint model using historical RSSI and weather.
    print(f"[5/10] Fitting direct Joint VARX({varx_cfg['lags']})...")
    varx_model = JointDirectVARXModel(
        len(target_names), varx_cfg["lags"], pred_length, varx_cfg["alpha"]
    )
    varx_started = time.perf_counter()
    varx_model.fit(X_train, y_train)
    varx_train_s = time.perf_counter() - varx_started
    lat_varx = measure_inference_speed(lambda x: varx_model.predict(x, pred_length), X_test[:32])
    y_pred_varx_dbm = pipeline.inverse_transform_targets(varx_model.predict(X_test, pred_length))
    preds_dbm["Joint_VARX"] = y_pred_varx_dbm
    results["Joint_VARX"] = {
        "metrics": calculate_metrics(y_test_dbm, y_pred_varx_dbm, target_names),
        **deployment_metadata(varx_model.total_parameters(), 1, varx_train_s,
                              bytes_per_parameter=8),
        "latency_ms": round(lat_varx, 3),
        "type": "joint_statistical_varx_historical_weather",
        "config": varx_cfg,
    }

    # 5. ARIMA, fit per node and update state at every rolling origin.
    print(f"[6/10] Evaluating Baseline 3: ARIMA{tuple(arima_cfg['order'])}...")
    arima = ARIMAModel(n_targets=len(target_names), order=tuple(arima_cfg["order"]))
    arima_started = time.perf_counter()
    arima.fit(X_train, y_train)
    arima_train_s = time.perf_counter() - arima_started
    lat_arima = measure_inference_speed(lambda x: arima.predict(x, pred_length), X_test[:32])
    y_pred_arima_scaled = arima.predict(X_test, pred_length)
    y_pred_arima_dbm = pipeline.inverse_transform_targets(y_pred_arima_scaled)
    # Keep a stable model key across horizons; the selected order is metadata.
    preds_dbm["ARIMA"] = y_pred_arima_dbm
    results["ARIMA"] = {
        "metrics": calculate_metrics(y_test_dbm, y_pred_arima_dbm, target_names),
        **deployment_metadata(arima.total_parameters(), len(target_names), arima_train_s,
                              bytes_per_parameter=8),
        "latency_ms": round(lat_arima, 3),
        "type": "baseline_arima", "order": list(arima_cfg["order"])
    }

    # 4. Single-Node Independent LSTMs (8 independent models)
    print("[7/10] Training Baseline 4: Single-Node LSTMs (8 dedicated models)...")
    single_cfg = selected_configs.get("SingleNode_LSTM", {})
    reset_model_seed()
    single_node = EnsembleSingleNodeLSTM(
        n_targets=len(target_names),
        hidden_dim=single_cfg.get("hidden_dim", 24),
        pred_length=pred_length,
        num_layers=single_cfg.get("num_layers", 1),
        lr=single_cfg.get("lr", 1e-3), dropout=single_cfg.get("dropout", 0.1),
        weight_decay=single_cfg.get("weight_decay", 1e-4),
    )
    single_train_s = timed_fit(single_node, X_train, y_train, X_val, y_val,
                               epochs=epochs, batch_size=single_cfg.get("batch_size", 32))
    lat_single = measure_inference_speed(single_node.predict, X_test[:32])
    y_pred_sn_scaled = single_node.predict(X_test)
    y_pred_sn_dbm = pipeline.inverse_transform_targets(y_pred_sn_scaled)
    preds_dbm["SingleNode_LSTM"] = y_pred_sn_dbm
    results["SingleNode_LSTM"] = {
        "metrics": calculate_metrics(y_test_dbm, y_pred_sn_dbm, target_names),
        **deployment_metadata(single_node.total_parameters(), len(target_names), single_train_s,
                              single_node.training_summary),
        "latency_ms": round(lat_single, 3),
        "type": "single_node_ensemble"
    }

    # 5. Multi-Node Seq2Seq with Temporal Attention
    print("[8/10] Training Integrated Model 1: Multi-Node Seq2Seq + Attention...")
    reset_model_seed()
    seq2seq = MultiNodeSeq2SeqTrainer(
        in_features=X_train.shape[-1],
        n_targets=len(target_names),
        pred_length=pred_length,
        hidden_dim=selected_configs.get("MultiNode_Seq2Seq", {}).get("hidden_dim", 64),
        num_layers=selected_configs.get("MultiNode_Seq2Seq", {}).get("num_layers", 2),
        lr=selected_configs.get("MultiNode_Seq2Seq", {}).get("lr", 1e-3),
        dropout=selected_configs.get("MultiNode_Seq2Seq", {}).get("dropout", 0.0),
        weight_decay=selected_configs.get("MultiNode_Seq2Seq", {}).get("weight_decay", 1e-4),
    )
    seq_cfg = selected_configs.get("MultiNode_Seq2Seq", {})
    seq_train_s = timed_fit(seq2seq, X_train, y_train, X_val, y_val, epochs=epochs,
                            batch_size=seq_cfg.get("batch_size", 32))
    lat_seq2seq = measure_inference_speed(seq2seq.predict, X_test[:32])
    y_pred_seq_scaled = seq2seq.predict(X_test)
    y_pred_seq_dbm = pipeline.inverse_transform_targets(y_pred_seq_scaled)
    preds_dbm["MultiNode_Seq2Seq"] = y_pred_seq_dbm
    results["MultiNode_Seq2Seq"] = {
        "metrics": calculate_metrics(y_test_dbm, y_pred_seq_dbm, target_names),
        **deployment_metadata(seq2seq.total_parameters(), 1, seq_train_s, seq2seq.training_summary),
        "latency_ms": round(lat_seq2seq, 3),
        "type": "integrated_recurrent"
    }

    # 6. Modern Baseline: NLinear
    print("[9/10] Training Integrated Model 2: NLinear (AAAI 2023)...")
    nlinear_cfg = selected_configs.get("NLinear", {})
    reset_model_seed()
    dlinear = NLinearTrainer(
        seq_len=seq_length,
        pred_len=pred_length,
        in_features=X_train.shape[-1],
        n_targets=len(target_names), lr=nlinear_cfg.get("lr", 1e-3),
        weight_decay=nlinear_cfg.get("weight_decay", 1e-3),
    )
    nlinear_train_s = timed_fit(dlinear, X_train, y_train, X_val, y_val, epochs=epochs,
                                batch_size=nlinear_cfg.get("batch_size", 32))
    lat_dlinear = measure_inference_speed(dlinear.predict, X_test[:32])
    y_pred_dlinear_scaled = dlinear.predict(X_test)
    y_pred_dlinear_dbm = pipeline.inverse_transform_targets(y_pred_dlinear_scaled)
    preds_dbm["NLinear"] = y_pred_dlinear_dbm
    results["NLinear"] = {
        "metrics": calculate_metrics(y_test_dbm, y_pred_dlinear_dbm, target_names),
        **deployment_metadata(dlinear.total_parameters(), 1, nlinear_train_s,
                              dlinear.training_summary),
        "latency_ms": round(lat_dlinear, 3),
        "type": "integrated_linear_decomposition"
    }

    # 7. Spatio-Temporal Graph Neural Network (Adaptive STGNN)
    print("[10/10] Training Integrated Model 3: Physical-Adaptive STGNN...")
    reset_model_seed()
    stgnn = AdaptiveSTGNNTrainer(
        n_targets=len(target_names),
        n_exogenous=len(data["pipeline"].exogenous_cols),
        seq_length=seq_length,
        pred_length=pred_length,
        hidden_dim=selected_configs.get("PhysicalAdaptive_STGNN", {}).get("hidden_dim", 16),
        num_blocks=selected_configs.get("PhysicalAdaptive_STGNN", {}).get("blocks", 1),
        lr=selected_configs.get("PhysicalAdaptive_STGNN", {}).get("lr", 1e-3),
        dropout=selected_configs.get("PhysicalAdaptive_STGNN", {}).get("dropout", 0.0),
        weight_decay=selected_configs.get("PhysicalAdaptive_STGNN", {}).get("weight_decay", 1e-4),
    )
    stgnn_cfg = selected_configs.get("PhysicalAdaptive_STGNN", {})
    stgnn_train_s = timed_fit(stgnn, X_train, y_train, X_val, y_val, epochs=epochs,
                              batch_size=stgnn_cfg.get("batch_size", 32))
    lat_stgnn = measure_inference_speed(stgnn.predict, X_test[:32])
    y_pred_stgnn_scaled = stgnn.predict(X_test)
    y_pred_stgnn_dbm = pipeline.inverse_transform_targets(y_pred_stgnn_scaled)
    preds_dbm["PhysicalAdaptive_STGNN"] = y_pred_stgnn_dbm
    results["PhysicalAdaptive_STGNN"] = {
        "metrics": calculate_metrics(y_test_dbm, y_pred_stgnn_dbm, target_names),
        **deployment_metadata(stgnn.total_parameters(), 1, stgnn_train_s, stgnn.training_summary),
        "latency_ms": round(lat_stgnn, 3),
        "type": "spatio_temporal_graph"
    }

    # Save learned graph adjacency heatmap
    adj_matrix = stgnn.get_learned_adjacency()
    plot_learned_adjacency_heatmap(
        adj_matrix, target_names,
        save_path=os.path.join(output_dir, f"physical_adaptive_graph_H{pred_length}.png")
    )

    # Generate plots
    print("Generating figures...")
    plot_predictions_comparison(
        y_test_dbm, preds_dbm, target_names,
        save_path=os.path.join(output_dir, f"predictions_comparison_H{pred_length}.png")
    )
    plot_benchmark_metrics(
        results,
        save_path=os.path.join(output_dir, f"metrics_comparison_H{pred_length}.png")
    )
    plot_node_metrics(
        results, target_names,
        save_path=os.path.join(output_dir, f"metrics_per_node_H{pred_length}.png")
    )

    # Preserve origin-level predictions for later node-wise diagnostics.
    np.savez_compressed(
        os.path.join(output_dir, f"predictions_H{pred_length}.npz"),
        y_true=y_test_dbm,
        **{name: values for name, values in preds_dbm.items()},
    )

    per_node_path = os.path.join(output_dir, f"metrics_per_node_H{pred_length}.csv")
    with open(per_node_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["model", "node", "mae_dbm", "rmse_dbm"])
        for model_name, entry in results.items():
            for node, metric in entry["metrics"]["per_node"].items():
                writer.writerow([model_name, node, metric["mae_dbm"], metric["rmse_dbm"]])

    json_path = os.path.join(output_dir, f"benchmark_summary_H{pred_length}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    protocol = {
        "seed": seed,
        "history_hours": seq_length,
        "forecast_horizon_hours": pred_length,
        "task": "rolling_origin_direct_multi_horizon",
        "evaluation": f"Each origin directly predicts t+1 through t+{pred_length} using observed history.",
        "epochs": epochs,
        "hyperparameter_source": config_source,
        "selected_hyperparameters": selected_configs or {
            "Joint_VAR": {"lags": 6, "alpha": 0.0},
            "Joint_VARX": {"lags": 6, "alpha": 0.01},
            "MultiNode_Seq2Seq": {
                "hidden_dim": 64, "num_layers": 2, "lr": 1e-3,
                "dropout": 0.0, "weight_decay": 1e-4
            },
            "PhysicalAdaptive_STGNN": {
                "hidden_dim": 16, "blocks": 1, "lr": 1e-3,
                "dropout": 0.0, "weight_decay": 1e-4
            },
            "SingleNode_LSTM": {
                "hidden_dim": 24, "num_layers": 1, "lr": 1e-3,
                "dropout": 0.1, "weight_decay": 1e-4, "batch_size": 32
            },
            "NLinear": {"lr": 1e-3, "weight_decay": 1e-3, "batch_size": 32}
        },
        "data_quality": data["quality_report"],
        "metric_semantics": {
            "global": "mean over all forecast origins, lead times, and nodes",
            "terminal_horizon": f"error specifically at t+{pred_length}",
        },
        "latency": "median milliseconds to forecast all nodes for a batch of 32 origins on CPU; not per node",
        "resource_metrics": {
            "model_instances": "number of independently parameterized predictors (they may still be packaged in one file)",
            "parameter_storage_kib": "raw numeric coefficient/weight memory estimate, excluding framework overhead",
            "training_seconds": "wall-clock fit time aggregated over the complete all-node solution",
        },
    }
    with open(os.path.join(output_dir, f"benchmark_protocol_H{pred_length}.json"), "w") as f:
        json.dump(protocol, f, indent=4)

    return results


def run_multi_horizon_benchmark(
    horizons: List[int] = [1, 6, 12, 24],
    data_file: str = "data/combined_hourly_data.csv",
    seq_length: int = 24,
    epochs: int = 128,
    output_dir: str = "benchmark_results",
    seed: int = 42,
    selected_configs_by_horizon: Dict[int, Dict[str, Any]] = None,
    config_source: str = "built_in_defaults",
) -> Dict[int, Dict[str, Any]]:
    all_horizon_results: Dict[int, Dict[str, Any]] = {}

    for h in horizons:
        res = run_full_benchmark(
            data_file=data_file,
            seq_length=seq_length,
            pred_length=h,
            epochs=epochs,
            output_dir=output_dir,
            seed=seed,
            selected_configs=(selected_configs_by_horizon or {}).get(h),
            config_source=config_source,
        )
        all_horizon_results[h] = res

    plot_multi_horizon_degradation(
        all_horizon_results,
        save_path=os.path.join(output_dir, "multi_horizon_degradation.png")
    )

    summary_file = os.path.join(output_dir, "multi_horizon_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_horizon_results, f, indent=4)

    return all_horizon_results
