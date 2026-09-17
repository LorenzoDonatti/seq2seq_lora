"""Command-line interface for the LoRaWAN forecasting benchmark."""

from __future__ import annotations

import argparse
from typing import Any, Dict

from src.config_store import load_optimized_config, save_optimized_config
from src.evaluation import run_full_benchmark
from src.hyperparameter_search import search_integrated_models
from src.statistical_search import search_statistical_models


def print_single_table(results: dict, horizon: int) -> None:
    """Print accuracy by model, accuracy by node, and all-node resource cost."""
    print(f"\n{'=' * 75}")
    print(f" BENCHMARK SUMMARY (Horizon H = {horizon} hour(s))")
    print(f"{'=' * 75}")
    header = (
        f"{'Model':<24} | {'MAE (dB)':<10} | {'RMSE (dB)':<10} | "
        f"{'Params':<9} | {'Models':<6}"
    )
    print(header)
    print("-" * len(header))
    sorted_models = sorted(
        results, key=lambda name: results[name]["metrics"]["global"]["mae_db"]
    )
    for name in sorted_models:
        row = results[name]
        params = row.get("parameters", 0)
        print(
            f"{name:<24} | {row['metrics']['global']['mae_db']:<10.4f} | "
            f"{row['metrics']['global']['rmse_db']:<10.4f} | "
            f"{f'{params:,}' if params else '-':<9} | "
            f"{row.get('model_instances', 1):<6}"
        )
    print(f"{'=' * 75}\n")

    nodes = list(next(iter(results.values()))["metrics"]["per_node"])
    print("MAE POR NÓ (dB; menor é melhor)")
    node_header = f"{'Model':<24}" + "".join(f" | {node:<7}" for node in nodes) + " | média"
    print(node_header)
    print("-" * len(node_header))
    for name in sorted_models:
        values = results[name]["metrics"]["per_node"]
        columns = "".join(f" | {values[node]['mae_db']:<7.4f}" for node in nodes)
        print(f"{name:<24}{columns} | {results[name]['metrics']['global']['mae_db']:.4f}")

    print("\nCUSTO PARA ATENDER TODOS OS NÓS")
    print(
        f"{'Model':<24} | {'preditores':>10} | {'parâmetros':>12} | "
        f"{'KiB pesos':>10} | {'treino(s)':>9}"
    )
    for name in sorted_models:
        row = results[name]
        print(
            f"{name:<24} | {row.get('model_instances', 1):>10} | "
            f"{row.get('parameters', 0):>12,} | "
            f"{row.get('parameter_storage_kib', 0):>10.2f} | "
            f"{row.get('training_seconds', 0):>9.2f}"
        )


def _selected_configs(statistical: Dict[str, Any], neural: Dict[str, Any]) -> Dict[str, Any]:
    """Extract pre-test configurations from in-memory search reports."""
    selected = dict(statistical["selected_configs"])
    selected.update({name: value["best"]["config"] for name, value in neural["models"].items()})
    return selected


def _optimize_horizon(args: argparse.Namespace, horizon: int) -> Dict[str, Any]:
    """Optimize and persist every model family for one forecast horizon."""
    print(f"\n{'=' * 72}\nOtimizando horizonte H={horizon}\n{'=' * 72}\n", flush=True)
    statistical = search_statistical_models(args.data_file, args.history, horizon,
        checkpoint_path=f"{args.output_dir}/statistical_search_H{horizon}.json")
    neural = search_integrated_models(
        args.data_file, horizon, args.history, args.trials,
        args.search_epochs, args.seed, device=args.device, patience=args.patience,
        checkpoint_path=f"{args.output_dir}/neural_search_H{horizon}.json",
    )
    selected = _selected_configs(statistical, neural)
    save_optimized_config(
        selected, data_file=args.data_file, history=args.history,
        seed=args.seed, trials=args.trials, search_epochs=args.search_epochs,
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRaWAN multi-node RSSI benchmark")
    parser.add_argument("--horizon", "-H", type=int, choices=[1], default=1)
    parser.add_argument("--history", "-L", type=int, default=24)
    parser.add_argument("--epochs", "-e", type=int, default=64)
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--optimize", "--search", action="store_true",
                              help="Select and persist statistical configurations on training criteria and neural configurations on validation")
    config_group.add_argument("--use-defaults", action="store_true",
                              help="Ignore saved optimized configurations and use built-in defaults")
    parser.add_argument("--trials", type=int, default=12,
                        help="Neural configurations per model when --optimize is enabled")
    parser.add_argument("--search-epochs", type=int, default=None,
                        help="Training epochs per neural configuration during optimization")
    parser.add_argument("--data-file", "--data_file", "-d", default="data/combined_hourly_data.csv")
    parser.add_argument("--output-dir", "--output_dir", "-o", default="benchmark_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--graph-ablations", action="store_true",
                        help="Refit graph controls: no edges, physical only, adaptive only")
    args = parser.parse_args()
    if args.search_epochs is None:
        args.search_epochs = args.epochs
    if min(args.history, args.epochs, args.search_epochs, args.trials, args.patience) < 1:
        parser.error("history, epochs, search-epochs, trials and patience must be positive")
    from src.runtime import resolve_device
    try:
        args.device = resolve_device(args.device)
    except RuntimeError as error:
        parser.error(str(error))
    from pathlib import Path
    if args.history < 8:
        parser.error("history must be at least 8 hours")
    if (Path(args.output_dir) / f"benchmark_summary_H{args.horizon}.json").exists():
        parser.error("Results exist: choose a new --output-dir")
    if (Path(args.output_dir) / "run.log").exists():
        parser.error("Run log exists: choose a new --output-dir")
    selected_configs = None
    config_source = "built_in_defaults"
    from src.runtime import start_run_log
    if args.optimize:
        start_run_log(args.output_dir, vars(args))
        selected_configs = _optimize_horizon(args, args.horizon)
        config_source = "fresh_optimization"
        print("\nOtimização concluída; executando benchmark final...\n", flush=True)
    elif not args.use_defaults:
        try:
            selected_configs = load_optimized_config(
                data_file=args.data_file, history=args.history,
            )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        start_run_log(args.output_dir, vars(args))
        config_source = "last_saved_optimization"
        print(
            "\nReutilizando configurações da última otimização salva em "
            ".lora_benchmark/last_optimized_configs.json.\n",
            flush=True,
        )
    else:
        start_run_log(args.output_dir, vars(args))
        print("\nUsando configurações padrão por solicitação explícita.\n", flush=True)

    results = run_full_benchmark(
        data_file=args.data_file, seq_length=args.history, pred_length=args.horizon,
        epochs=args.epochs, output_dir=args.output_dir, seed=args.seed,
        selected_configs=selected_configs, config_source=config_source,
        device=args.device, patience=args.patience, graph_ablations=args.graph_ablations,
    )
    print_single_table(results, args.horizon)


if __name__ == "__main__":
    main()
