"""Validation-only search with the same stopping policy as final training."""
import json
import random
import time
from pathlib import Path
import numpy as np
from src.data_loader import get_prepared_datasets
from src.metrics import calculate_metrics
from src.model_registry import NEURAL_MODELS, make_neural
from src.runtime import resolve_device, seed_everything, synchronize


def search_integrated_models(data_file, horizon=6, history=24, trials=12,
                             epochs=64, seed=42, models=None, device="auto", patience=10, checkpoint_path=None):
    device = resolve_device(device)
    data = get_prepared_datasets(data_file, history, horizon)
    common = [dict(lr=lr, dropout=d, weight_decay=wd, batch_size=bs)
              for lr in (1e-4, 1e-3, 3e-3) for d in (0.0, 0.1)
              for wd in (0.0, 1e-3) for bs in (32, 128)]
    recurrent = [{**c, "hidden_dim": h, "num_layers": layers}
                 for c in common for h in (16, 32, 64) for layers in (1, 2)]
    linear = [dict(lr=lr, weight_decay=wd, batch_size=bs)
              for lr in (1e-4, 1e-3, 1e-2) for wd in (0.0, 1e-3) for bs in (32, 128)]
    spaces = {"SingleNode_Seq2Seq": recurrent, "MultiNode_Seq2Seq": recurrent,
              "SingleNode_NLinear": linear, "MultiNode_NLinear": linear,
              "SingleNode_DLinear": linear, "MultiNode_DLinear": linear,
              "PhysicalAdaptive_STGNN": [{**c, "hidden_dim": h, "blocks": b}
                for c in common for h in (16, 32, 64) for b in (1, 2, 3)]}
    result = {"protocol": {"selection": "validation macro MAE in dB", "test_used": False,
              "seed": seed, "trials_per_model": trials, "epochs_per_trial": epochs,
              "patience": patience, "history": history, "horizon": horizon,
              "device": device, "search_space": spaces}, "models": {}}
    for name in models or NEURAL_MODELS:
        candidates = spaces[name][:]
        random.Random(seed).shuffle(candidates)
        rows = []
        for i, cfg in enumerate(candidates[:trials]):
            seed_everything(seed)
            trainer = make_neural(name, cfg, data, horizon, device)
            synchronize(device)
            start = time.perf_counter()
            trainer.fit(*data["train"], *data["val"], epochs=epochs,
                        batch_size=cfg["batch_size"], patience=patience,
                        target_scale=data["target_scale"])
            prediction = trainer.predict(data["val"][0])
            metrics = calculate_metrics(data["pipeline"].inverse_transform_targets(data["val"][1]),
                        data["pipeline"].inverse_transform_targets(prediction), data["target_names"])
            synchronize(device)
            row = dict(trial=i, seed=seed, config=cfg, val_mae_db=metrics["global"]["mae_db"],
                       val_rmse_db=metrics["global"]["rmse_db"], per_node=metrics["per_node"],
                       parameters=trainer.total_parameters(), training=trainer.training_summary,
                       elapsed_seconds=round(time.perf_counter()-start, 3))
            rows.append(row)
            print(f"{name} H={horizon} trial={i+1}/{min(trials,len(candidates))}: "
                  f"val MAE={row['val_mae_db']:.4f} dB ({row['elapsed_seconds']:.1f}s)", flush=True)
            del trainer
        rows.sort(key=lambda r: r["val_mae_db"])
        result["models"][name] = {"best": rows[0], "trials": rows}
        if checkpoint_path is not None:
            save_search(result, checkpoint_path)
    return result


def save_search(result, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")

    def numpy_json(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, default=numpy_json)
    temporary.replace(target)
