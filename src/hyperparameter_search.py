"""Small, reproducible validation-only search for integrated RSSI models.

The test set is deliberately not touched here.  The returned configuration is
intended to be refit (if desired) before the final benchmark is run.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List

import numpy as np
import torch

from src.data_loader import get_prepared_datasets
from src.metrics import calculate_metrics
from src.models import (
    MultiNodeSeq2SeqTrainer, AdaptiveSTGNNTrainer,
    EnsembleSingleNodeLSTM, NLinearTrainer,
)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make(model_name: str, cfg: Dict[str, Any], data: Dict[str, Any], horizon: int):
    X_train, _ = data["train"]
    common = dict(in_features=X_train.shape[-1], n_targets=len(data["target_names"]),
                  pred_length=horizon, lr=cfg["lr"], dropout=cfg["dropout"],
                  weight_decay=cfg["weight_decay"])
    if model_name == "MultiNode_Seq2Seq":
        return MultiNodeSeq2SeqTrainer(**common, hidden_dim=cfg["hidden_dim"],
                                       num_layers=cfg["num_layers"])
    if model_name == "PhysicalAdaptive_STGNN":
        return AdaptiveSTGNNTrainer(n_targets=len(data["target_names"]),
                                    n_exogenous=len(data["pipeline"].exogenous_cols),
                                    seq_length=X_train.shape[1], pred_length=horizon,
                                    hidden_dim=cfg["hidden_dim"], num_blocks=cfg["blocks"],
                                    lr=cfg["lr"], dropout=cfg["dropout"],
                                    weight_decay=cfg["weight_decay"])
    if model_name == "SingleNode_LSTM":
        return EnsembleSingleNodeLSTM(n_targets=len(data["target_names"]),
                                      pred_length=horizon, hidden_dim=cfg["hidden_dim"],
                                      num_layers=cfg["num_layers"], lr=cfg["lr"],
                                      dropout=cfg["dropout"],
                                      weight_decay=cfg["weight_decay"])
    if model_name == "NLinear":
        return NLinearTrainer(seq_len=X_train.shape[1], pred_len=horizon,
                              in_features=X_train.shape[-1],
                              n_targets=len(data["target_names"]), lr=cfg["lr"],
                              weight_decay=cfg["weight_decay"])
    raise ValueError(f"Unknown integrated model: {model_name}")


def search_integrated_models(data_file: str, horizon: int = 6, history: int = 24,
                             trials: int = 12, epochs: int = 10, seed: int = 42,
                             models: List[str] | None = None) -> Dict[str, Any]:
    """Random search selected by validation MAE in physical dBm units."""
    common = [{"lr": lr, "dropout": d, "weight_decay": wd, "batch_size": bs}
              for lr in (1e-4, 3e-4, 1e-3, 3e-3)
              for d in (0.0, 0.1, 0.3) for wd in (0.0, 1e-4, 1e-3)
              for bs in (16, 32, 64, 128)]
    spaces = {
        "MultiNode_Seq2Seq": [{**c, "hidden_dim": h, "num_layers": l}
                              for c in common for h in (16, 32, 64, 128) for l in (1, 2)],
        "PhysicalAdaptive_STGNN": [{**c, "hidden_dim": h, "blocks": b}
                                    for c in common for h in (16, 24, 32, 64) for b in (1, 2, 3)],
        "SingleNode_LSTM": [{**c, "hidden_dim": h, "num_layers": l}
                             for c in common for h in (16, 24, 32, 64) for l in (1, 2)],
        # Dropout has no role in a single linear layer; remove duplicate configs.
        "NLinear": [{"lr": lr, "weight_decay": wd, "batch_size": bs, "dropout": 0.0}
                    for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
                    for wd in (0.0, 1e-4, 1e-3) for bs in (16, 32, 64, 128)],
    }
    selected = models or list(spaces)
    data = get_prepared_datasets(data_file, history, horizon)
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    pipeline = data["pipeline"]
    result: Dict[str, Any] = {"protocol": {"selection": "validation MAE", "test_used": False,
        "seed": seed, "trials_per_model": trials, "epochs_per_trial": epochs,
        "history": history, "horizon": horizon, "search_space": spaces}, "models": {}}
    for name in selected:
        candidates = spaces[name][:]
        random.Random(seed).shuffle(candidates)
        rows = []
        total = min(trials, len(candidates))
        print(f"\n[{name}] {total} trials | history={history} | horizon={horizon} | epochs={epochs}", flush=True)
        for i, cfg in enumerate(candidates[:trials]):
            # A common seed makes configurations comparable; stochastic
            # robustness across repeated seeds is a separate experiment.
            trial_seed = seed
            started = time.perf_counter()
            print(f"  trial {i + 1:02d}/{total} | seed={trial_seed} | {cfg} | treinando...",
                  end="", flush=True)
            _seed(trial_seed)
            trainer = _make(name, cfg, data, horizon)
            trainer.fit(Xtr, ytr, Xva, yva, epochs=epochs,
                        batch_size=cfg.get("batch_size", 32))
            pred = trainer.predict(Xva)
            metric = calculate_metrics(pipeline.inverse_transform_targets(yva),
                                        pipeline.inverse_transform_targets(pred), data["target_names"])
            row = {"trial": i, "seed": trial_seed, "config": cfg,
                         "val_mae_dbm": metric["global"]["mae_dbm"],
                         "val_rmse_dbm": metric["global"]["rmse_dbm"],
                         "parameters": trainer.total_parameters(),
                         "training": getattr(trainer, "training_summary", {}),
                         "elapsed_seconds": round(time.perf_counter() - started, 2)}
            rows.append(row)
            best_so_far = min(r["val_mae_dbm"] for r in rows)
            print(f" concluído em {row['elapsed_seconds']:.1f}s | "
                  f"MAE={row['val_mae_dbm']:.4f} dBm | melhor={best_so_far:.4f} dBm",
                  flush=True)
        rows.sort(key=lambda r: r["val_mae_dbm"])
        result["models"][name] = {"best": rows[0], "trials": rows}
        print(f"  vencedor: MAE={rows[0]['val_mae_dbm']:.4f} dBm | {rows[0]['config']}", flush=True)
    return result


def save_search(result: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
