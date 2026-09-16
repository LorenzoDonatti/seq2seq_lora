"""Persistent storage for the latest validation-selected model configurations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


DEFAULT_CONFIG_STORE = Path(".lora_benchmark/last_optimized_configs.json")


def _data_identity(path: str) -> str:
    data_path = Path(path).expanduser()
    if data_path.is_file():
        digest = hashlib.sha256()
        with data_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    return f"missing:{data_path.name}"


def save_optimized_configs(
    configs_by_horizon: Dict[int, Dict[str, Any]],
    *,
    data_file: str,
    history: int,
    seed: int,
    trials: int,
    search_epochs: int,
    path: Path = DEFAULT_CONFIG_STORE,
) -> None:
    """Merge newly optimized horizons into the project-local configuration store."""
    payload: Dict[str, Any] = {"version": 1, "horizons": {}}
    if path.exists():
        with path.open(encoding="utf-8") as file:
            existing = json.load(file)
        if existing.get("version") == 1:
            payload = existing

    optimized_at = datetime.now(timezone.utc).isoformat()
    for horizon, configs in configs_by_horizon.items():
        payload["horizons"][str(horizon)] = {
            "configs": configs,
            "data_file": _data_identity(data_file),
            "history": history,
            "seed": seed,
            "trials": trials,
            "search_epochs": search_epochs,
            "optimized_at": optimized_at,
        }
    payload["updated_at"] = optimized_at
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    temporary.replace(path)


def load_optimized_configs(
    horizons: Iterable[int],
    *,
    data_file: str,
    history: int,
    path: Path = DEFAULT_CONFIG_STORE,
) -> Dict[int, Dict[str, Any]]:
    """Load compatible optimized configurations or raise an actionable error."""
    requested = list(horizons)
    if not path.exists():
        raise FileNotFoundError(
            f"Nenhuma configuração otimizada foi salva em {path}. "
            "Execute primeiro com --optimize ou use --use-defaults explicitamente."
        )
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    stored = payload.get("horizons", {})
    missing = [horizon for horizon in requested if str(horizon) not in stored]
    if missing:
        raise ValueError(
            f"Não há configuração otimizada salva para H={missing}. "
            "Execute esses horizontes com --optimize."
        )

    expected_data = _data_identity(data_file)
    incompatible = []
    for horizon in requested:
        entry = stored[str(horizon)]
        if entry.get("history") != history or entry.get("data_file") != expected_data:
            incompatible.append(horizon)
    if incompatible:
        raise ValueError(
            f"As configurações de H={incompatible} foram otimizadas para outro dataset "
            "ou tamanho de histórico. Execute novamente com --optimize."
        )
    return {horizon: stored[str(horizon)]["configs"] for horizon in requested}
