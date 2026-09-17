"""Persistent storage for the latest pre-test-selected model configurations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict
from src.model_registry import DEFAULTS

DEFAULT_CONFIG_STORE = Path(".lora_benchmark/last_optimized_configs.json")


def _valid_model_set(configs: Dict[str, Any]) -> bool:
    """Only configurations for the current benchmark composition are accepted."""
    return set(configs) == set(DEFAULTS)


def _data_identity(path: str) -> str:
    data_path = Path(path).expanduser()
    if data_path.is_file():
        digest = hashlib.sha256()
        with data_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        topology = data_path.with_suffix(".topology.json")
        if topology.exists():
            digest.update(b"\0topology\0")
            digest.update(topology.read_bytes())
        return f"sha256:{digest.hexdigest()}"
    return f"missing:{data_path.name}"


def save_optimized_config(
    configs: Dict[str, Any],
    *,
    data_file: str,
    history: int,
    seed: int,
    trials: int,
    search_epochs: int,
    path: Path = DEFAULT_CONFIG_STORE,
) -> None:
    """Save the latest optimized configuration for a dataset and history."""
    payload: Dict[str, Any] = {"experiments": {}}
    if path.exists():
        with path.open(encoding="utf-8") as file:
            existing = json.load(file)
        if set(existing) <= {"experiments", "updated_at"}:
            payload = existing

    optimized_at = datetime.now(timezone.utc).isoformat()
    identity = _data_identity(data_file)
    experiment = payload["experiments"].setdefault(identity, {
        "data_file": Path(data_file).name, "histories": {}
    })
    if not _valid_model_set(configs):
        raise ValueError("Optimized configurations do not match the current model set.")
    experiment["histories"][str(history)] = {
        "configs": configs,
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


def load_optimized_config(
    *,
    data_file: str,
    history: int,
    path: Path = DEFAULT_CONFIG_STORE,
) -> Dict[str, Any]:
    """Load compatible optimized configurations or raise an actionable error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Nenhuma configuração otimizada foi salva em {path}. "
            "Execute primeiro com --optimize ou use --use-defaults explicitamente."
        )
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    identity = _data_identity(data_file)
    experiment = payload.get("experiments", {}).get(identity)
    stored = (experiment or {}).get("histories", {}).get(str(history))
    if not stored:
        raise ValueError(
            "Não há configuração otimizada salva para esse dataset e histórico. "
            "Execute com --optimize."
        )
    configs = stored["configs"]
    if not _valid_model_set(configs):
        raise ValueError("Modelos salvos não correspondem ao benchmark atual; execute --optimize.")
    return configs
