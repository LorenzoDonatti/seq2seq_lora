"""Persistent storage for the latest validation-selected model configurations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


PROTOCOL_VERSION = 4

DEFAULT_CONFIG_STORE = Path(".lora_benchmark/last_optimized_configs.json")


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
    payload: Dict[str, Any] = {"version": PROTOCOL_VERSION, "experiments": {}}
    if path.exists():
        with path.open(encoding="utf-8") as file:
            existing = json.load(file)
        if existing.get("version") == PROTOCOL_VERSION:
            payload = existing

    optimized_at = datetime.now(timezone.utc).isoformat()
    identity = _data_identity(data_file)
    experiment = payload["experiments"].setdefault(identity, {
        "data_file": Path(data_file).name, "histories": {}
    })
    stored = experiment["histories"].setdefault(str(history), {"horizons": {}})
    for horizon, configs in configs_by_horizon.items():
        stored["horizons"][str(horizon)] = {
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

    if payload.get("version") != PROTOCOL_VERSION:
        raise ValueError("Protocolo/modelos alterados: execute novamente com --optimize.")
    identity = _data_identity(data_file)
    experiment = payload.get("experiments", {}).get(identity)
    stored = ((experiment or {}).get("histories", {}).get(str(history), {})
              .get("horizons", {}))
    missing = [horizon for horizon in requested if str(horizon) not in stored]
    if missing:
        raise ValueError(
            f"Não há configuração otimizada salva para H={missing}. "
            "Execute esses horizontes com --optimize."
        )

    return {horizon: stored[str(horizon)]["configs"] for horizon in requested}
