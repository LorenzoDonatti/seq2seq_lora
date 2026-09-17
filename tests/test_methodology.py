import numpy as np
import json

from src.config_store import load_optimized_configs, save_optimized_configs
from src.data_loader import create_sliding_windows
from src.metrics import calculate_metrics
from src.models.stgnn import AdaptiveSTGNNTrainer


def test_windows_do_not_cross_missing_values_or_time_gaps():
    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    timestamps = np.arange(10).astype("timedelta64[h]") + np.datetime64("2021-01-01")
    timestamps[7:] += np.timedelta64(3, "h")
    data[3, 0] = np.nan

    X, y = create_sliding_windows(data, 2, 1, 1, timestamps)

    assert len(X) == 3
    assert np.isfinite(X).all() and np.isfinite(y).all()


def test_metrics_separate_average_and_terminal_horizon():
    truth = np.zeros((2, 3, 1), dtype=np.float32)
    pred = np.array([[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]])

    metrics = calculate_metrics(truth, pred, ["node"])

    assert metrics["global"]["mae_dbm"] == 2.0
    assert metrics["terminal_horizon"]["mae_dbm"] == 3.0
    assert metrics["per_lead_time"]["t+1"]["mae_dbm"] == 1.0


def test_physical_adaptive_graph_is_row_stochastic():
    trainer = AdaptiveSTGNNTrainer(seq_length=24, pred_length=1)
    adjacency = trainer.get_learned_adjacency()

    assert adjacency.shape == (8, 8)
    np.testing.assert_allclose(adjacency.sum(axis=1), 1.0, atol=1e-6)
    assert (adjacency >= 0).all()


def test_last_optimized_configs_are_reused_by_horizon(tmp_path):
    store = tmp_path / "last.json"
    configs = {1: {"ARX": {"lags": 12}}, 6: {"ARX": {"lags": 24}}}
    data_file = str(tmp_path / "data.csv")

    save_optimized_configs(
        configs, data_file=data_file, history=24, seed=42, trials=12,
        search_epochs=10, path=store,
    )

    assert load_optimized_configs(
        [1, 6], data_file=data_file, history=24, path=store,
    ) == configs
    payload = json.loads(store.read_text())
    assert payload["horizons"]["1"]["seed"] == 42
