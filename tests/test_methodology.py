import numpy as np
import warnings
import json
from statsmodels.tsa.arima.model import ARIMA

from src.config_store import load_optimized_configs, save_optimized_configs
from src.data_loader import create_sliding_windows
from src.metrics import calculate_metrics
from src.models.stgnn import AdaptiveSTGNNTrainer
from src.models import JointVARModel, JointDirectVARXModel, ARIMAModel


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


def test_joint_var_models_predict_all_nodes_and_horizons():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 8, 5)).astype(np.float32)
    y = rng.normal(size=(40, 3, 3)).astype(np.float32)

    var = JointVARModel(n_targets=3, lags=4, alpha=0.01).fit(X, y)
    varx = JointDirectVARXModel(
        n_targets=3, lags=4, pred_length=3, alpha=0.01
    ).fit(X, y)

    for prediction in (var.predict(X[:5], 3), varx.predict(X[:5], 3)):
        assert prediction.shape == (5, 3, 3)
        assert np.isfinite(prediction).all()


def test_vectorized_arima_matches_statsmodels_state_update():
    rng = np.random.default_rng(7)
    series = np.zeros(100, dtype=np.float32)
    for index in range(1, len(series)):
        series[index] = 0.2 + 0.65 * series[index - 1] + rng.normal(0, 0.05)
    X = np.stack([series[index:index + 16] for index in range(70)])[:, :, None]
    y = np.stack([series[index + 16:index + 17] for index in range(70)])[:, :, None]

    vectorized = ARIMAModel(1, (2, 0, 1)).fit(X, y)
    actual = vectorized.predict(X[-5:], 1)[:, 0, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = ARIMA(y[:, 0, 0], order=(2, 0, 1), trend="c").fit()
        expected = np.array([
            reference.apply(history[:, 0], refit=False).forecast(1)[0]
            for history in X[-5:]
        ])

    np.testing.assert_allclose(actual, expected, atol=5e-4)


def test_last_optimized_configs_are_reused_by_horizon(tmp_path):
    store = tmp_path / "last.json"
    configs = {1: {"AR": {"lags": 12}}, 6: {"AR": {"lags": 24}}}
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
