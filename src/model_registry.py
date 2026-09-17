"""Registry of paired statistical/neural families and graph ablations."""
from src.models.baselines import HistoricalWeatherARIMAX
from src.models.dedicated import DedicatedNodeTrainer
from src.models.multi_node_seq2seq import MultiNodeSeq2SeqTrainer
from src.models.dlinear import NLinearTrainer, DLinearTrainer
from src.models.stgnn import AdaptiveSTGNNTrainer

STATISTICAL_MODELS = ("ARX", "ARIMAX", "VARX", "VARIMAX")
NEURAL_MODELS = ("SingleNode_Seq2Seq", "MultiNode_Seq2Seq",
                 "SingleNode_NLinear", "MultiNode_NLinear",
                 "SingleNode_DLinear", "MultiNode_DLinear", "PhysicalAdaptive_STGNN")
GRAPH_ABLATIONS = {"STGNN_NoGraph": "none", "STGNN_PhysicalOnly": "physical",
                   "STGNN_AdaptiveOnly": "adaptive"}
DEFAULTS = {
    **{name: {"lags": 3, "alpha": 1.0, "difference": 0} for name in STATISTICAL_MODELS},
    **{name: {"hidden_dim": 24, "num_layers": 1, "blocks": 2,
              "lr": 0.001, "weight_decay": 0.0001, "dropout": 0.1,
              "batch_size": 32} for name in NEURAL_MODELS},
}


def model_metadata(name, nodes=8):
    local = name in ("ARX", "ARIMAX") or name.startswith("SingleNode_")
    linear = name.endswith(("NLinear", "DLinear"))
    graph = name == "PhysicalAdaptive_STGNN" or name in GRAPH_ABLATIONS
    cross = not local and not linear and name != "STGNN_NoGraph"
    return {"model_instances": nodes if local else 1,
            "joint_multi_node_model": cross, "cross_node_inputs": cross,
            "parameter_sharing": not local and name not in STATISTICAL_MODELS,
            "uses_historical_weather": not linear,
            "uses_geometry": graph,
            "integration": "dedicated" if local else ("shared_channel_independent" if linear
                           or name == "STGNN_NoGraph" else "cross_node"),
            "forecast_strategy": "recursive" if name in STATISTICAL_MODELS or "Seq2Seq" in name else "direct"}


def make_statistical(name, cfg, data, horizon):
    return HistoricalWeatherARIMAX(
        n_targets=len(data["target_names"]), pred_length=horizon,
        lags=cfg["lags"], alpha=cfg["alpha"], joint=name.startswith("V"),
        difference=cfg["difference"], moving_average=name in ("ARIMAX", "VARIMAX"))


def make_neural(name, cfg, data, horizon, device):
    features = data["train"][0].shape[-1]
    history = data["train"][0].shape[1]
    nodes = len(data["target_names"])
    local = name.startswith("SingleNode_")
    common = dict(lr=cfg["lr"], weight_decay=cfg["weight_decay"], device=device)
    if name.endswith(("NLinear", "DLinear")):
        cls = NLinearTrainer if name.endswith("NLinear") else DLinearTrainer
        kwargs = dict(seq_len=history, pred_len=horizon,
                      in_features=1 if local else features,
                      n_targets=1 if local else nodes, individual=False, **common)
        return (DedicatedNodeTrainer(cls, nodes, kwargs, include_weather=False)
                if local else cls(**kwargs))
    common.update(hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
                  n_targets=nodes, pred_length=horizon)
    if name == "PhysicalAdaptive_STGNN" or name in GRAPH_ABLATIONS:
        topology = data.get("topology", {})
        return AdaptiveSTGNNTrainer(n_exogenous=features-nodes, seq_length=history,
                                    num_blocks=cfg["blocks"],
                                    graph_mode=GRAPH_ABLATIONS.get(name, "hybrid"),
                                    node_coordinates=topology.get("node_coordinates"),
                                    gateway_coordinates=topology.get("gateway_coordinates"),
                                    gateway_distances_m=topology.get("gateway_distances_m"),
                                    **common)
    common.update(num_layers=cfg["num_layers"],
                  in_features=1+features-nodes if local else features)
    if local:
        common["n_targets"] = 1
        return DedicatedNodeTrainer(MultiNodeSeq2SeqTrainer, nodes, common)
    if name == "MultiNode_Seq2Seq":
        return MultiNodeSeq2SeqTrainer(**common)
    raise ValueError(name)
