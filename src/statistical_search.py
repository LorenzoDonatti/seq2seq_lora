"""Validation-only paired statistical search, including explicit failed candidates."""
import time
from src.data_loader import get_prepared_datasets
from src.metrics import calculate_metrics
from src.model_registry import STATISTICAL_MODELS, make_statistical
from src.models.baselines import UnstableModelError
from src.hyperparameter_search import save_search as save_statistical_search


def search_statistical_models(data_file="data/combined_hourly_data.csv", history=24,
                              horizon=1, checkpoint_path=None):
    data = get_prepared_datasets(data_file, history, horizon)
    pipeline = data["pipeline"]
    truth = pipeline.inverse_transform_targets(data["val"][1])
    rows = []
    for name in STATISTICAL_MODELS:
        differences = (0,1) if name in ("ARIMAX","VARIMAX") else (0,)
        for lag in (1,3,6):
            if history <= max(horizon,lag+max(differences)):
                continue
            for difference in differences:
                for alpha in (0.01,1.0,10.0):
                    cfg = dict(lags=lag, alpha=alpha, difference=difference)
                    start = time.perf_counter()
                    model = make_statistical(name,cfg,data,horizon)
                    try:
                        model.fit(*data["train"], training_segments=data["training_segments"])
                        prediction = pipeline.inverse_transform_targets(model.predict(data["val"][0]))
                        metrics = calculate_metrics(truth,prediction,data["target_names"])
                        row = dict(model=name,config=cfg,status="ok",
                                   val_mae_dbm=metrics["global"]["mae_dbm"],
                                   val_rmse_dbm=metrics["global"]["rmse_dbm"],
                                   per_node=metrics["per_node"], parameters=model.total_parameters(),
                                   training=model.training_summary)
                    except UnstableModelError as exc:
                        row = dict(model=name,config=cfg,status="rejected_unstable",reason=str(exc))
                    row["elapsed_seconds"] = time.perf_counter()-start
                    rows.append(row)
                    print(f"{name} H={horizon} {cfg}: {row.get('val_mae_dbm',row['status'])}",flush=True)
        if checkpoint_path is not None:
            save_statistical_search({"trials":rows,"complete":False},checkpoint_path)
        if not any(r["model"]==name and r["status"]=="ok" for r in rows):
            raise ValueError(f"No stable {name} candidate; inspect validation search report.")
    valid = sorted((r for r in rows if r["status"]=="ok"),key=lambda r:r["val_mae_dbm"])
    return {"protocol":{"selection":"validation macro MAE in dB","test_used":False,
                        "history":history,"horizon":horizon,
                        "weather_lag_hours":horizon,
                        "ma_structure":"diagonal MA(1), invertible bound +/-0.95",
                        "fit":"regularized conditional sum of squares",
                        "stability":"reject AR spectral radius >=1; no fallback"},
            "complete":True,"best":valid[0],"trials":rows}
