"""Training-only Box--Jenkins identification and multivariate order selection."""
from __future__ import annotations

import time
import warnings

import numpy as np
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.data_loader import get_prepared_datasets
from src.metrics import calculate_metrics
from src.model_registry import make_statistical
from src.models.baselines import UnstableModelError, _fit_segmented_sarimax
from src.hyperparameter_search import save_search as save_statistical_search


def _correlogram(values, difference, nlags):
    """Return tests and correlograms after the declared Box--Jenkins transform."""
    transformed = np.diff(values, n=difference) if difference else values
    usable_lags = min(nlags, max(1, len(transformed) // 4 - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acf_values = acf(transformed, nlags=usable_lags, fft=True)
        pacf_values = pacf(transformed, nlags=usable_lags, method="ywm")
        adf_stat, adf_pvalue, *_ = adfuller(transformed, autolag="AIC")
        kpss_stat, kpss_pvalue, *_ = kpss(
            transformed, regression="c", nlags="auto")
    threshold = 1.96 / np.sqrt(len(transformed))
    acf_significant = [lag for lag in range(1, usable_lags + 1)
                       if abs(acf_values[lag]) > threshold]
    pacf_significant = [lag for lag in range(1, usable_lags + 1)
                        if abs(pacf_values[lag]) > threshold]
    return {
        "difference": difference, "observations": len(transformed),
        "confidence_threshold": float(threshold),
        "acf": [float(value) for value in acf_values],
        "pacf": [float(value) for value in pacf_values],
        "significant_acf_lags": acf_significant,
        "significant_pacf_lags": pacf_significant,
        "adf": {"statistic": float(adf_stat), "pvalue": float(adf_pvalue)},
        "kpss": {"statistic": float(kpss_stat), "pvalue": float(kpss_pvalue)},
        "i0_consensus": bool(adf_pvalue < 0.05 and kpss_pvalue > 0.05),
        # Parsimonious non-seasonal bounds are predeclared; the full 24-lag
        # correlogram is retained to expose unmodelled daily structure.
        "suggested_p": max((lag for lag in pacf_significant if lag <= 4), default=1),
        "suggested_q": max((lag for lag in acf_significant if lag <= 3), default=1),
    }


def _arimax_candidate_orders(diagnostic):
    """Small, deterministic Box--Jenkins neighbourhood around ACF/PACF hints."""
    orders = set()
    for difference in diagnostic["admissible_differences"]:
        view = diagnostic["by_difference"][str(difference)]
        p, q = view["suggested_p"], view["suggested_q"]
        proposed = {(p, q), (max(0, p-1), q), (min(4, p+1), q),
                    (p, max(0, q-1)), (p, min(3, q+1)),
                    (1, 0), (0, 1), (1, 1)}
        if difference == 1:
            proposed.add((0, 0))
        orders.update((candidate_p, difference, candidate_q)
                      for candidate_p, candidate_q in proposed
                      if difference == 1 or candidate_p or candidate_q)
    return sorted(orders)


def _series_diagnostics(segments, node, nlags=24):
    """Reproducible Box--Jenkins identification on the longest train segment."""
    segment = max(segments, key=len)
    values = np.asarray(segment[:, node], dtype=np.float64)
    views = {str(difference): _correlogram(values, difference, nlags)
             for difference in (0, 1)}
    level, first_difference = views["0"], views["1"]
    if level["adf"]["pvalue"] < 0.05:
        admissible = [0]
        decision = ("level ADF/KPSS consensus" if level["kpss"]["pvalue"] > 0.05
                    else "ADF rejects a unit root; KPSS conflict retained as a structural-instability warning")
    elif (level["adf"]["pvalue"] >= 0.05 and level["kpss"]["pvalue"] < 0.05
          and first_difference["i0_consensus"]):
        admissible = [1]
        decision = "level unit-root consensus; stationary first difference"
    elif first_difference["i0_consensus"]:
        admissible = [0, 1]
        decision = "inconclusive level tests; compare d=0 and d=1 by training AICc"
    else:
        raise ValueError("Neither levels nor first differences pass ADF/KPSS consensus")
    diagnostic = {"segment_length": len(values), "by_difference": views,
                  "admissible_differences": admissible,
                  "difference_decision": decision}
    diagnostic["candidate_orders"] = [list(order)
                                      for order in _arimax_candidate_orders(diagnostic)]
    return diagnostic


def _integration_diagnostics(data):
    """Check I(0)/I(1) evidence and the endogenous system rank on train only."""
    segment = max(data["training_segments"], key=len)
    features = []
    for column, name in enumerate(data["feature_names"]):
        values = np.asarray(segment[:, column], dtype=np.float64)
        differenced = np.diff(values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            level_adf = float(adfuller(values, autolag="AIC")[1])
            level_kpss = float(kpss(values, regression="c", nlags="auto")[1])
            diff_adf = float(adfuller(differenced, autolag="AIC")[1])
            diff_kpss = float(kpss(differenced, regression="c", nlags="auto")[1])
        features.append({
            "feature": name,
            "level": {"adf_pvalue": level_adf, "kpss_pvalue": level_kpss},
            "first_difference": {"adf_pvalue": diff_adf, "kpss_pvalue": diff_kpss},
            "level_i0_consensus": level_adf < 0.05 and level_kpss > 0.05,
            "difference_i0_consensus": diff_adf < 0.05 and diff_kpss > 0.05,
        })
    nodes = len(data["target_names"])
    johansen_sensitivity = []
    for lagged_differences in (1, 2, 3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            johansen = coint_johansen(
                segment[:, :nodes], det_order=0,
                k_ar_diff=lagged_differences)
        johansen_sensitivity.append({
            "lagged_differences": lagged_differences,
            "trace_statistics": johansen.lr1.tolist(),
            "trace_critical_values": johansen.cvt[:, 1].tolist(),
            "rank": int(np.sum(johansen.lr1 > johansen.cvt[:, 1])),
        })
    # Require the conclusion to be robust to the short lag specifications used in
    # the benchmark rather than choosing the most convenient Johansen result.
    rank = min(item["rank"] for item in johansen_sensitivity)
    return {
        "scope": "longest complete contiguous training segment",
        "segment_length": len(segment), "features": features,
        "johansen": {
            "deterministic_order": 0, "lagged_difference_sensitivity": [1, 2, 3],
            "significance": 0.05, "sensitivity_results": johansen_sensitivity,
            "rank": rank, "dimension": nodes,
            "interpretation": ("full rank: level VAR is admissible; cointegration among I(1) series is not the operative case"
                               if rank == nodes else
                               "reduced rank: investigate VECM rather than interpreting a level VAR"),
        },
    }


def _identify_arimax(data, horizon):
    segments = data["training_segments"]
    nodes = len(data["target_names"])
    diagnostics, trials, selected_orders = [], [], []
    for node, node_name in enumerate(data["target_names"]):
        diagnostic = _series_diagnostics(segments, node)
        diagnostic["node"] = node_name
        diagnostics.append(diagnostic)
        candidates = []
        candidate_orders = [tuple(order) for order in diagnostic["candidate_orders"]]
        total = len(candidate_orders)
        completed = 0
        print(f"ARIMAX {node_name}: evaluating {total} Box-Jenkins candidates...",
              flush=True)
        for order in candidate_orders:
            started = time.perf_counter()
            try:
                fit = _fit_segmented_sarimax(
                    segments, nodes, horizon, node, order)
                status = "ok" if fit["converged"] else "rejected_nonconverged"
                row = {key: value for key, value in fit.items() if key != "params"}
                row.update(model="ARIMAX", node=node_name, status=status,
                           elapsed_seconds=time.perf_counter() - started)
                if status == "ok":
                    candidates.append(row)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                row = {"model": "ARIMAX", "node": node_name,
                       "order": list(order), "status": "failed",
                       "reason": str(error),
                       "elapsed_seconds": time.perf_counter() - started}
            trials.append(row)
            completed += 1
            print(f"  ARIMAX {node_name}: {completed}/{total} order={order} "
                  f"status={row['status']} ({row['elapsed_seconds']:.1f}s)", flush=True)
        if not candidates:
            raise ValueError(f"No converged Box-Jenkins candidate for {node_name}")
        winner = min(candidates, key=lambda row: row["aicc"])
        selected_orders.append(list(winner["order"]))
        diagnostic["selected_order"] = list(winner["order"])
        diagnostic["selected_aicc"] = winner["aicc"]
        diagnostic["selected_ljung_box_min_pvalue"] = winner["ljung_box_min_pvalue"]
        print(f"ARIMAX {node_name}: order={winner['order']} AICc={winner['aicc']:.2f}",
              flush=True)
    return {"orders": selected_orders}, diagnostics, trials


def _select_multivariate(name, data, horizon):
    rows = []
    history = int(data["train"][0].shape[1])
    maximum_lag = min(24, history)
    dense = [list(range(1, maximum + 1)) for maximum in range(1, maximum_lag + 1)]
    seasonal = [lags for lags in ([1, 6, 12, 24], [1, 2, 3, 6, 12, 24])
                if max(lags) <= maximum_lag]
    lag_structures = []
    for lags in dense + seasonal:
        if lags not in lag_structures:
            lag_structures.append(lags)
    total = len(lag_structures) * 3
    completed = 0
    print(f"{name}: evaluating {len(lag_structures)} lag structures × 3 penalties "
          f"({total} candidates)...", flush=True)
    for lags in lag_structures:
        for alpha in (0.01, 1.0, 10.0):
            cfg = {"lags": lags, "alpha": alpha}
            if history < max(horizon, max(lags)):
                continue
            started = time.perf_counter()
            try:
                model = make_statistical(name, cfg, data, horizon)
                model.fit(*data["train"], training_segments=data["training_segments"])
                row = {"model": name, "config": cfg, "status": "ok",
                       "training_bic": model.training_summary["bic"],
                       "parameters": model.total_parameters(),
                       "training": model.training_summary}
            except (UnstableModelError, ValueError, RuntimeError,
                    np.linalg.LinAlgError) as error:
                row = {"model": name, "config": cfg, "status": "failed",
                       "reason": str(error)}
            row["elapsed_seconds"] = time.perf_counter() - started
            rows.append(row)
            completed += 1
            lag_label = (f"1..{lags[-1]}" if lags == list(range(1, lags[-1] + 1))
                         else "{" + ",".join(map(str, lags)) + "}")
            score = (f"BIC={row['training_bic']:.2f}"
                     if row["status"] == "ok" else f"{row['status']}")
            print(f"  {name}: {completed}/{total} lags={lag_label} "
                  f"alpha={alpha:g} {score}", flush=True)
    valid = [row for row in rows if row["status"] == "ok"]
    if not valid:
        raise ValueError(f"No stable {name} candidate; inspect the search report")
    winner = min(valid, key=lambda row: row["training_bic"])
    return winner["config"], rows


def search_statistical_models(data_file="data/combined_hourly_data.csv", history=24,
                              horizon=1, checkpoint_path=None):
    data = get_prepared_datasets(data_file, history, horizon)
    print("Statistical diagnostics: ADF/KPSS and Johansen rank...", flush=True)
    integration = _integration_diagnostics(data)
    ranks = [item["rank"] for item in
             integration["johansen"]["sensitivity_results"]]
    print(f"Johansen ranks for k_diff=1,2,3: {ranks} "
          f"(dimension={integration['johansen']['dimension']})", flush=True)
    if integration["johansen"]["rank"] != integration["johansen"]["dimension"]:
        raise ValueError(
            "Johansen diagnostics indicate reduced rank. A VECM specification must be "
            "considered before fitting a level VARX model."
        )
    selected = {}
    print("Starting per-node Box-Jenkins ARIMAX identification...", flush=True)
    selected["ARIMAX"], diagnostics, arimax_trials = _identify_arimax(data, horizon)
    trials = list(arimax_trials)
    for name in ("VARX",):
        selected[name], rows = _select_multivariate(name, data, horizon)
        trials.extend(rows)
        print(f"{name}: config={selected[name]} selected by training BIC", flush=True)

    truth = data["pipeline"].inverse_transform_targets(data["val"][1])
    validation = {}
    for name, config in selected.items():
        model = make_statistical(name, config, data, horizon)
        model.fit(*data["train"], training_segments=data["training_segments"])
        prediction = data["pipeline"].inverse_transform_targets(model.predict(data["val"][0]))
        validation[name] = calculate_metrics(truth, prediction, data["target_names"])

    result = {
        "protocol": {
            "selection": "training-only Box-Jenkins AICc per node (ARIMAX); training BIC (VARX)",
            "validation_role": "reported after identification; not used to choose statistical orders",
            "test_used": False, "history": history, "horizon": horizon,
            "weather_lag_hours": horizon,
            "arimax_fit": "exact Gaussian state-space likelihood over independent contiguous segments",
            "var_lag_search": "dense lag sets 1..p for p<=24 plus sparse seasonal sets; training BIC",
            "stability": "reject VAR spectral radius >= 1; no fallback",
        },
        "complete": True, "selected_configs": selected,
        "integration_and_cointegration": integration,
        "arimax_identification": diagnostics,
        "validation_metrics": validation, "trials": trials,
    }
    if checkpoint_path is not None:
        save_statistical_search(result, checkpoint_path)
    return result
