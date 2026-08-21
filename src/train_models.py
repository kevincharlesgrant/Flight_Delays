from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from common import INTERIM, MODELS, REPORTS, ensure_dirs, load_config
from model_preprocessing import ExplicitSparseNumeric


TARGET = "arrival_delay_15min"
GROUPS = {
    "Flight identity": ["airline", "origin_airport", "destination_airport"],
    "Schedule timing": ["departure_month", "departure_weekday", "scheduled_departure_hour",
                        "scheduled_duration_minutes"],
    "Scheduled congestion": ["origin_hourly_scheduled_flights", "destination_daily_scheduled_flights"],
    "Federal holiday calendar": ["within_three_days_of_federal_holiday"],
    "FAA airport class": ["origin_faa_hub_category", "destination_faa_hub_category"],
    "NWS forecast hazards": [
        "origin_nws_hazard_valid_at_departure", "destination_nws_hazard_valid_at_departure",
        "origin_nws_hazard_active_at_cutoff", "destination_nws_hazard_active_at_cutoff",
        "origin_nws_winter_hazard", "destination_nws_winter_hazard",
    ],
    "Lagged network centrality": [
        "origin_network_pagerank", "destination_network_pagerank",
        "origin_network_betweenness", "destination_network_betweenness",
    ],
    "Recent delay propagation": [
        "origin_recent_delay_rate", "destination_recent_delay_rate", "route_recent_delay_rate"
    ],
}
CATEGORICAL = {
    "airline", "origin_airport", "destination_airport", "departure_month", "departure_weekday",
    "origin_faa_hub_category", "destination_faa_hub_category",
}
PROPAGATION_RATES = [
    "origin_recent_delay_rate", "destination_recent_delay_rate", "route_recent_delay_rate"
]
ALWAYS_INCLUDE = {"Flight identity", "Schedule timing"}
# Ten basis points of absolute average precision on the 2024 validation year.
SCREEN_THRESHOLD_AP = 0.00100
DISPLAY = {
    "airline": "Airline", "origin_airport": "Origin airport", "destination_airport": "Destination airport",
    "departure_month": "Departure month", "departure_weekday": "Day of week",
    "scheduled_departure_hour": "Scheduled departure hour",
    "scheduled_duration_minutes": "Scheduled gate-to-gate duration (minutes)",
    "origin_hourly_scheduled_flights": "Origin flights scheduled in the same hour",
    "destination_daily_scheduled_flights": "Destination flights scheduled that day",
    "within_three_days_of_federal_holiday": "Within three days of a federal holiday",
    "origin_faa_hub_category": "FAA origin hub class",
    "destination_faa_hub_category": "FAA destination hub class",
    "origin_nws_hazard_valid_at_departure": "Origin NWS hazard valid at departure",
    "destination_nws_hazard_valid_at_departure": "Destination NWS hazard valid at scheduled departure",
    "origin_nws_hazard_active_at_cutoff": "Origin NWS hazard active at prediction time",
    "destination_nws_hazard_active_at_cutoff": "Destination NWS hazard active at prediction time",
    "origin_nws_convective_hazard": "Origin convective hazard forecast",
    "destination_nws_convective_hazard": "Destination convective hazard forecast",
    "origin_nws_winter_hazard": "Origin winter hazard forecast",
    "destination_nws_winter_hazard": "Destination winter hazard forecast",
    "origin_network_pagerank": "Origin network PageRank (prior month)",
    "destination_network_pagerank": "Destination network PageRank (prior month)",
    "origin_network_betweenness": "Origin network betweenness (prior month)",
    "destination_network_betweenness": "Destination network betweenness (prior month)",
    "origin_recent_delay_rate": "Recent origin delay rate (days -8 to -2)",
    "destination_recent_delay_rate": "Recent destination delay rate (days -8 to -2)",
    "route_recent_delay_rate": "Recent route delay rate (days -29 to -2)",
}
EXCLUDED_CATALOG = [
    ("distance_miles", "Schedule timing", "Scheduled distance (miles)", "Excluded: strongly redundant with scheduled duration"),
    ("origin_daily_scheduled_flights", "Scheduled congestion", "Origin flights scheduled that day", "Excluded: redundant with airport and hourly volume"),
    ("federal_holiday_name", "Federal holiday calendar", "Federal holiday name", "Excluded: too sparse; use the holiday-window flag"),
    ("days_to_nearest_federal_holiday", "Federal holiday calendar", "Days to nearest federal holiday", "Excluded: redundant with holiday-window flag"),
    ("origin_faa_log_enplanements", "FAA airport class", "FAA origin annual enplanements", "Excluded: redundant with hub class and airport identity"),
    ("destination_faa_log_enplanements", "FAA airport class", "FAA destination annual enplanements", "Excluded: redundant with hub class and airport identity"),
    ("origin_network_weighted_out_share", "Lagged network centrality", "Origin share of network departures", "Excluded: volume proxy, not structural centrality"),
    ("destination_network_weighted_in_share", "Lagged network centrality", "Destination share of network arrivals", "Excluded: volume proxy, not structural centrality"),
]


def evaluate(y, probability: np.ndarray) -> dict[str, float]:
    y_array = np.asarray(y)
    probability = np.clip(np.asarray(probability), 1e-7, 1 - 1e-7)
    cutoff = np.quantile(probability, 0.9)
    top = probability >= cutoff
    base = float(np.mean(y_array))
    top_rate = float(np.mean(y_array[top]))
    return {"roc_auc": float(roc_auc_score(y_array, probability)),
            "average_precision": float(average_precision_score(y_array, probability)),
            "log_loss": float(log_loss(y_array, probability)),
            "brier_score": float(brier_score_loss(y_array, probability)), "base_rate": base,
            "top_decile_delay_rate": top_rate, "top_decile_lift": top_rate / base,
            "share_delays_captured_top_decile": float(y_array[top].sum() / y_array.sum())}


def paired_loss_improvement(y: np.ndarray, reference: np.ndarray, alternative: np.ndarray):
    """Positive means reference has lower (better) per-row log loss."""
    eps = 1e-7
    ref, alt = np.clip(reference, eps, 1 - eps), np.clip(alternative, eps, 1 - eps)
    ref_loss = -(y * np.log(ref) + (1 - y) * np.log(1 - ref))
    alt_loss = -(y * np.log(alt) + (1 - y) * np.log(1 - alt))
    delta = alt_loss - ref_loss
    mean = float(delta.mean())
    half = 1.96 * float(delta.std(ddof=1) / np.sqrt(len(delta)))
    return mean, mean - half, mean + half


def load_data():
    train = pd.read_parquet(INTERIM / "model_train_enriched.parquet")
    test = pd.read_parquet(INTERIM / "model_test_enriched.parquet")
    for frame in (train, test):
        frame[TARGET] = frame[TARGET].astype(int)
        frame["flight_date"] = pd.to_datetime(frame.flight_date)
        for feature in CATEGORICAL:
            frame[feature] = frame[feature].fillna("Missing").astype(str)
    return train, test


def split_features(features):
    return [c for c in features if c in CATEGORICAL], [c for c in features if c not in CATEGORICAL]


def preprocessor_for(features):
    """Encode categoricals but preserve numeric NaNs for XGBoost's native handling."""
    categorical, numeric = split_features(features)
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=150), categorical),
        ("numeric", ExplicitSparseNumeric(), numeric),
    ], sparse_threshold=1.0)


def prepare_logistic_features(frame, features, fallback_values=None):
    """Create complete logistic inputs with explicit missing-value flags."""
    data = frame[features].copy()
    present_rates = [f for f in PROPAGATION_RATES if f in features]
    if fallback_values is None:
        fallback_values = {}
        if present_rates:
            pooled_rate = float(pd.concat(
                [pd.to_numeric(data[f], errors="coerce") for f in present_rates], ignore_index=True
            ).mean())
            fallback_values.update({
                feature: (float(pd.to_numeric(data[feature], errors="coerce").mean())
                          if pd.to_numeric(data[feature], errors="coerce").notna().any() else pooled_rate)
                for feature in present_rates
            })
        if "scheduled_duration_minutes" in features:
            duration = pd.to_numeric(data["scheduled_duration_minutes"], errors="coerce")
            route_medians = pd.DataFrame({
                "origin": data["origin_airport"], "destination": data["destination_airport"],
                "duration": duration,
            }).groupby(["origin", "destination"]).duration.median().to_dict()
            fallback_values["scheduled_duration_global"] = float(duration.median())
            fallback_values["scheduled_duration_by_route"] = route_medians
    original = {feature: pd.to_numeric(data[feature], errors="coerce") for feature in present_rates}
    indicator_names = []
    for feature in present_rates:
        indicator = f"{feature}_history_unavailable"
        data[indicator] = original[feature].isna().astype("int8")
        indicator_names.append(indicator)

    origin = original.get("origin_recent_delay_rate")
    destination = original.get("destination_recent_delay_rate")
    route = original.get("route_recent_delay_rate")
    if route is not None:
        airport_backoff = pd.concat([x for x in (origin, destination) if x is not None], axis=1).mean(axis=1)
        data["route_recent_delay_rate"] = route.fillna(airport_backoff).fillna(
            fallback_values["route_recent_delay_rate"])
    if origin is not None:
        candidates = [x for x in (route, destination) if x is not None]
        backoff = pd.concat(candidates, axis=1).mean(axis=1) if candidates else np.nan
        data["origin_recent_delay_rate"] = origin.fillna(backoff).fillna(
            fallback_values["origin_recent_delay_rate"])
    if destination is not None:
        candidates = [x for x in (route, origin) if x is not None]
        backoff = pd.concat(candidates, axis=1).mean(axis=1) if candidates else np.nan
        data["destination_recent_delay_rate"] = destination.fillna(backoff).fillna(
            fallback_values["destination_recent_delay_rate"])

    if "scheduled_duration_minutes" in features:
        duration = pd.to_numeric(data["scheduled_duration_minutes"], errors="coerce")
        indicator = "scheduled_duration_history_unavailable"
        data[indicator] = duration.isna().astype("int8")
        indicator_names.append(indicator)
        route_keys = pd.Series(
            list(zip(data["origin_airport"], data["destination_airport"])), index=data.index
        )
        route_duration = route_keys.map(fallback_values["scheduled_duration_by_route"])
        data["scheduled_duration_minutes"] = duration.fillna(route_duration).fillna(
            fallback_values["scheduled_duration_global"]
        )
    return data, fallback_values, indicator_names


def logistic_preprocessor(features, indicator_names):
    categorical, numeric = split_features(features)
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=150), categorical),
        ("numeric", StandardScaler(with_mean=False), numeric + indicator_names),
    ], sparse_threshold=1.0)


def xgb_model(params, seed, n_estimators=500):
    merged = {"n_estimators": n_estimators, "max_depth": 6, "learning_rate": 0.05,
              "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 10,
              "reg_lambda": 2.0, "objective": "binary:logistic", "eval_metric": "logloss",
              "tree_method": "hist", "n_jobs": -1, "random_state": seed}
    merged.update(params)
    return XGBClassifier(**merged)


def fit_xgb_probability(train, valid, features, params, seed, n_estimators=250):
    prep = preprocessor_for(features)
    x_train = prep.fit_transform(train[features])
    x_valid = prep.transform(valid[features])
    model = xgb_model(params, seed, n_estimators)
    model.fit(x_train, train[TARGET])
    return model.predict_proba(x_valid)[:, 1], model, prep


def screen_feature_groups(train, seed):
    tune_train = train[train.year.le(2023)].sample(n=min(600_000, (train.year <= 2023).sum()), random_state=seed)
    valid = train[train.year.eq(2024)].sample(n=min(500_000, (train.year == 2024).sum()), random_state=seed + 1)
    all_features = [f for values in GROUPS.values() for f in values]
    full_prob, model, prep = fit_xgb_probability(tune_train, valid, all_features, {}, seed, 300)
    full_metrics = evaluate(valid[TARGET], full_prob)
    rows = []
    for group, group_features in GROUPS.items():
        if group in ALWAYS_INCLUDE:
            rows.append({"feature_group": group, "validation_ap_without_group": np.nan,
                         "validation_ap_gain": np.nan, "validation_auc_gain": np.nan,
                         "paired_log_loss_gain": np.nan, "loss_gain_ci_low": np.nan,
                         "loss_gain_ci_high": np.nan, "included": True,
                         "decision": "Required task definition"})
            continue
        reduced = [f for f in all_features if f not in group_features]
        prob, _, _ = fit_xgb_probability(tune_train, valid, reduced, {}, seed, 300)
        metrics = evaluate(valid[TARGET], prob)
        loss_gain, ci_low, ci_high = paired_loss_improvement(valid[TARGET].to_numpy(), full_prob, prob)
        ap_gain = full_metrics["average_precision"] - metrics["average_precision"]
        included = bool(ap_gain >= SCREEN_THRESHOLD_AP)
        rows.append({"feature_group": group, "validation_ap_without_group": metrics["average_precision"],
                     "validation_ap_gain": ap_gain, "validation_auc_gain": full_metrics["roc_auc"] - metrics["roc_auc"],
                     "paired_log_loss_gain": loss_gain, "loss_gain_ci_low": ci_low, "loss_gain_ci_high": ci_high,
                     "included": included, "decision": "Included" if included else "Excluded: no material validation lift"})
        print(f"screen {group}: AP gain={ap_gain:.6f}, log-loss gain={loss_gain:.6f}", flush=True)
    screen = pd.DataFrame(rows)
    screen.to_csv(REPORTS / "feature_group_screening.csv", index=False)

    encoded_valid = prep.transform(valid[all_features])
    baseline = model.predict_proba(encoded_valid)[:, 1]
    permutation_rows = []
    for feature in GROUPS["Lagged network centrality"] + GROUPS["Recent delay propagation"]:
        shuffled = valid[all_features].copy()
        shuffled[feature] = np.random.default_rng(seed).permutation(shuffled[feature].to_numpy())
        perm_prob = model.predict_proba(prep.transform(shuffled))[:, 1]
        gain, low, high = paired_loss_improvement(valid[TARGET].to_numpy(), baseline, perm_prob)
        permutation_rows.append({"feature": feature, "display_name": DISPLAY[feature],
                                 "permutation_log_loss_gain": gain, "loss_gain_ci_low": low,
                                 "loss_gain_ci_high": high, "statistically_detectable": bool(low > 0)})
    significance = pd.DataFrame(permutation_rows).sort_values("permutation_log_loss_gain", ascending=False)
    significance.to_csv(REPORTS / "centrality_propagation_significance.csv", index=False)
    included_groups = set(screen.loc[screen.included, "feature_group"]) | ALWAYS_INCLUDE
    selected = [f for group, values in GROUPS.items() if group in included_groups for f in values]
    return selected, screen, significance


def tune_models(train, features, seed):
    base_train, base_valid = train[train.year.le(2023)], train[train.year.eq(2024)]
    tune_train = base_train.sample(n=min(650_000, len(base_train)), random_state=seed)
    valid = base_valid.sample(n=min(450_000, len(base_valid)), random_state=seed + 1)
    xgb_prep = preprocessor_for(features)
    xgb_train = xgb_prep.fit_transform(tune_train[features])
    xgb_valid = xgb_prep.transform(valid[features])
    logistic_train, fallback_values, indicators = prepare_logistic_features(tune_train, features)
    logistic_valid, _, _ = prepare_logistic_features(valid, features, fallback_values)
    logistic_prep = logistic_preprocessor(features, indicators)
    logistic_x_train = logistic_prep.fit_transform(logistic_train)
    logistic_x_valid = logistic_prep.transform(logistic_valid)
    rows = []
    logistic_grid = [{"alpha": 3e-6, "penalty": "l2"}, {"alpha": 1e-5, "penalty": "l2"},
                     {"alpha": 3e-5, "penalty": "l2"},
                     {"alpha": 1e-5, "penalty": "elasticnet", "l1_ratio": 0.05}]
    for params in logistic_grid:
        model = SGDClassifier(loss="log_loss", max_iter=80, tol=5e-4, early_stopping=True,
                              validation_fraction=0.1, n_iter_no_change=5, average=True,
                              random_state=seed, **params)
        model.fit(logistic_x_train, tune_train[TARGET])
        metrics = evaluate(valid[TARGET], model.predict_proba(logistic_x_valid)[:, 1])
        rows.append({"model": "logistic_regression", "params": json.dumps(params, sort_keys=True), **metrics})
        print(f"tune logistic {params}: AP={metrics['average_precision']:.6f}", flush=True)
    xgb_grid = [
        {"max_depth": 4, "min_child_weight": 10, "learning_rate": 0.05, "reg_lambda": 2.0},
        {"max_depth": 6, "min_child_weight": 10, "learning_rate": 0.05, "reg_lambda": 2.0},
        {"max_depth": 8, "min_child_weight": 20, "learning_rate": 0.04, "reg_lambda": 3.0},
        {"max_depth": 6, "min_child_weight": 25, "learning_rate": 0.04, "reg_lambda": 5.0,
         "subsample": 0.9, "colsample_bytree": 0.9},
    ]
    for params in xgb_grid:
        model = xgb_model(params, seed, 450)
        model.fit(xgb_train, tune_train[TARGET])
        metrics = evaluate(valid[TARGET], model.predict_proba(xgb_valid)[:, 1])
        rows.append({"model": "xgboost", "params": json.dumps(params, sort_keys=True), **metrics})
        print(f"tune XGB {params}: AP={metrics['average_precision']:.6f}", flush=True)

    ebm_train = tune_train.sample(n=min(350_000, len(tune_train)), random_state=seed + 2).copy()
    ebm_valid = valid.sample(n=min(250_000, len(valid)), random_state=seed + 3).copy()
    categorical, numeric = split_features(features)
    for frame in (ebm_train, ebm_valid):
        for c in numeric:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    ebm_grid = [{"interactions": 0, "learning_rate": 0.05, "max_rounds": 900, "min_samples_leaf": 30},
                {"interactions": 5, "learning_rate": 0.04, "max_rounds": 1100, "min_samples_leaf": 30},
                {"interactions": 8, "learning_rate": 0.03, "max_rounds": 1300, "min_samples_leaf": 50}]
    feature_types = ["nominal" if f in categorical else "continuous" for f in features]
    for params in ebm_grid:
        model = ExplainableBoostingClassifier(feature_names=features, feature_types=feature_types,
                                              max_bins=256, max_interaction_bins=32, outer_bags=4,
                                              n_jobs=-1, random_state=seed, **params)
        model.fit(ebm_train[features], ebm_train[TARGET])
        metrics = evaluate(ebm_valid[TARGET], model.predict_proba(ebm_valid[features])[:, 1])
        rows.append({"model": "ebm", "params": json.dumps(params, sort_keys=True), **metrics})
        print(f"tune EBM {params}: AP={metrics['average_precision']:.6f}", flush=True)
    tuning = pd.DataFrame(rows)
    tuning.to_csv(REPORTS / "hyperparameter_tuning.csv", index=False)
    best = {}
    for name in ["logistic_regression", "xgboost", "ebm"]:
        best_row = tuning[tuning.model.eq(name)].sort_values("average_precision", ascending=False).iloc[0]
        best[name] = json.loads(best_row.params)
    (REPORTS / "best_hyperparameters.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best, tuning


def build_feature_catalog(screen, selected):
    screen_map = screen.set_index("feature_group").to_dict("index")
    rows = []
    for group, features in GROUPS.items():
        group_row = screen_map.get(group, {})
        for feature in features:
            source = ("FAA" if group == "FAA airport class" else "NOAA/NWS" if group == "NWS forecast hazards"
                      else "U.S. federal holiday calendar" if group == "Federal holiday calendar"
                      else "BTS-derived (lagged)" if group in {"Lagged network centrality", "Recent delay propagation"}
                      else "BTS schedule")
            rows.append({"feature": feature, "display_name": DISPLAY[feature], "feature_group": group,
                         "source": source, "validation_ap_gain_for_group": group_row.get("validation_ap_gain"),
                         "included": feature in selected, "decision": group_row.get("decision", "Included")})
    for feature, group, name, decision in EXCLUDED_CATALOG:
        rows.append({"feature": feature, "display_name": name, "feature_group": group,
                     "source": "Candidate / derived", "validation_ap_gain_for_group": np.nan,
                     "included": False, "decision": decision})
    pd.DataFrame(rows).to_csv(REPORTS / "feature_catalog.csv", index=False)


def train_final(train, test, features, best, seed):
    y_train, y_test = train[TARGET], test[TARGET]
    results, predictions = {}, pd.DataFrame({"actual": y_test.to_numpy(), "flight_date": test.flight_date.to_numpy()})
    started = time.time()
    logistic_train, fallback_values, indicators = prepare_logistic_features(train, features)
    logistic_test, _, _ = prepare_logistic_features(test, features, fallback_values)
    logistic_prep = logistic_preprocessor(features, indicators)
    logistic_x_train = logistic_prep.fit_transform(logistic_train)
    logistic_x_test = logistic_prep.transform(logistic_test)
    logistic = SGDClassifier(loss="log_loss", max_iter=100, tol=5e-4, early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=6, average=True,
                             random_state=seed, **best["logistic_regression"])
    logistic.fit(logistic_x_train, y_train)
    prob = logistic.predict_proba(logistic_x_test)[:, 1]
    results["logistic_regression"] = evaluate(y_test, prob) | {"fit_minutes": (time.time() - started) / 60}
    predictions["logistic_regression"] = prob
    joblib.dump({"preprocessor": logistic_prep, "model": logistic, "features": features,
                 "fallback_values": fallback_values, "missing_indicators": indicators},
                MODELS / "logistic_regression.joblib", compress=3)
    pd.DataFrame({"feature": logistic_prep.get_feature_names_out(), "coefficient": logistic.coef_[0]}).assign(
        abs_importance=lambda x: x.coefficient.abs()).sort_values("abs_importance", ascending=False).to_csv(
            REPORTS / "logistic_coefficients.csv", index=False)
    print("finished tuned logistic regression", flush=True)

    started = time.time()
    xgb_prep = preprocessor_for(features)
    xgb_train = xgb_prep.fit_transform(train[features])
    xgb_test = xgb_prep.transform(test[features])
    xgb = xgb_model(best["xgboost"], seed, 650)
    xgb.fit(xgb_train, y_train)
    prob = xgb.predict_proba(xgb_test)[:, 1]
    results["xgboost"] = evaluate(y_test, prob) | {"fit_minutes": (time.time() - started) / 60}
    predictions["xgboost"] = prob
    xgb.save_model(MODELS / "xgboost.json")
    joblib.dump({"preprocessor": xgb_prep, "features": features,
                 "numeric_missing_handling": "XGBoost native missing branches"},
                MODELS / "xgboost_preprocessor.joblib", compress=3)
    pd.DataFrame({"feature": xgb_prep.get_feature_names_out(), "importance": xgb.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(REPORTS / "xgboost_importance.csv", index=False)
    print("finished tuned XGBoost", flush=True)

    started = time.time()
    categorical, numeric = split_features(features)
    for frame in (train, test):
        for c in numeric:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    feature_types = ["nominal" if f in categorical else "continuous" for f in features]
    ebm = ExplainableBoostingClassifier(feature_names=features, feature_types=feature_types,
                                        max_bins=256, max_interaction_bins=32, outer_bags=4,
                                        n_jobs=-1, random_state=seed, **best["ebm"])
    ebm.fit(train[features], y_train)
    prob = ebm.predict_proba(test[features])[:, 1]
    results["ebm"] = evaluate(y_test, prob) | {"fit_minutes": (time.time() - started) / 60}
    predictions["ebm"] = prob
    joblib.dump({"model": ebm, "features": features}, MODELS / "ebm.joblib", compress=3)
    pd.DataFrame({"term": ebm.term_names_, "importance": ebm.term_importances()}).sort_values(
        "importance", ascending=False).to_csv(REPORTS / "ebm_term_importance.csv", index=False)
    print("finished tuned EBM", flush=True)
    predictions.to_parquet(REPORTS / "test_predictions.parquet", index=False)
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (REPORTS / "selected_features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    return results


def main():
    ensure_dirs()
    config = load_config()
    train, test = load_data()
    print(f"train={len(train):,}, test={len(test):,}; delay rates={train[TARGET].mean():.3f}/{test[TARGET].mean():.3f}", flush=True)
    selected, screen, _ = screen_feature_groups(train, config["random_seed"])
    build_feature_catalog(screen, selected)
    print(f"selected {len(selected)} features: {selected}", flush=True)
    best, _ = tune_models(train, selected, config["random_seed"])
    results = train_final(train, test, selected, best, config["random_seed"])
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
