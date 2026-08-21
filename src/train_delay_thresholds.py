from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from common import INTERIM, MODELS, REPORTS, ensure_dirs, load_config
from train_models import CATEGORICAL


THRESHOLDS = [30, 60, 120]


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
    }


def main() -> None:
    ensure_dirs()
    config = load_config()
    train = pd.read_parquet(INTERIM / "model_train_enriched.parquet").sample(
        1_200_000, random_state=config["random_seed"] + 20
    )
    test = pd.read_parquet(INTERIM / "model_test_enriched.parquet").sample(
        600_000, random_state=config["random_seed"] + 21
    )
    features = json.loads((REPORTS / "selected_features.json").read_text(encoding="utf-8"))
    best = json.loads((REPORTS / "best_hyperparameters.json").read_text(encoding="utf-8"))["ebm"]
    categorical = [feature for feature in features if feature in CATEGORICAL]
    numeric = [feature for feature in features if feature not in CATEGORICAL]
    for frame in (train, test):
        for column in categorical:
            frame[column] = frame[column].fillna("Missing").astype(str)
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    feature_types = ["nominal" if feature in categorical else "continuous" for feature in features]
    predictions = pd.DataFrame({"actual_delay_minutes": test.ArrDelay.to_numpy()})
    result, models = {}, {}
    for threshold in THRESHOLDS:
        y_train = train.ArrDelay.ge(threshold).astype(int)
        y_test = test.ArrDelay.ge(threshold).astype(int).to_numpy()
        model = ExplainableBoostingClassifier(
            feature_names=features, feature_types=feature_types, max_bins=256,
            max_interaction_bins=32, outer_bags=4, n_jobs=-1,
            random_state=config["random_seed"] + threshold, **best,
        )
        started = time.time()
        model.fit(train[features], y_train)
        probability = model.predict_proba(test[features])[:, 1]
        result[f"delay_ge_{threshold}"] = metrics(y_test, probability) | {
            "fit_minutes": (time.time() - started) / 60
        }
        predictions[f"p_delay_ge_{threshold}"] = probability
        models[threshold] = model
        print(f">={threshold} minutes: {result[f'delay_ge_{threshold}']}", flush=True)

    # Independent binary models can cross.  The cumulative minimum is the
    # least-invasive coherence correction: P(>=120) <= P(>=60) <= P(>=30).
    raw = predictions[[f"p_delay_ge_{t}" for t in THRESHOLDS]].to_numpy()
    coherent = np.minimum.accumulate(raw, axis=1)
    for index, threshold in enumerate(THRESHOLDS):
        predictions[f"p_delay_ge_{threshold}_coherent"] = coherent[:, index]

    joblib.dump({"models": models, "features": features}, MODELS / "ebm_delay_thresholds.joblib", compress=3)
    predictions.to_parquet(REPORTS / "delay_threshold_predictions.parquet", index=False)
    (REPORTS / "delay_threshold_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Score the flight selected by the primary EBM waterfall.
    waterfall = pd.read_csv(REPORTS / "ebm_waterfall_example.csv")
    example_index = int(waterfall.example_index.iloc[0])
    primary_test = pd.read_parquet(INTERIM / "model_test_enriched.parquet")
    for column in categorical:
        primary_test[column] = primary_test[column].fillna("Missing").astype(str)
    for column in numeric:
        primary_test[column] = pd.to_numeric(primary_test[column], errors="coerce")
    example = primary_test.iloc[[example_index]]
    primary_ebm = joblib.load(MODELS / "ebm.joblib")["model"]
    raw_profile = [float(primary_ebm.predict_proba(example[features])[:, 1][0])]
    raw_profile.extend(float(models[t].predict_proba(example[features])[:, 1][0]) for t in THRESHOLDS)
    coherent_profile = np.minimum.accumulate(np.asarray(raw_profile))
    profile = pd.DataFrame({
        "delay_threshold_minutes": [15] + THRESHOLDS,
        "raw_probability": raw_profile,
        "coherent_probability": coherent_profile,
        "origin_airport": example.origin_airport.iloc[0],
        "destination_airport": example.destination_airport.iloc[0],
        "scheduled_departure_hour": example.scheduled_departure_hour.iloc[0],
    })
    profile.to_csv(REPORTS / "delay_threshold_example.csv", index=False)


if __name__ == "__main__":
    main()
