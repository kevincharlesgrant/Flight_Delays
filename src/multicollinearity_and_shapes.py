from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "data" / "interim" / "matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, log_loss

from common import FIGURES, INTERIM, MODELS, REPORTS, ensure_dirs
from make_report import ACCENT_ORANGE, INK, MUTED_GRAY, pretty_term
from train_models import CATEGORICAL


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    if n == 0:
        return np.nan
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True) / n
    chi2 = np.divide((observed - expected) ** 2, expected, out=np.zeros_like(observed), where=expected > 0).sum()
    phi2 = chi2 / n
    rows, cols = observed.shape
    corrected = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / max(n - 1, 1))
    rows_c = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    cols_c = cols - ((cols - 1) ** 2) / max(n - 1, 1)
    denominator = min(cols_c - 1, rows_c - 1)
    return float(np.sqrt(corrected / denominator)) if denominator > 0 else 0.0


def association_ratio(categories: pd.Series, values: pd.Series) -> float:
    frame = pd.DataFrame({"category": categories.astype(str), "value": pd.to_numeric(values, errors="coerce")}).dropna()
    if frame.empty:
        return np.nan
    grand = frame.value.mean()
    grouped = frame.groupby("category").value.agg(["mean", "size"])
    between = (grouped["size"] * (grouped["mean"] - grand) ** 2).sum()
    total = ((frame.value - grand) ** 2).sum()
    return float(np.sqrt(between / total)) if total > 0 else 0.0


def extract_curve(explanation, ebm, term: str):
    data = explanation.data(ebm.term_names_.index(term))
    names, scores = np.asarray(data["names"]), np.asarray(data["scores"], dtype=float)
    if len(names) == len(scores) + 1:
        x = (names[:-1].astype(float) + names[1:].astype(float)) / 2
    else:
        x = names.astype(float)[:len(scores)]
    keep = np.isfinite(x) & np.isfinite(scores)
    return x[keep], scores[keep]


def shape_review(ebm, features: list[str]) -> pd.DataFrame:
    explanation = ebm.explain_global()
    recommendations = {
        "scheduled_duration_minutes": "Smooth with ~15-minute bins or max_bins~=64; do not impose global monotonicity",
        "scheduled_departure_hour": "Keep unconstrained; adjacent-hour smoothing is acceptable but the daily pattern is non-monotone",
        "origin_hourly_scheduled_flights": "Consider weak non-decreasing constraint after checking within-airport calibration",
        "destination_daily_scheduled_flights": "Smooth/coarsen; avoid a hard monotone constraint because hub identity creates plateaus",
        "origin_recent_delay_rate": "Apply a non-decreasing monotonic constraint",
        "destination_recent_delay_rate": "Apply a non-decreasing monotonic constraint",
        "route_recent_delay_rate": "Apply a non-decreasing monotonic constraint",
    }
    rows = []
    for feature in features:
        if feature in CATEGORICAL or feature not in ebm.term_names_:
            continue
        x, y = extract_curve(explanation, ebm, feature)
        if len(y) < 3:
            roughness, reversals, rho = 0.0, 0, np.nan
        else:
            first = np.diff(y)
            roughness = float(np.abs(np.diff(y, n=2)).sum() / max(np.abs(first).sum(), 1e-9))
            material = first[np.abs(first) > max(np.ptp(y) * 0.005, 1e-5)]
            reversals = int(np.sum(np.sign(material[1:]) != np.sign(material[:-1]))) if len(material) > 1 else 0
            rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        recommendation = recommendations.get(feature, "Binary indicator: no smoothing or monotonic constraint needed")
        rows.append({"feature": feature, "display_name": pretty_term(feature), "bins_or_levels": len(y),
                     "roughness_ratio": roughness, "direction_reversals": reversals,
                     "spearman_x_to_effect": rho, "production_recommendation": recommendation})
    return pd.DataFrame(rows).sort_values(["roughness_ratio", "direction_reversals"], ascending=False)


def make_shape_review_figure(ebm) -> None:
    explanation = ebm.explain_global()
    metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    reference = float(metrics["ebm"]["base_rate"])
    reference_logit = np.log(reference / (1 - reference))
    risk_points = lambda scores: (1 / (1 + np.exp(-np.clip(reference_logit + scores, -30, 30))) - reference) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    x, y = extract_curve(explanation, ebm, "scheduled_duration_minutes")
    keep = (x >= 40) & (x <= 420)
    x, y = x[keep], y[keep]
    window = min(31, len(y) if len(y) % 2 else len(y) - 1)
    smooth = savgol_filter(y, window_length=max(5, window), polyorder=2) if window >= 5 else y
    axes[0].plot(x, risk_points(y), color=MUTED_GRAY, linewidth=1.2, alpha=0.8, label="Current EBM")
    axes[0].plot(x, risk_points(smooth), color=ACCENT_ORANGE, linewidth=2.5, label="Illustrative adjacent-bin smoothing")
    axes[0].axhline(0, color=INK, linewidth=0.8)
    axes[0].set(title="Scheduled gate-to-gate duration", xlabel="Minutes",
                ylabel="Risk change (percentage points)\nfrom 22.3% reference")
    axes[0].legend(frameon=False)

    x, y = extract_curve(explanation, ebm, "route_recent_delay_rate")
    keep = (x >= 0) & (x <= 0.8)
    x, y = x[keep], y[keep]
    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(x, y)
    axes[1].plot(x, risk_points(y), color=MUTED_GRAY, linewidth=1.2, alpha=0.8, label="Current EBM")
    axes[1].plot(x, risk_points(isotonic), color=ACCENT_ORANGE, linewidth=2.5, label="Illustrative monotone fit")
    axes[1].axhline(0, color=INK, linewidth=0.8)
    axes[1].set(title="Recent route delay rate", xlabel="Delay share, days -29 to -2",
                ylabel="Risk change (percentage points)\nfrom 22.3% reference")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "ebm_shape_production_review.png", dpi=190)
    plt.close(fig)


def conditional_permutation_audit(ebm, features: list[str]) -> pd.DataFrame:
    """Measure congestion's incremental signal after preserving airport identity."""
    columns = list(dict.fromkeys(features + ["arrival_delay_15min"]))
    test = pd.read_parquet(INTERIM / "model_test_enriched.parquet", columns=columns).sample(
        250_000, random_state=321
    )
    for feature in features:
        if feature in CATEGORICAL:
            test[feature] = test[feature].fillna("Missing").astype(str)
        else:
            test[feature] = pd.to_numeric(test[feature], errors="coerce")
    y = test["arrival_delay_15min"].astype(int).to_numpy()
    baseline = ebm.predict_proba(test[features])[:, 1]
    baseline_ap, baseline_ll = average_precision_score(y, baseline), log_loss(y, baseline)
    rng = np.random.default_rng(808)
    checks = [
        ("origin_hourly_scheduled_flights", "origin_airport"),
        ("destination_daily_scheduled_flights", "destination_airport"),
    ]
    rows = []
    for feature, within in checks:
        shuffled = test[features].copy()
        values = shuffled[feature].to_numpy(copy=True)
        for indices in test.groupby(within, observed=True).indices.values():
            values[indices] = rng.permutation(values[indices])
        shuffled[feature] = values
        probability = ebm.predict_proba(shuffled)[:, 1]
        shuffled_ap, shuffled_ll = average_precision_score(y, probability), log_loss(y, probability)
        rows.append({
            "feature": feature,
            "conditioned_within": within,
            "baseline_average_precision": baseline_ap,
            "conditional_permutation_average_precision": shuffled_ap,
            "average_precision_drop_basis_points": (baseline_ap - shuffled_ap) * 10_000,
            "log_loss_increase": shuffled_ll - baseline_ll,
            "interpretation": "Retains within-airport incremental signal" if baseline_ap - shuffled_ap >= 0.0001
                              else "Mostly an airport-identity proxy; remove or residualize",
        })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    features = json.loads((REPORTS / "selected_features.json").read_text(encoding="utf-8"))
    sample = pd.read_parquet(INTERIM / "model_train_enriched.parquet", columns=features).sample(300_000, random_state=123)
    categorical = [feature for feature in features if feature in CATEGORICAL]
    numeric = [feature for feature in features if feature not in CATEGORICAL]

    numeric_frame = sample[numeric].apply(pd.to_numeric, errors="coerce")
    variance = numeric_frame.var().rename("variance").reset_index()
    variance.columns = ["feature", "variance"]
    variance["display_name"] = variance.feature.map(pretty_term)
    variance["zero_or_near_zero_variance"] = variance.variance.fillna(0).lt(1e-10)
    variance.to_csv(REPORTS / "feature_variance_screen.csv", index=False)

    usable = variance.loc[~variance.zero_or_near_zero_variance, "feature"].tolist()
    filled = numeric_frame[usable].fillna(numeric_frame[usable].median())
    spearman = filled.corr(method="spearman")
    pairs = []
    for left, right in combinations(usable, 2):
        pairs.append({"feature_1": left, "feature_2": right, "spearman_correlation": spearman.loc[left, right],
                      "absolute_correlation": abs(spearman.loc[left, right])})
    correlation_pairs = pd.DataFrame(pairs).sort_values("absolute_correlation", ascending=False)
    correlation_pairs.to_csv(REPORTS / "numeric_correlation_pairs.csv", index=False)

    pearson = filled.corr().to_numpy()
    vif = pd.DataFrame({"feature": usable, "vif": np.diag(np.linalg.pinv(pearson))})
    vif["display_name"] = vif.feature.map(pretty_term)
    vif.sort_values("vif", ascending=False).to_csv(REPORTS / "numeric_vif.csv", index=False)

    categorical_rows = []
    for left, right in combinations(categorical, 2):
        categorical_rows.append({"feature_1": left, "feature_2": right,
                                 "corrected_cramers_v": cramers_v(sample[left], sample[right])})
    pd.DataFrame(categorical_rows).sort_values("corrected_cramers_v", ascending=False).to_csv(
        REPORTS / "categorical_association.csv", index=False)

    mixed_rows = []
    for category in categorical:
        for value in usable:
            mixed_rows.append({"categorical_feature": category, "numeric_feature": value,
                               "correlation_ratio_eta": association_ratio(sample[category], sample[value])})
    pd.DataFrame(mixed_rows).sort_values("correlation_ratio_eta", ascending=False).to_csv(
        REPORTS / "mixed_type_association.csv", index=False)

    artifact = joblib.load(MODELS / "ebm.joblib")
    ebm, model_features = artifact["model"], artifact["features"]
    review = shape_review(ebm, model_features)
    review.to_csv(REPORTS / "ebm_shape_production_review.csv", index=False)
    make_shape_review_figure(ebm)
    conditional = conditional_permutation_audit(ebm, model_features)
    conditional.to_csv(REPORTS / "conditional_permutation_redundancy.csv", index=False)

    summary = {
        "maximum_numeric_vif": float(vif.vif.max()),
        "maximum_absolute_numeric_spearman": float(correlation_pairs.absolute_correlation.max()),
        "maximum_corrected_cramers_v": float(pd.DataFrame(categorical_rows).corrected_cramers_v.max()),
        "zero_variance_features": variance.loc[variance.zero_or_near_zero_variance, "feature"].tolist(),
        "numeric_pairs_above_0_7": int(correlation_pairs.absolute_correlation.ge(0.7).sum()),
        "vif_above_5": int(vif.vif.ge(5).sum()),
        "conditional_permutation": conditional.to_dict(orient="records"),
    }
    (REPORTS / "multicollinearity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nTop numeric pairs:\n", correlation_pairs.head(10).to_string(index=False))
    print("\nConditional permutation audit:\n", conditional.to_string(index=False))


if __name__ == "__main__":
    main()
