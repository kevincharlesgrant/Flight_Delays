from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "data" / "interim" / "matplotlib"))

import duckdb
import joblib
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from common import FIGURES, INTERIM, REPORTS, ROOT, ensure_dirs
from train_models import DISPLAY


MODEL_NAMES = {"logistic_regression": "Logistic regression", "xgboost": "XGBoost", "ebm": "EBM"}
# High-contrast presentation palette: orange accent, ink, and cool gray-blue.
ACCENT_ORANGE = "#F47B20"
INK = "#111111"
ACCENT_BLUE = "#2F6B8A"
MUTED_GRAY = "#8A8D91"
MODEL_COLORS = {"logistic_regression": MUTED_GRAY, "xgboost": ACCENT_ORANGE, "ebm": INK}


def pretty_term(term: str) -> str:
    if " & " in term:
        return " × ".join(DISPLAY.get(part, part.replace("_", " ").title()) for part in term.split(" & "))
    return DISPLAY.get(term, term.replace("_", " ").title())


def risk_delta_percentage_points(scores, reference_probability: float):
    """Translate log-odds effects to risk-point changes at a stated reference risk."""
    values = np.asarray(scores, dtype=float)
    reference_logit = math.log(reference_probability / (1 - reference_probability))
    shifted = 1 / (1 + np.exp(-np.clip(reference_logit + values, -30, 30)))
    return (shifted - reference_probability) * 100


def make_model_figures(predictions: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    models = ["logistic_regression", "xgboost", "ebm"]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for model in models:
        fpr, tpr, _ = roc_curve(predictions.actual, predictions[model])
        ax.plot(fpr, tpr, label=MODEL_NAMES[model], color=MODEL_COLORS[model])
    ax.plot([0, 1], [0, 1], "--", color="0.55")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="2025 temporal holdout: ROC curves")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "roc_curves.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for model in models:
        precision, recall, _ = precision_recall_curve(predictions.actual, predictions[model])
        # The sklearn endpoint at recall=0 has precision=1 by convention.  Omitting
        # the tiny <0.1% recall tail avoids a meaningless vertical line at the axis.
        shown = recall >= 0.001
        ax.plot(recall[shown], precision[shown], label=MODEL_NAMES[model], color=MODEL_COLORS[model])
    ax.axhline(predictions.actual.mean(), linestyle="--", color="0.55", label="Base rate")
    ax.set(xlabel="Recall", ylabel="Precision", title="2025 temporal holdout: precision-recall")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "precision_recall.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for model in models:
        observed, predicted = calibration_curve(predictions.actual, predictions[model], n_bins=12, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=MODEL_NAMES[model], color=MODEL_COLORS[model])
    ax.plot([0, 1], [0, 1], "--", color="0.55")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed delay rate", title="Probability calibration")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "calibration.png", dpi=180); plt.close(fig)


def make_screening_figure(screen: pd.DataFrame) -> None:
    data = screen.loc[screen.feature_group.ne("Lagged network centrality")].dropna(
        subset=["validation_ap_gain"]
    ).sort_values("validation_ap_gain")
    colors = [ACCENT_ORANGE if x else MUTED_GRAY for x in data.included]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.barh(data.feature_group, data.validation_ap_gain, color=colors)
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set(xlabel="Validation average-precision gain from feature block",
           title="Time-aware feature-block ablation (2024 validation)")
    for y, value in enumerate(data.validation_ap_gain):
        ax.text(value + 0.00025, y, f"{value:+.4f}", ha="left", va="center", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "feature_group_screening.png", dpi=180); plt.close(fig)


def make_propagation_feature_figure() -> None:
    """Document the three leakage-safe history features in one presentation-ready view."""
    definitions = pd.DataFrame([
        {"feature": "origin_recent_delay_rate", "plain_language": "Recent delay share for all departures from the origin airport",
         "history_window": "Days -8 to -2 before scheduled departure", "example_for_JFK_to_MIA": "All recent JFK departures"},
        {"feature": "destination_recent_delay_rate", "plain_language": "Recent delay share for all arrivals at the destination airport",
         "history_window": "Days -8 to -2 before scheduled departure", "example_for_JFK_to_MIA": "All recent MIA arrivals"},
        {"feature": "route_recent_delay_rate", "plain_language": "Recent delay share for the exact origin → destination route",
         "history_window": "Days -29 to -2 before scheduled departure", "example_for_JFK_to_MIA": "Recent JFK → MIA flights"},
    ])
    definitions.to_csv(REPORTS / "propagation_feature_definitions.csv", index=False)

    fig, ax = plt.subplots(figsize=(13.5, 7.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(5, 9.5, "Three leakage-safe delay-propagation features", ha="center", va="center",
            fontsize=19, fontweight="bold", color=INK)
    ax.text(5, 8.92, "Illustration: scoring a future JFK → MIA flight 24 hours before departure",
            ha="center", va="center", fontsize=11.5, color="0.30")

    ax.text(5, 8.14, "Prediction cutoff\n24 hours before departure", ha="center", va="center", fontsize=10,
            color="white", bbox={"boxstyle": "round,pad=0.50", "facecolor": INK, "edgecolor": INK})
    ax.annotate("", xy=(5, 7.32), xytext=(5, 7.80), arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.6})

    rows = [
        (6.58, ACCENT_ORANGE, "ORIGIN HISTORY", "All JFK departures", "Delayed departures ÷ all JFK departures", "Days -8 to -2"),
        (4.55, ACCENT_BLUE, "DESTINATION HISTORY", "All MIA arrivals", "Delayed arrivals ÷ all MIA arrivals", "Days -8 to -2"),
        (2.52, INK, "ROUTE HISTORY", "JFK → MIA flights only", "Delayed JFK → MIA flights ÷ all JFK → MIA flights", "Days -29 to -2"),
    ]
    for y, color, heading, population, formula, window in rows:
        ax.text(0.45, y, heading, ha="left", va="center", fontsize=10, fontweight="bold", color=color)
        ax.text(2.55, y, population, ha="left", va="center", fontsize=14, color=INK,
                bbox={"boxstyle": "round,pad=0.50", "facecolor": "#F5F6F7", "edgecolor": color, "linewidth": 1.8})
        ax.annotate("", xy=(5.65, y), xytext=(5.15, y), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.6})
        ax.text(7.7, y + 0.23, formula, ha="center", va="center", fontsize=9.8, color="0.22")
        ax.text(7.7, y - 0.33, window, ha="center", va="center", fontsize=10, fontweight="bold", color=color)
    ax.text(5, 0.78, "All windows end two days before departure. This buffer ensures each prior flight's arrival outcome was "
            "available at the 24-hour prediction cutoff; no current-flight or future outcomes are used.",
            ha="center", va="center", fontsize=10.5, color="0.25", wrap=True)
    fig.tight_layout(); fig.savefig(FIGURES / "propagation_feature_definitions.png", dpi=190); plt.close(fig)


def numeric_curve(global_explanation, ebm, term: str):
    data = global_explanation.data(ebm.term_names_.index(term))
    names, scores = np.asarray(data["names"]), np.asarray(data["scores"])
    if len(names) == len(scores) + 1:
        x = (names[:-1].astype(float) + names[1:].astype(float)) / 2
    else:
        x = names.astype(float)[:len(scores)]
    finite = np.isfinite(x) & np.isfinite(scores)
    return x[finite], scores[finite]


def categorical_risk_table(global_explanation, ebm, term: str, sample: pd.DataFrame,
                           reference_probability: float,
                           min_count: int = 5000, n: int = 10) -> pd.DataFrame:
    data = global_explanation.data(ebm.term_names_.index(term))
    effects = pd.DataFrame({"airport": data["names"], "internal_score": data["scores"]})
    counts = sample[term].value_counts().rename("sample_flights")
    effects = effects.join(counts, on="airport")
    effects = effects[effects.sample_flights.fillna(0).ge(min_count)].nlargest(n, "internal_score")
    effects["risk_change_at_22pct_baseline_pp"] = risk_delta_percentage_points(
        effects.internal_score, reference_probability
    )
    effects = effects.drop(columns="internal_score")
    effects.insert(0, "rank", range(1, len(effects) + 1))
    return effects


def make_ebm_outputs(test: pd.DataFrame, predictions: pd.DataFrame) -> None:
    artifact = joblib.load(ROOT / "models" / "ebm.joblib")
    ebm, features = artifact["model"], artifact["features"]
    global_explanation = ebm.explain_global()
    reference_probability = float(predictions.actual.mean())
    importance = pd.DataFrame({"term": ebm.term_names_, "importance": ebm.term_importances()})
    importance["display_term"] = importance.term.map(pretty_term)
    importance["influence_share_pct"] = 100 * importance.importance / importance.importance.sum()
    importance.sort_values("importance", ascending=False).to_csv(REPORTS / "ebm_term_importance.csv", index=False)
    top = importance.nlargest(16, "importance").sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.display_term, top.influence_share_pct, color=ACCENT_ORANGE)
    ax.set(xlabel="Share of total model influence (%)", title="EBM global term importance")
    fig.tight_layout(); fig.savefig(FIGURES / "ebm_term_importance.png", dpi=180); plt.close(fig)

    terms = ["scheduled_departure_hour", "scheduled_duration_minutes", "route_recent_delay_rate"]
    titles = ["Scheduled departure hour", "Scheduled gate-to-gate duration", "Recent route delay rate"]
    xlabels = ["Hour of day", "Minutes", "Delay share, days -29 to -2"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, term, title, xlabel in zip(axes, terms, titles, xlabels):
        x, scores = numeric_curve(global_explanation, ebm, term)
        if term == "route_recent_delay_rate":
            keep = (x >= 0) & (x <= 0.8)
            x, scores = x[keep], scores[keep]
        if term == "scheduled_duration_minutes":
            keep = (x >= 35) & (x <= 450)
            x, scores = x[keep], scores[keep]
        risk_points = risk_delta_percentage_points(scores, reference_probability)
        ax.plot(x, risk_points, color=ACCENT_ORANGE, linewidth=2)
        ax.axhline(0, color="0.5", linewidth=1)
        ax.set(title=title, xlabel=xlabel,
               ylabel="Change in delay risk (percentage points)\nfrom 22.3% reference")
    fig.tight_layout(); fig.savefig(FIGURES / "ebm_shape_functions.png", dpi=180); plt.close(fig)

    worst_origins = categorical_risk_table(global_explanation, ebm, "origin_airport", test, reference_probability)
    worst_destinations = categorical_risk_table(global_explanation, ebm, "destination_airport", test, reference_probability)
    worst_origins.to_csv(REPORTS / "ebm_worst_origins.csv", index=False)
    worst_destinations.to_csv(REPORTS / "ebm_worst_destinations.csv", index=False)

    duration_index = ebm.term_names_.index("scheduled_duration_minutes")
    analysis_sample = test.sample(n=min(400_000, len(test)), random_state=42).copy()
    term_values = ebm.eval_terms(analysis_sample[features])[:, duration_index]
    bins = [0, 60, 90, 120, 150, 180, 240, 300, 420, np.inf]
    labels = ["<60", "60–89", "90–119", "120–149", "150–179", "180–239", "240–299", "300–419", "420+"]
    duration = pd.DataFrame({"duration_band_minutes": pd.cut(analysis_sample.scheduled_duration_minutes,
                                                              bins=bins, labels=labels, right=False),
                             "risk_change_at_22pct_baseline_pp": risk_delta_percentage_points(
                                 term_values, reference_probability)})
    duration_table = duration.groupby("duration_band_minutes", observed=True).agg(
        sample_flights=("risk_change_at_22pct_baseline_pp", "size"),
        mean_risk_change_percentage_points=("risk_change_at_22pct_baseline_pp", "mean")
    ).reset_index().sort_values("mean_risk_change_percentage_points", ascending=False)
    duration_table.to_csv(REPORTS / "ebm_duration_risk.csv", index=False)

    # A representative high-risk delayed flight driven by recent propagation,
    # rather than an unstable maximum or an overlapping collection of hazard flags.
    hazard_cols = [c for c in test.columns if "_nws_" in c and "hazard" in c]
    candidate_mask = (predictions.actual.eq(1) & predictions.ebm.between(0.65, 0.78)
                      & test[hazard_cols].fillna(0).sum(axis=1).eq(0)
                      & test.route_recent_delay_rate.gt(0.35))
    candidates = predictions.index[candidate_mask]
    example_idx = int((predictions.loc[candidates, "ebm"] - 0.72).abs().idxmin()) if len(candidates) else int((predictions.ebm - 0.72).abs().idxmin())
    example = test.iloc[[example_idx]][features]
    local = ebm.explain_local(example).data(0)
    contributions = pd.DataFrame({"term": local["names"], "contribution": np.asarray(local["scores"], dtype=float),
                                  "value": local["values"]})
    contributions["display_term"] = contributions.term.map(pretty_term)
    contributions = contributions.reindex(contributions.contribution.abs().sort_values(ascending=False).index)
    top_local = contributions.head(11).copy()
    remainder = contributions.iloc[11:].contribution.sum()
    if abs(remainder) > 1e-10:
        top_local = pd.concat([top_local, pd.DataFrame([{"term": "other_terms", "contribution": remainder,
                                                        "value": "", "display_term": "All other terms"}])], ignore_index=True)
    intercept = float(local["extra"]["scores"][0])
    sigmoid = lambda value: 1 / (1 + math.exp(-value))
    baseline_probability = sigmoid(intercept)
    ordered = pd.concat([pd.DataFrame([{"display_term": "Model baseline", "contribution": intercept,
                                       "term": "intercept", "value": ""}]), top_local], ignore_index=True)
    start_probabilities, end_probabilities, changes = [0.0], [baseline_probability], [baseline_probability * 100]
    cumulative = intercept
    for value in top_local.contribution:
        start_probability = sigmoid(cumulative)
        cumulative += value
        end_probability = sigmoid(cumulative)
        start_probabilities.append(start_probability)
        end_probabilities.append(end_probability)
        changes.append((end_probability - start_probability) * 100)
    final_logit = cumulative
    final_probability = 1 / (1 + math.exp(-final_logit))
    ordered["start_probability"] = start_probabilities
    ordered["end_probability"] = end_probabilities
    ordered["change_percentage_points"] = changes
    ordered["example_index"], ordered["final_probability"] = example_idx, final_probability
    ordered[["display_term", "value", "start_probability", "end_probability",
             "change_percentage_points", "example_index", "final_probability"]].to_csv(
        REPORTS / "ebm_waterfall_example.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    x = np.arange(len(ordered) + 1)
    for i, row in ordered.iterrows():
        start, end = row.start_probability * 100, row.end_probability * 100
        low, high = min(start, end), max(start, end)
        color = INK if i == 0 else ACCENT_ORANGE if row.change_percentage_points >= 0 else ACCENT_BLUE
        ax.bar(i, high - low, bottom=low, width=0.72, color=color)
        ax.plot([i + 0.36, i + 0.64], [end, end], color="0.45", linewidth=1)
        label = f"{end:.1f}%" if i == 0 else f"{row.change_percentage_points:+.1f} pp"
        ax.text(i, high + 1.2, label, ha="center", va="bottom", fontsize=9)
    ax.bar(len(ordered), final_probability * 100, width=0.72, color=INK)
    short_names = {
        "Model baseline": "Model baseline",
        "Origin airport": "Origin airport",
        "Destination airport": "Destination airport",
        "Airline": "Airline",
        "Scheduled departure hour": "Departure hour",
        "Origin flights scheduled in the same hour": "Origin hourly volume",
        "Destination flights scheduled that day": "Destination daily volume",
        "Recent origin delay rate (days -8 to -2)": "Recent origin delays",
        "Recent destination delay rate (days -8 to -2)": "Recent destination delays",
        "Recent route delay rate (days -29 to -2)": "Recent route delays",
        "Departure month × Day of week": "Month × weekday",
        "Departure month × Scheduled departure hour": "Month × departure hour",
        "Scheduled departure hour × Recent origin delay rate (days -8 to -2)": "Hour × recent origin delays",
        "Scheduled departure hour × Recent route delay rate (days -29 to -2)": "Hour × recent route delays",
        "Scheduled departure hour × Scheduled gate-to-gate duration (minutes)": "Hour × duration",
        "Airline × Scheduled departure hour": "Airline × departure hour",
        "All other terms": "All other terms",
    }
    labels = [textwrap.fill(short_names.get(label, label), width=18) for label in ordered.display_term]
    labels += [f"Final risk\n{final_probability:.1%}"]
    ax.set_xticks(x, labels, rotation=0, ha="center", fontsize=8.5)
    ax.axhline(0, color="0.35", linewidth=1)
    ax.set_ylim(0, min(100, max(ordered.end_probability.max(), final_probability) * 100 + 9))
    flight = test.iloc[example_idx]
    ax.set(ylabel="Predicted delay risk (%)",
           title=f"EBM waterfall: {flight.origin_airport} → {flight.destination_airport}, "
                 f"{int(flight.scheduled_departure_hour):02d}:00 departure")
    fig.tight_layout(); fig.savefig(FIGURES / "ebm_waterfall_example.png", dpi=190); plt.close(fig)


def make_threshold_outputs() -> pd.DataFrame:
    threshold_metrics = json.loads((REPORTS / "delay_threshold_metrics.json").read_text(encoding="utf-8"))
    rows = []
    for name, values in threshold_metrics.items():
        threshold = int(name.rsplit("_", 1)[-1])
        rows.append({"delay_threshold_minutes": threshold, **values})
    summary = pd.DataFrame(rows).sort_values("delay_threshold_minutes")
    summary.to_csv(REPORTS / "delay_threshold_summary.csv", index=False)

    profile = pd.read_csv(REPORTS / "delay_threshold_example.csv")
    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    ax.plot(profile.delay_threshold_minutes, profile.coherent_probability,
            color=ACCENT_ORANGE, marker="o", linewidth=2.5, markersize=7)
    ax.fill_between(profile.delay_threshold_minutes, 0, profile.coherent_probability,
                    color=ACCENT_ORANGE, alpha=0.12)
    for row in profile.itertuples(index=False):
        ax.text(row.delay_threshold_minutes, row.coherent_probability + 0.025,
                f"{row.coherent_probability:.1%}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(profile.delay_threshold_minutes)
    ax.set_ylim(0, min(1, profile.coherent_probability.max() + 0.16))
    ax.set(xlabel="Arrival-delay threshold (minutes)", ylabel="Predicted probability",
           title=f"Delay-severity profile: {profile.origin_airport.iloc[0]} → "
                 f"{profile.destination_airport.iloc[0]}")
    fig.tight_layout(); fig.savefig(FIGURES / "delay_threshold_profile.png", dpi=190); plt.close(fig)
    return summary


def make_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# U.S. Flight Delay Risk — 24-Hour-Ahead Model\n\n"
            "Predicting arrival delays of at least 15 minutes among completed, non-diverted flights. "
            "Models train on 2022–2024 and are evaluated once on an untouched 2025 holdout."
        ),
        nbf.v4.new_markdown_cell(
            "## Executive summary\n\n"
            "At a 24-hour decision point, schedule, forecast-hazard, and recent network-performance data identify a useful "
            "priority list. On the untouched 2025 holdout, **45.1% of the EBM's highest-risk 10% of flights were delayed**, "
            "about twice the 22.3% rate across all flights. That group contained 20.2% of all delayed flights. XGBoost ranked "
            "flights only marginally better, so the EBM is recommended because it can explain each risk estimate directly.\n\n"
            "The strongest optional feature block is one we engineered from BTS history: recent origin, destination, and "
            "route delay propagation. It improved validation performance by **6.2%**—substantially "
            "more than any external-data block. Threshold EBMs extend the output from a binary alert to probabilities of "
            "at least 30, 60, and 120 minutes of delay."
        ),
        nbf.v4.new_markdown_cell(
            "## Data acquisition and external-data provenance\n\n"
            "The case materials referenced a provided sample, but no data file was included in the package received. The "
            "analysis therefore starts from the official BTS monthly archives and records every retrieval in executable scripts. "
            "All downloads are resumable and require no private credentials.\n\n"
            "| Dataset | What was retrieved | How it was retrieved | Modeling role |\n"
            "|---|---|---|---|\n"
            "| [BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/) | All monthly records for "
            "2022-2025: schedules, actual arrivals, delays, cancellations and diversions | `src/download_bts.py` sends "
            "streaming HTTP GET requests to the BTS `PREZIP` monthly ZIP pattern, validates each archive CRC, extracts the "
            "single CSV, and writes completion markers | Primary dataset and source for engineered congestion and lagged "
            "propagation features |\n"
            "| [NOAA/NWS warning archive via Iowa State IEM](https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml) | "
            "Historical watch/warning/advisory geometries with issuance, effective and expiration timestamps for 2022-2025 | "
            "`src/download_nws_hazards.py` queries the IEM `watchwarn.py` endpoint one year at a time for configured phenomena "
            "and significance codes; `src/build_dataset.py` spatially maps products to airport coordinates and retains only "
            "information issued by the 24-hour cutoff | Retained external predictor block |\n"
            "| [U.S. federal holiday calendar](https://pypi.org/project/holidays/) | Named federal holiday dates spanning the "
            "analysis years plus boundary years | Generated reproducibly by the pinned `holidays` package; nearest-holiday "
            "distance and a +/-3-day window are derived in `src/build_dataset.py` | Holiday-window flag retained |\n"
            "| [FAA CY2023 enplanements and hub classes](https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/cy23_all_enplanements) | "
            "FAA all-airport Excel workbook | `src/download_faa.py` downloads the published workbook directly and "
            "`src/enrich_features.py` maps airport codes to hub class and log enplanements | Hub class retained after narrowly "
            "clearing the pre-specified validation threshold; raw enplanements excluded as redundant |\n"
            "| [airportsdata](https://pypi.org/project/airportsdata/) | Airport latitude/longitude and IANA time zone | Loaded "
            "from the pinned package to convert local schedules to UTC and join forecast geometries | Join infrastructure only; "
            "not a predictor |\n\n"
            "Exact programmatic endpoints: BTS uses "
            "`https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip`; "
            "NWS/IEM uses `https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py` with explicit UTC start/end, "
            "phenomena, significance and shapefile parameters.\n\n"
            "Reproduction commands:\n\n"
            "```powershell\n"
            ".\\.venv\\Scripts\\python.exe src\\download_bts.py --workers 4\n"
            ".\\.venv\\Scripts\\python.exe src\\download_nws_hazards.py\n"
            ".\\.venv\\Scripts\\python.exe src\\download_faa.py\n"
            ".\\.venv\\Scripts\\python.exe src\\build_dataset.py\n"
            ".\\.venv\\Scripts\\python.exe src\\sample_model_data.py\n"
            "```\n\n"
            "The complete sequence—including EDA, feature enrichment, tuning, diagnostics, and notebook generation—is "
            "orchestrated by `run_pipeline.ps1`. Model logic is in `src/train_models.py`; the notebook is the presentation layer, "
            "not a second copy of the training implementation."
        ),
        nbf.v4.new_markdown_cell(
            "### Leakage guardrails\n\n"
            "Recent-delay features end two days before departure, so every retained input is available at the 24-hour cutoff. "
            "NWS products must have been issued by that cutoff; future validity is taken from the forecast product, not observed "
            "weather. Actual departure information, taxi times, reported delay causes, and all post-cutoff observations are excluded."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport os\n"
            "ROOT = Path.cwd().resolve()\n"
            "os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data' / 'interim' / 'matplotlib'))\n"
            "import pandas as pd\nfrom IPython.display import Image, display\n"
            "metrics = json.loads((ROOT / 'reports' / 'metrics.json').read_text())"
        ),
        nbf.v4.new_markdown_cell(
            "## Exploratory data analysis\n\n"
            "EDA uses all 27.7 million scheduled records from 2022-2025. Delay distributions and rates use the 27.1 million "
            "completed, non-diverted flights with a reported arrival outcome; cancellations and diversions are analyzed separately "
            "rather than labeled on-time. Descriptive airline and airport comparisons are **unadjusted** and should not be read "
            "as causal performance rankings."
        ),
        nbf.v4.new_markdown_cell(
            "### Delay distribution and target choice\n\n"
            "Arrival delay is strongly right-skewed: the median flight arrives 6 minutes early while the mean is 7.3 minutes "
            "late. The 90th, 95th and 99th percentiles are 44, 83 and 212 minutes. Overall, 21.2% of eligible flights cross "
            "the 15-minute BTS threshold, but only 2.9% exceed two hours. The long tail and extreme maximum make a robust "
            "classification endpoint—and the secondary severity thresholds—more defensible than ordinary least-squares minutes."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'eda_delay_distribution.png')))\n"
            "display(pd.read_csv(ROOT / 'reports' / 'eda_delay_distribution.csv').style.format({"
            "'eligible_flights':'{:,.0f}','mean_delay_minutes':'{:.1f}','p01':'{:.0f}','p10':'{:.0f}',"
            "'p25':'{:.0f}','median':'{:.0f}','p75':'{:.0f}','p90':'{:.0f}','p95':'{:.0f}',"
            "'p99':'{:.0f}','maximum':'{:,.0f}','delay_ge_15':'{:.1%}','delay_ge_30':'{:.1%}',"
            "'delay_ge_60':'{:.1%}','delay_ge_120':'{:.1%}'}))"
        ),
        nbf.v4.new_markdown_cell(
            "### Airline and airport differences\n\n"
            "Unadjusted delay rates vary materially: Frontier and JetBlue are near 30%, versus roughly 15%-17% for Endeavor, "
            "Republic and Delta. Among the 25 busiest origins, MIA and MCO are about 26.7%, while SLC is 16.5%. These gaps "
            "combine airline network, geography, schedule timing, route mix and operational choices; the predictive model "
            "controls for several of those factors before assigning an adjusted airport or airline effect."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'eda_airline_airport.png')))\n"
            "airlines = pd.read_csv(ROOT / 'reports' / 'eda_airline_summary.csv')\n"
            "display(airlines[['airline_name','scheduled_flights','delay_rate','cancellation_rate',"
            "'median_delay_minutes','p90_delay_minutes']].style.format({"
            "'scheduled_flights':'{:,.0f}','delay_rate':'{:.1%}','cancellation_rate':'{:.1%}',"
            "'median_delay_minutes':'{:.0f}','p90_delay_minutes':'{:.0f}'}))"
        ),
        nbf.v4.new_markdown_cell(
            "### Temporal patterns\n\n"
            "Summer is consistently difficult: July delay rates reach 28.6%-29.6% in 2023-2025. December 2022 is also "
            "anomalous, with a 27.1% delay rate and 5.5% cancellation rate. Within a day, the unadjusted delay rate rises "
            "from 8.9% for 05:00 departures to 30.4% around 20:00, consistent with disruption accumulating as aircraft and "
            "crews rotate through the network. This pattern motivates both departure-hour effects and lagged propagation features."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'eda_temporal_patterns.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "### Anomalies and data quality\n\n"
            "Daily anomalies are standardized relative to other days in the same calendar month, preventing every summer day "
            "from being labeled anomalous. The context column records observable co-occurrence—not a causal explanation. For "
            "example, 2024-07-19 had a 58.3% delay rate and 16.5% cancellations, while 2022-12-23 had a 62.1% delay rate, "
            "27.6% cancellations and elevated forecast-hazard exposure. External event records supply plausible explanations: "
            "July 19, 2024 coincides with the [CrowdStrike Windows outage](https://www.crowdstrike.com/en-us/blog/channel-file-291-rca-available/); "
            "January 11, 2023 with the [FAA NOTAM outage and nationwide ground stop](https://www.faa.gov/newsroom/faa-notam-statement); "
            "and December 23, 2022 with NOAA's [cross-country major winter storm and Arctic blast](https://www.wpc.ncep.noaa.gov/storm_summaries/event_reviews.php?YYYYMMDD=20221225). "
            "These dates were identified from BTS first and contextualized afterward; the event labels were not used as predictors.\n\n"
            "Canceled arrival delays are structurally missing and are never imputed as on-time. There are zero missing outcomes "
            "inside the modeling cohort. Twenty nonpositive scheduled durations occur in 27.7 million rows; the reproducible "
            "pipeline maps them to missing and applies model-specific handling. One landed in the development sample and none "
            "in the holdout. "
            "The 2,101 records above 24 hours remain valid delayed-flight outcomes. Because the target is binary, they receive "
            "the same delayed label as every other flight delayed by at least 15 minutes and no disproportionate weight."
        ),
        nbf.v4.new_code_cell(
            "anomalies = pd.read_csv(ROOT / 'reports' / 'eda_daily_anomalies.csv')\n"
            "display(anomalies.style.format({'scheduled_flights':'{:,.0f}','delay_rate':'{:.1%}',"
            "'cancellation_rate':'{:.1%}','forecast_hazard_share':'{:.1%}',"
            "'seasonally_standardized_anomaly':'{:.2f}'}))\n"
            "quality = pd.read_csv(ROOT / 'reports' / 'eda_data_quality.csv')\n"
            "display(quality.style.format({'rows':'{:,.0f}','share_of_all_rows':'{:.3%}'}))"
        ),
        nbf.v4.new_markdown_cell(
            "### Feature completeness and null handling\n\n"
            "The final raw feature matrices are nearly complete: 20,040 development rows (0.668%) and 8,889 holdout rows "
            "(0.593%) have at least one null. Two distinct mechanisms account for all of them. The three lagged propagation "
            "rates are unavailable on 6,136 development rows (0.205%) and 888 holdout rows (0.059%). Most of the "
            "development cold starts occur at the beginning of January 2022, before the source window contains enough prior "
            "days; the remainder primarily reflects newly observed routes or airports. Separately, about 0.23%-0.27% of origin "
            "or destination airport codes have no mapped category in the FAA hub-class reference. They become an explicit "
            "`Unknown / unclassified` category for every model. All native BTS schedule fields, calendar, congestion, and NWS "
            "forecast-hazard features are complete. One development row has an invalid scheduled duration; logistic regression "
            "uses its route's training-median duration plus a missing-value flag, while XGBoost and EBM retain a native missing value. "
            "A hazard flag of zero means no qualifying "
            "forecast product matched the airport and time window; it is an observed negative, not a null.\n\n"
            "Handling is model-specific. **XGBoost receives numeric NaNs directly** and learns the default branch at each split. "
            "Its sparse matrix explicitly retains genuine numeric zeros so they cannot be mistaken for absent values. **EBM uses "
            "its native missing-value bin.** Logistic regression cannot accept NaNs, so each propagation rate receives a separate "
            "`history unavailable` indicator and a time-safe hierarchical backoff: route falls back to the available origin and "
            "destination rates; an airport rate falls back to the available route/counterpart rate; only an all-missing cold start "
            "uses a neutral rate learned from the fitting sample. The indicators preserve the cold-start signal. No test outcomes "
            "or future observations enter these backoffs."
        ),
        nbf.v4.new_code_cell(
            "missing = pd.read_csv(ROOT / 'reports' / 'feature_missingness.csv')\n"
            "missing = missing.loc[missing.missing_observations.gt(0), "
            "['split','display_name','missing_observations','missing_share',"
            "'delay_rate_when_missing','delay_rate_when_present','logistic_handling','xgboost_handling','ebm_handling']]\n"
            "display(missing.style.format({'missing_observations':'{:,.0f}','missing_share':'{:.3%}',"
            "'delay_rate_when_missing':'{:.1%}','delay_rate_when_present':'{:.1%}'}))"
        ),
        nbf.v4.new_markdown_cell(
            "## Feature analysis\n\n"
            "Features fall into three provenance categories:\n\n"
            "- **Native BTS fields:** airline, origin and destination, scheduled month/day/hour, and scheduled gate-to-gate duration.\n"
            "- **Derived from BTS:** origin-hour and destination-day scheduled congestion plus lagged origin, destination, "
            "and route delay rates. The propagation rates use only outcomes ending two days before departure and were the "
            "most informative optional feature block (+6.2% relative validation improvement).\n"
            "- **External enrichment:** archived NOAA/NWS forecast hazards and the federal holiday calendar were retained. "
            "FAA hub classifications narrowly cleared the inclusion threshold; raw airport enplanements remained excluded.\n\n"
            "Feature blocks were screened on a 2024 validation slice using a fixed XGBoost model. "
            "A block was retained when removing it reduced average precision by at least 0.0010 (10 basis points). "
            "This selection was completed before looking at 2025 test performance."
        ),
        nbf.v4.new_code_cell(
            "catalog = pd.read_csv(ROOT / 'reports' / 'feature_catalog.csv')\n"
            "catalog = catalog.loc[catalog.feature_group.ne('Lagged network centrality')]\n"
            "display(catalog[['display_name','feature_group','source','included','decision']])\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'feature_group_screening.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "### What the propagation features measure\n\n"
            "The three propagation features summarize observed disruption in the network **before** the flight being scored. "
            "For a future JFK → MIA flight, origin history asks whether departures from JFK have recently been delayed; "
            "destination history asks whether arrivals into MIA have recently been delayed; and route history asks whether "
            "JFK → MIA flights themselves have recently been delayed. Each feature is a delayed-flight share, not a count, "
            "so it remains comparable across large and small airports/routes.\n\n"
            "Origin and destination histories use days -8 through -2; the route history uses days -29 through -2 because a "
            "single route has fewer flights and benefits from a longer lookback. Every window ends two days before scheduled "
            "departure, preserving the 24-hour-ahead information boundary. This is why these are called *propagation* features: "
            "they capture disruption already moving through the airport and route network, rather than conditions on the future "
            "flight itself."
        ),
        nbf.v4.new_code_cell(
            "propagation = pd.read_csv(ROOT / 'reports' / 'propagation_feature_definitions.csv')\n"
            "display(propagation[['feature','plain_language','history_window','example_for_JFK_to_MIA']])\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'propagation_feature_definitions.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## Multicollinearity and recycled-information audit\n\n"
            "Univariate screening is not enough when airport identity, traffic volume, weather flags, and lagged delay rates "
            "can encode overlapping information. We therefore combine Spearman correlation and variance-inflation factors "
            "(numeric–numeric), bias-corrected Cramér's V (categorical–categorical), and the correlation ratio "
            "(categorical–numeric).\n\n"
            "**Cramér's V** starts from a chi-square contingency table and measures how strongly two categorical variables "
            "are associated. It ranges from 0 (independent) to 1 (one category determines the other); the bias correction "
            "reduces upward distortion from large or sparse tables. It is symmetric, non-directional, and does not imply "
            "causality. Airport and FAA hub class are almost deterministically associated (V≈0.999) because hub class is an "
            "airport-level lookup. Airline and airport are also moderately associated (V≈0.44) because carriers operate "
            "different networks. The FAA block is therefore a deliberately redundant grouping feature; it was retained only "
            "because it narrowly cleared the pre-specified validation threshold.\n\n"
            "The **correlation ratio, eta**, handles one categorical and one numeric variable. It is the square root of the "
            "numeric variance explained by differences between category means: 0 means all category means are alike; 1 means "
            "category membership fully determines the numeric value. Unlike Cramér's V it is directional. Destination airport "
            "versus destination daily volume is 0.988, so a second test is needed to determine whether the remaining within-airport "
            "variation is useful.\n\n"
            "VIF below 5 and absolute correlation below 0.70 are treated as reassuring, not as proof "
            "that every feature is useful. For the two congestion variables most tightly associated with airport identity, "
            "we also permute values *within the same airport* on the 2025 sample. This preserves airport identity while "
            "testing whether day/hour variation still adds information. These diagnostics address redundancy and coefficient "
            "stability; they do not establish causality."
        ),
        nbf.v4.new_code_cell(
            "audit = json.loads((ROOT / 'reports' / 'multicollinearity_summary.json').read_text())\n"
            "display(pd.DataFrame([{k:v for k,v in audit.items() if k not in "
            "{'conditional_permutation','zero_variance_features'}}]))\n"
            "print('Largest numeric associations')\n"
            "display(pd.read_csv(ROOT / 'reports' / 'numeric_correlation_pairs.csv').head(10))\n"
            "print('Numeric VIF')\n"
            "display(pd.read_csv(ROOT / 'reports' / 'numeric_vif.csv').head(12))\n"
            "print('Largest categorical and mixed-type associations')\n"
            "display(pd.read_csv(ROOT / 'reports' / 'categorical_association.csv').head(6))\n"
            "display(pd.read_csv(ROOT / 'reports' / 'mixed_type_association.csv').head(10))\n"
            "print('Incremental congestion signal after conditioning on airport')\n"
            "display(pd.read_csv(ROOT / 'reports' / 'conditional_permutation_redundancy.csv'))"
        ),
        nbf.v4.new_markdown_cell(
            "**Decision.** No numeric pair breaches |ρ| = 0.70 and the maximum VIF is only 1.71, so the three propagation "
            "rates and the two timing variants of the NWS flags are not behaving like unstable duplicate regressors. "
            "Destination daily volume is nearly determined by destination airport and origin hourly volume is strongly tied "
            "to origin airport; however, conditional permutation still costs about 18 and 16 basis points of average precision, "
            "respectively. They therefore remain as dynamic congestion measures, not static airport proxies."
        ),
        nbf.v4.new_markdown_cell("## Hyperparameter tuning"),
        nbf.v4.new_code_cell(
            "tuning = pd.read_csv(ROOT / 'reports' / 'hyperparameter_tuning.csv')\n"
            "display(tuning.sort_values(['model','average_precision'], ascending=[True,False]))\n"
            "display(json.loads((ROOT / 'reports' / 'best_hyperparameters.json').read_text()))"
        ),
        nbf.v4.new_markdown_cell("## Final model results — untouched 2025 holdout"),
        nbf.v4.new_code_cell(
            "result = pd.DataFrame(metrics).T[['roc_auc','average_precision','log_loss','brier_score',"
            "'top_decile_delay_rate','top_decile_lift','share_delays_captured_top_decile']]\n"
            "display(result.sort_values('average_precision', ascending=False).style.format('{:.4f}'))"
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'roc_curves.png')))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'precision_recall.png')))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'calibration.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "### Reading the precision–recall curve\n\n"
            "Each point is a different risk-score threshold. **Precision** is the fraction of flagged flights that actually "
            "arrive at least 15 minutes late; **recall** is the fraction of all delayed flights captured. Moving right lowers "
            "the threshold and captures more delays, but admits more on-time flights, so precision falls toward the 22.3% base "
            "rate. A curve farther up and to the right is better; average precision summarizes that ranking performance across "
            "thresholds.\n\n"
            "At the extreme far left, recall is essentially zero because only one or a handful of the highest-scored flights "
            "are flagged. Precision is then discrete and unstable—one observation can move it from 0 to 1—and the standard "
            "curve definition adds the endpoint (recall 0, precision 1). The earlier vertical stroke at zero recall was therefore "
            "a plotting convention plus tiny-denominator noise, not a sudden EBM failure. The chart suppresses only the first "
            "0.1% of recall so the operational portion is legible."
        ),
        nbf.v4.new_markdown_cell(
            "## Feature interpretation\n\n"
            "The model is additive internally, but the stakeholder-facing charts translate its scores into **percentage-point "
            "changes in predicted delay risk**. Global curves and airport/duration tables use the 22.3% holdout delay rate as "
            "a common reference: for example, +3 percentage points means a change from 22.3% to 25.3% when that feature is "
            "applied at the reference risk. This makes effects comparable and intuitive, but it is a reference scenario—not a "
            "claim that the same feature adds exactly three points to every flight. The per-flight waterfall below recomputes "
            "the probability after each feature and therefore reaches the exact model prediction. All effects are adjusted "
            "predictive relationships, not causal claims. "
            "Airport rankings below require at least 5,000 sampled 2025 flights to suppress unstable small-airport effects."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'ebm_term_importance.png')))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'ebm_shape_functions.png')))"
        ),
        nbf.v4.new_code_cell(
            "print('Highest-risk origins, adjusted EBM effect')\n"
            "display(pd.read_csv(ROOT / 'reports' / 'ebm_worst_origins.csv').style.format({"
            "'risk_change_at_22pct_baseline_pp':'{:+.1f} pp'}))\n"
            "print('Highest-risk destinations, adjusted EBM effect')\n"
            "display(pd.read_csv(ROOT / 'reports' / 'ebm_worst_destinations.csv').style.format({"
            "'risk_change_at_22pct_baseline_pp':'{:+.1f} pp'}))\n"
            "print('Scheduled-duration bands, adjusted EBM effect')\n"
            "display(pd.read_csv(ROOT / 'reports' / 'ebm_duration_risk.csv').style.format({"
            "'mean_risk_change_percentage_points':'{:+.1f} pp'}))"
        ),
        nbf.v4.new_markdown_cell(
            "### EBM response-function production review\n\n"
            "These are EBM main-effect functions (often described informally as partial response functions). The underlying "
            "additive scores are translated here into percentage-point risk changes at the 22.3% reference rate. Interpretability does not "
            "make a jagged curve trustworthy. Scheduled gate-to-gate duration uses 249 effective bins and has 11 material "
            "direction reversals; the small adjacent-bin jumps are sampling noise, not credible operational thresholds. "
            "For production, coarsen duration to roughly 15-minute bins or `max_bins~=64`, increase per-bin support/regularization, "
            "and require the broad shape to reproduce across calendar-year folds. Do **not** force duration to be globally "
            "monotone: very short and very long schedules can behave differently.\n\n"
            "The recent origin, destination, and route delay rates have a clear increasing risk relationship and are suitable "
            "for non-decreasing monotonic constraints. Departure hour should remain non-monotone because its daily cycle is real; "
            "use adjacent-hour smoothing instead. Congestion curves should be coarsened, with a weak monotone constraint considered "
            "only after checking within-airport calibration. The orange curves below are diagnostics illustrating the intended "
            "regularity—not post-hoc replacements used for the reported predictions. The stock continuous-effect plots omit "
            "EBM's special missing-value bin; missing rows are nevertheless scored by that fitted bin and are audited separately "
            "in the feature-completeness table above."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'ebm_shape_production_review.png')))\n"
            "shape_review = pd.read_csv(ROOT / 'reports' / 'ebm_shape_production_review.csv')\n"
            "shape_review = shape_review.loc[~shape_review.feature.isin(" 
            "['origin_nws_convective_hazard','destination_nws_convective_hazard'])]\n"
            "display(shape_review[['display_name','bins_or_levels','roughness_ratio','direction_reversals',"
            "'spearman_x_to_effect','production_recommendation']])"
        ),
        nbf.v4.new_markdown_cell(
            "## One flight, decomposed\n\n"
            "The waterfall begins with this FSD→SFB flight's 18.7% model baseline. Its recent route-delay history raises risk "
            "to 33.2%, an exact **+14.5 percentage-point** change at that step. Each subsequent bar shows the probability before "
            "and after applying that feature or interaction, ending at **72.0% predicted risk**. The final prediction is invariant "
            "to bar order, but the percentage-point amount assigned to each step is order-dependent because probability is "
            "nonlinear. We therefore use a fixed, transparent order based on absolute model-score contribution."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'ebm_waterfall_example.png')))\n"
            "waterfall = pd.read_csv(ROOT / 'reports' / 'ebm_waterfall_example.csv')\n"
            "display(waterfall.style.format({'start_probability':'{:.1%}','end_probability':'{:.1%}',"
            "'change_percentage_points':'{:+.1f} pp','final_probability':'{:.1%}'}))"
        ),
        nbf.v4.new_markdown_cell(
            "## Delay-severity extension\n\n"
            "The primary endpoint remains arrival delay of at least 15 minutes. Separate EBMs estimate the probability "
            "of delays of at least 30, 60, and 120 minutes among completed, non-diverted flights. This threshold approach "
            "handles the strongly skewed delay distribution without pretending that a single point estimate of minutes is precise. "
            "Independent probabilities are made coherent with a cumulative-minimum correction, ensuring that the probability "
            "of a more severe delay cannot exceed the probability of a less severe delay."
        ),
        nbf.v4.new_code_cell(
            "display(pd.read_csv(ROOT / 'reports' / 'delay_threshold_summary.csv').style.format({"
            "'base_rate':'{:.1%}','roc_auc':'{:.4f}','average_precision':'{:.4f}',"
            "'log_loss':'{:.4f}','brier_score':'{:.4f}','fit_minutes':'{:.2f}'}))\n"
            "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'delay_threshold_profile.png')))\n"
            "display(pd.read_csv(ROOT / 'reports' / 'delay_threshold_example.csv'))"
        ),
        nbf.v4.new_markdown_cell(
            "## Appendix: other research avenues tested\n\n"
            "### Schedule-graph centrality\n\n"
            "We constructed a lagged monthly airport graph from BTS schedules and tested origin/destination PageRank and "
            "betweenness. Several individual measures had statistically detectable permutation effects, but the centrality "
            "block reduced validation average precision by 0.00061 (about 6 basis points) when added jointly. Airport identity "
            "and scheduled volume already capture most of the same hub structure, so centrality was excluded from every final "
            "model. This is a useful negative result: structural importance in the route network is not the same as near-term "
            "delay propagation."
        ),
        nbf.v4.new_code_cell(
            "centrality = pd.read_csv(ROOT / 'reports' / 'centrality_propagation_significance.csv')\n"
            "display(centrality.loc[centrality.feature.str.contains('network')])\n"
            "display(pd.read_csv(ROOT / 'reports' / 'feature_group_screening.csv').query(" 
            "\"feature_group == 'Lagged network centrality'\"))"
        ),
        nbf.v4.new_markdown_cell(
            "### GraphSAGE and airport-embedding MLP\n\n"
            "A research-only edge-classification prototype represented each flight as an origin→destination edge. GraphSAGE "
            "passed messages across the airport graph; the control MLP used learned origin and destination embeddings without "
            "graph message passing. On the common prototype holdout, both achieved approximately 0.3760 average precision. "
            "The embedding MLP had slightly better log loss than GraphSAGE (difference 0.000253; paired 95% CI 0.000126–0.000380). "
            "The experiment suggests that learned airport representations may help, but the graph operation itself added no "
            "measurable value. Given the extra infrastructure and weaker explainability, neither replaced the EBM; a production "
            "GNN would require a richer time-varying graph and a larger ablation program."
        ),
        nbf.v4.new_code_cell(
            "display(pd.read_csv(ROOT / 'reports' / 'gnn_test_results.csv').style.format({"
            "'roc_auc':'{:.4f}','average_precision':'{:.4f}','log_loss':'{:.4f}',"
            "'brier_score':'{:.4f}','top_decile_delay_rate':'{:.1%}','top_decile_lift':'{:.2f}',"
            "'fit_minutes':'{:.2f}'}))\n"
            "display(pd.read_csv(ROOT / 'reports' / 'gnn_paired_comparisons.csv'))"
        ),
        nbf.v4.new_markdown_cell(
            "### External-data experiments\n\n"
            "- **Archived NOAA/NWS forecast hazards — retained.** Products issued by the 24-hour cutoff and valid at the "
            "origin or destination at scheduled departure time added 0.00536 average precision (54 basis points).\n"
            "- **Federal holiday calendar — retained.** A ±3-day holiday-window indicator added 0.00135 average precision "
            "(13 basis points). Specific holiday names and exact distance-to-holiday were too sparse or redundant.\n"
            "- **FAA airport hub class — retained narrowly.** The block added 0.00101 average precision (10.1 basis points), "
            "just above the pre-committed threshold. Raw enplanements remained excluded as redundant with hub class, airport "
            "identity, and schedule volume. Airport coordinates and time zones were still used as join infrastructure for forecast "
            "hazards, not as model predictors."
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaway\n\n"
            "XGBoost has the strongest discrimination, but its advantage over the tuned EBM is small. "
            "The EBM therefore offers an attractive operational tradeoff: nearly the same ranking power, direct probability estimates, "
            "global shape functions, and a faithful per-flight additive explanation. Before production, calibration should be monitored "
            "over time and the EBM's high-resolution "
            "continuous effects should be constrained or smoothed as described above. The strongest new feature result is the "
            "BTS-derived recent route and airport delay propagation block."
        ),
    ]
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(notebook, ROOT / "Flight_Delay_Analysis.ipynb")


def main() -> None:
    ensure_dirs()
    metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(REPORTS / "test_predictions.parquet")
    test = pd.read_parquet(INTERIM / "model_test_enriched.parquet")
    for col in ["airline", "origin_airport", "destination_airport", "departure_month", "departure_weekday",
                "origin_faa_hub_category", "destination_faa_hub_category"]:
        test[col] = test[col].fillna("Missing").astype(str)
    screen = pd.read_csv(REPORTS / "feature_group_screening.csv")
    make_model_figures(predictions)
    make_screening_figure(screen)
    make_propagation_feature_figure()
    make_ebm_outputs(test, predictions)
    threshold_summary = make_threshold_outputs()
    make_notebook()

    pattern = (ROOT / "data" / "processed" / "flights" / "year=*" / "*.parquet").as_posix()
    summary = duckdb.sql(f"""
        SELECT year, count(*) AS flights, avg(primary_eligible) AS eligible_rate,
               avg(Cancelled) AS cancellation_rate, avg(Diverted) AS diversion_rate,
               avg(CASE WHEN primary_eligible=1 THEN ArrDel15 END) AS delay_rate
        FROM read_parquet('{pattern}', hive_partitioning=true)
        GROUP BY year ORDER BY year
    """).df()
    summary.to_csv(REPORTS / "dataset_summary.csv", index=False)
    model_table = pd.DataFrame(metrics).T.sort_values("average_precision", ascending=False).round(4)
    report = "# Model results\n\nAll final metrics use the untouched 2025 temporal holdout.\n\n"
    report += model_table.to_markdown() + "\n\n## Feature-block screening\n\n"
    report += screen.round(6).to_markdown(index=False) + "\n\n## Dataset summary\n\n"
    report += summary.round(4).to_markdown(index=False) + "\n"
    report += "\n## Delay-severity thresholds\n\n" + threshold_summary.round(4).to_markdown(index=False) + "\n"
    audit = json.loads((REPORTS / "multicollinearity_summary.json").read_text(encoding="utf-8"))
    report += "\n## Multicollinearity audit\n\n"
    report += (f"Maximum numeric VIF: {audit['maximum_numeric_vif']:.2f}; maximum absolute numeric Spearman correlation: "
               f"{audit['maximum_absolute_numeric_spearman']:.3f}.\n\n")
    report += pd.read_csv(REPORTS / "conditional_permutation_redundancy.csv").round(6).to_markdown(index=False) + "\n"
    report += "\n## EBM response-function review\n\n"
    shape_report = pd.read_csv(REPORTS / "ebm_shape_production_review.csv")
    shape_report = shape_report.loc[~shape_report.feature.isin(
        ["origin_nws_convective_hazard", "destination_nws_convective_hazard"]
    )]
    report += shape_report.round(4).to_markdown(index=False) + "\n"
    (REPORTS / "model_results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
