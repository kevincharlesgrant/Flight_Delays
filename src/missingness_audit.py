from __future__ import annotations

import json

import duckdb
import pandas as pd

from common import INTERIM, REPORTS, ensure_dirs
from train_models import CATEGORICAL, DISPLAY, PROPAGATION_RATES


SPLITS = {
    "Development (2022-2024 sample)": INTERIM / "model_train_enriched.parquet",
    "Holdout (2025 sample)": INTERIM / "model_test_enriched.parquet",
}


def main():
    ensure_dirs()
    features = json.loads((REPORTS / "selected_features.json").read_text(encoding="utf-8"))
    con = duckdb.connect()
    rows = []
    summaries = []
    for split, path in SPLITS.items():
        relation = f"read_parquet('{path.as_posix()}')"
        total = con.execute(f"SELECT count(*) FROM {relation}").fetchone()[0]
        any_missing = " OR ".join(f'"{feature}" IS NULL' for feature in features)
        rows_with_missing = con.execute(
            f"SELECT count(*) FROM {relation} WHERE {any_missing}"
        ).fetchone()[0]
        summaries.append({"split": split, "rows": total, "rows_with_any_feature_missing": rows_with_missing,
                          "share_with_any_feature_missing": rows_with_missing / total})
        for feature in features:
            missing, missing_rate, present_rate = con.execute(
                f'''SELECT count(*) - count("{feature}"),
                           avg(CASE WHEN "{feature}" IS NULL THEN arrival_delay_15min END),
                           avg(CASE WHEN "{feature}" IS NOT NULL THEN arrival_delay_15min END)
                    FROM {relation}'''
            ).fetchone()
            if feature in PROPAGATION_RATES:
                reason = "Insufficient lag history for a new date/airport/route"
                logistic = "Missing-history indicator + hierarchical, training-only backoff"
                xgboost = "Native learned missing branch; no imputation"
                ebm = "Native missing-value bin; no imputation"
            elif feature == "scheduled_duration_minutes":
                reason = "Missing or nonpositive BTS scheduled duration"
                logistic = "Missing-value indicator + training-only route median backoff"
                xgboost = "Native learned missing branch; no imputation"
                ebm = "Native missing-value bin; no imputation"
            elif feature in CATEGORICAL:
                reason = ("Airport code has no mapped FAA hub classification"
                          if feature.endswith("faa_hub_category") else "Unclassified source category")
                logistic = xgboost = ebm = "Explicit Unknown / unclassified category"
            else:
                reason = "No missing observations"
                logistic = "Observed value; no imputation"
                xgboost = "Native learned missing branch if a future NaN occurs"
                ebm = "Native missing-value bin if a future NaN occurs"
            rows.append({
                "split": split,
                "feature": feature,
                "display_name": DISPLAY.get(feature, feature),
                "rows": total,
                "missing_observations": missing,
                "missing_share": missing / total,
                "delay_rate_when_missing": missing_rate,
                "delay_rate_when_present": present_rate,
                "structural_reason": reason,
                "logistic_handling": logistic,
                "xgboost_handling": xgboost,
                "ebm_handling": ebm,
            })
    con.close()
    pd.DataFrame(rows).to_csv(REPORTS / "feature_missingness.csv", index=False)
    pd.DataFrame(summaries).to_csv(REPORTS / "feature_missingness_summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(pd.DataFrame(rows).query("missing_observations > 0")[
        ["split", "display_name", "missing_observations", "missing_share", "delay_rate_when_missing"]
    ].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
