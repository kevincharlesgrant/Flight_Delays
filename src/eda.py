from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "data" / "interim" / "matplotlib"))

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import FIGURES, PROCESSED, REPORTS, ensure_dirs
from make_report import ACCENT_BLUE, ACCENT_ORANGE, INK, MUTED_GRAY


AIRLINES = {
    "9E": "Endeavor", "AA": "American", "AS": "Alaska", "B6": "JetBlue",
    "DL": "Delta", "F9": "Frontier", "G4": "Allegiant", "HA": "Hawaiian",
    "MQ": "Envoy", "NK": "Spirit", "OH": "PSA", "OO": "SkyWest",
    "QX": "Horizon", "UA": "United", "WN": "Southwest", "YV": "Mesa",
    "YX": "Republic",
}


def query(connection: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return connection.execute(sql).df()


def main() -> None:
    ensure_dirs()
    pattern = (PROCESSED / "flights" / "year=*" / "*.parquet").as_posix()
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"CREATE VIEW flights AS SELECT * FROM read_parquet('{pattern}', hive_partitioning=true)")

    distribution = query(con, """
        SELECT count(*) AS eligible_flights,
               avg(ArrDelay) AS mean_delay_minutes,
               quantile_cont(ArrDelay, 0.01) AS p01,
               quantile_cont(ArrDelay, 0.10) AS p10,
               quantile_cont(ArrDelay, 0.25) AS p25,
               quantile_cont(ArrDelay, 0.50) AS median,
               quantile_cont(ArrDelay, 0.75) AS p75,
               quantile_cont(ArrDelay, 0.90) AS p90,
               quantile_cont(ArrDelay, 0.95) AS p95,
               quantile_cont(ArrDelay, 0.99) AS p99,
               max(ArrDelay) AS maximum,
               avg((ArrDelay >= 15)::INTEGER) AS delay_ge_15,
               avg((ArrDelay >= 30)::INTEGER) AS delay_ge_30,
               avg((ArrDelay >= 60)::INTEGER) AS delay_ge_60,
               avg((ArrDelay >= 120)::INTEGER) AS delay_ge_120
        FROM flights WHERE primary_eligible=1 AND ArrDelay IS NOT NULL
    """)
    distribution.to_csv(REPORTS / "eda_delay_distribution.csv", index=False)

    histogram = query(con, """
        SELECT greatest(-60, least(300, floor(ArrDelay / 10) * 10)) AS delay_bin,
               count(*) AS flights
        FROM flights WHERE primary_eligible=1 AND ArrDelay IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """)

    airline = query(con, """
        SELECT Reporting_Airline AS airline, count(*) AS scheduled_flights,
               avg(Cancelled) AS cancellation_rate,
               sum(primary_eligible) AS eligible_flights,
               avg(CASE WHEN primary_eligible=1 THEN ArrDel15 END) AS delay_rate,
               quantile_cont(CASE WHEN primary_eligible=1 THEN ArrDelay END, 0.5) AS median_delay_minutes,
               quantile_cont(CASE WHEN primary_eligible=1 THEN ArrDelay END, 0.9) AS p90_delay_minutes
        FROM flights GROUP BY 1 HAVING count(*) >= 100000 ORDER BY delay_rate DESC
    """)
    airline["airline_name"] = airline.airline.map(AIRLINES).fillna(airline.airline)
    airline.to_csv(REPORTS / "eda_airline_summary.csv", index=False)

    airport = query(con, """
        SELECT Origin AS airport, count(*) AS scheduled_flights,
               avg(Cancelled) AS cancellation_rate,
               sum(primary_eligible) AS eligible_flights,
               avg(CASE WHEN primary_eligible=1 THEN ArrDel15 END) AS origin_delay_rate,
               quantile_cont(CASE WHEN primary_eligible=1 THEN ArrDelay END, 0.9) AS p90_delay_minutes
        FROM flights GROUP BY 1 HAVING count(*) >= 100000 ORDER BY scheduled_flights DESC
    """)
    airport.to_csv(REPORTS / "eda_airport_summary.csv", index=False)

    monthly = query(con, """
        SELECT date_trunc('month', FlightDate)::DATE AS month, count(*) AS scheduled_flights,
               avg(Cancelled) AS cancellation_rate,
               avg(CASE WHEN primary_eligible=1 THEN ArrDel15 END) AS delay_rate,
               avg(CASE WHEN primary_eligible=1 THEN ArrDelay END) AS mean_delay_minutes
        FROM flights GROUP BY 1 ORDER BY 1
    """)
    monthly["month"] = pd.to_datetime(monthly.month)
    monthly.to_csv(REPORTS / "eda_monthly_patterns.csv", index=False)

    hourly = query(con, """
        SELECT departure_hour, count(*) AS eligible_flights, avg(ArrDel15) AS delay_rate,
               avg(ArrDelay) AS mean_delay_minutes
        FROM flights WHERE primary_eligible=1 GROUP BY 1 ORDER BY 1
    """)
    hourly.to_csv(REPORTS / "eda_hourly_patterns.csv", index=False)

    daily = query(con, """
        SELECT FlightDate::DATE AS flight_date, month(FlightDate) AS calendar_month,
               count(*) AS scheduled_flights, sum(primary_eligible) AS eligible_flights,
               avg(Cancelled) AS cancellation_rate,
               avg(CASE WHEN primary_eligible=1 THEN ArrDel15 END) AS delay_rate,
               avg(CASE WHEN primary_eligible=1 THEN greatest(origin_hazard_forecast_any,
                   dest_hazard_forecast_any) END) AS forecast_hazard_share,
               max(within_3d_holiday) AS holiday_window
        FROM flights GROUP BY 1, 2 HAVING sum(primary_eligible) >= 10000 ORDER BY 1
    """)
    month_stats = daily.groupby("calendar_month").delay_rate.agg(["mean", "std"])
    daily = daily.join(month_stats, on="calendar_month")
    daily["seasonally_standardized_anomaly"] = (daily.delay_rate - daily["mean"]) / daily["std"]
    hazard_cutoff = daily.forecast_hazard_share.quantile(0.90)
    cancellation_cutoff = daily.cancellation_rate.quantile(0.90)
    def context(row):
        reasons = []
        if row.forecast_hazard_share >= hazard_cutoff:
            reasons.append("elevated forecast-hazard exposure")
        if row.cancellation_rate >= cancellation_cutoff:
            reasons.append("elevated cancellations")
        if row.holiday_window:
            reasons.append("federal-holiday window")
        return "; ".join(reasons) if reasons else "no captured external indicator"
    anomalies = daily.nlargest(12, "seasonally_standardized_anomaly").copy()
    anomalies["observed_context_not_causal"] = anomalies.apply(context, axis=1)
    anomalies = anomalies[["flight_date", "scheduled_flights", "delay_rate", "cancellation_rate",
                           "forecast_hazard_share", "seasonally_standardized_anomaly",
                           "observed_context_not_causal"]]
    anomalies.to_csv(REPORTS / "eda_daily_anomalies.csv", index=False)

    quality = query(con, """
        SELECT count(*) AS total_rows,
               sum(Cancelled=1) AS cancelled_rows,
               sum(Diverted=1) AS diverted_rows,
               sum(Cancelled=1 AND ArrDelay IS NULL) AS cancelled_missing_arrival_delay,
               sum(primary_eligible=1 AND ArrDelay IS NULL) AS eligible_missing_arrival_delay,
               sum(CRSElapsedTime IS NULL OR CRSElapsedTime <= 0) AS invalid_scheduled_duration,
               sum(Distance IS NULL OR Distance <= 0) AS invalid_distance,
               sum(Reporting_Airline IS NULL OR Origin IS NULL OR Dest IS NULL) AS missing_identity,
               sum(primary_eligible=1 AND ArrDelay > 1440) AS delay_over_24_hours,
               sum(primary_eligible=1 AND ArrDelay < -120) AS arrival_over_2_hours_early
        FROM flights
    """)
    q = quality.iloc[0]
    rows = [
        ("Canceled flights", q.cancelled_rows, "Retained for QA; excluded from the arrival-delay target"),
        ("Diverted flights", q.diverted_rows, "Retained for QA; excluded from the arrival-delay target"),
        ("Canceled flights with missing arrival delay", q.cancelled_missing_arrival_delay,
         "Expected structural missingness; never imputed as on-time"),
        ("Eligible flights missing arrival delay", q.eligible_missing_arrival_delay,
         "Must equal zero for the modeling cohort"),
        ("Invalid/nonpositive scheduled duration", q.invalid_scheduled_duration,
         "Converted to missing before modeling; model-specific missing handling applies"),
        ("Invalid/nonpositive distance", q.invalid_distance, "Flagged before modeling"),
        ("Missing airline/origin/destination", q.missing_identity, "Flagged before modeling"),
        ("Arrival delays over 24 hours", q.delay_over_24_hours,
         "Valid but extreme BTS records; retained for binary endpoints"),
        ("Arrivals over 2 hours early", q.arrival_over_2_hours_early,
         "Potential data-quality tail; retained because target classification is unaffected"),
    ]
    quality_table = pd.DataFrame(rows, columns=["check", "rows", "treatment"])
    quality_table["share_of_all_rows"] = quality_table.rows / q.total_rows
    quality_table.to_csv(REPORTS / "eda_data_quality.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    histogram["share"] = histogram.flights / histogram.flights.sum()
    axes[0].bar(histogram.delay_bin, histogram.share * 100, width=9.2, color=ACCENT_ORANGE)
    axes[0].axvline(15, color=INK, linestyle="--", linewidth=2, label="15-minute target")
    axes[0].set(xlabel="Arrival delay (minutes; tails clipped at -60/+300)", ylabel="Eligible flights (%)",
                title="Arrival-delay distribution")
    axes[0].legend(frameon=False)
    thresholds = [15, 30, 60, 120]
    values = [distribution[f"delay_ge_{threshold}"].iloc[0] * 100 for threshold in thresholds]
    axes[1].bar([str(x) for x in thresholds], values, color=[ACCENT_ORANGE] + [ACCENT_BLUE] * 3)
    for index, value in enumerate(values):
        axes[1].text(index, value + 0.6, f"{value:.1f}%", ha="center", fontsize=11)
    axes[1].set(xlabel="Delay threshold (minutes)", ylabel="Eligible flights at or above threshold (%)",
                title="Right-tail severity")
    fig.tight_layout(); fig.savefig(FIGURES / "eda_delay_distribution.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    a = airline.sort_values("delay_rate")
    axes[0].barh(a.airline_name, a.delay_rate * 100, color=ACCENT_ORANGE)
    axes[0].axvline(distribution.delay_ge_15.iloc[0] * 100, color=INK, linestyle="--", label="Overall")
    axes[0].set(xlabel="Arrival delay rate (%)", title="Airline differences (unadjusted)")
    axes[0].legend(frameon=False)
    busiest = airport.nlargest(25, "scheduled_flights").sort_values("origin_delay_rate")
    axes[1].barh(busiest.airport, busiest.origin_delay_rate * 100, color=ACCENT_BLUE)
    axes[1].axvline(distribution.delay_ge_15.iloc[0] * 100, color=INK, linestyle="--", label="Overall")
    axes[1].set(xlabel="Arrival delay rate (%)", title="25 busiest origin airports (unadjusted)")
    axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "eda_airline_airport.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    axes[0].plot(monthly.month, monthly.delay_rate * 100, color=ACCENT_ORANGE, linewidth=2.4)
    axes[0].axhline(distribution.delay_ge_15.iloc[0] * 100, color=MUTED_GRAY, linestyle="--")
    axes[0].set(xlabel="Month", ylabel="Arrival delay rate (%)", title="Monthly pattern and year-to-year variation")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].plot(hourly.departure_hour, hourly.delay_rate * 100, color=ACCENT_BLUE, marker="o", linewidth=2.2)
    axes[1].axhline(distribution.delay_ge_15.iloc[0] * 100, color=MUTED_GRAY, linestyle="--")
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].set(xlabel="Scheduled departure hour (local)", ylabel="Arrival delay rate (%)",
                title="Delay risk accumulates through the operating day")
    fig.tight_layout(); fig.savefig(FIGURES / "eda_temporal_patterns.png", dpi=190); plt.close(fig)

    print(distribution.to_string(index=False))
    print("\nDaily anomalies:\n", anomalies.to_string(index=False))
    print("\nData quality:\n", quality_table.to_string(index=False))


if __name__ == "__main__":
    main()
