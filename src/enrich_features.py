from __future__ import annotations

"""Join leakage-safe network and airport context to the model samples."""

import math
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd

from common import INTERIM, PROCESSED, RAW, ensure_dirs


RENAME = {
    "Reporting_Airline": "airline",
    "Origin": "origin_airport",
    "Dest": "destination_airport",
    "month": "departure_month",
    "day_of_week": "departure_weekday",
    "departure_hour": "scheduled_departure_hour",
    "holiday_name": "federal_holiday_name",
    "CRSElapsedTime": "scheduled_duration_minutes",
    "Distance": "distance_miles",
    "origin_daily_scheduled": "origin_daily_scheduled_flights",
    "dest_daily_scheduled": "destination_daily_scheduled_flights",
    "origin_hourly_scheduled": "origin_hourly_scheduled_flights",
    "is_holiday": "is_federal_holiday",
    "within_3d_holiday": "within_three_days_of_federal_holiday",
    "days_to_nearest_holiday": "days_to_nearest_federal_holiday",
    "origin_hazard_forecast_any": "origin_nws_hazard_valid_at_departure",
    "origin_hazard_active_at_prediction": "origin_nws_hazard_active_at_cutoff",
    "origin_hazard_max_severity": "origin_nws_hazard_max_severity",
    "origin_hazard_convective": "origin_nws_convective_hazard",
    "origin_hazard_flood": "origin_nws_flood_hazard",
    "origin_hazard_tropical": "origin_nws_tropical_hazard",
    "origin_hazard_wind_fog": "origin_nws_wind_or_fog_hazard",
    "origin_hazard_winter": "origin_nws_winter_hazard",
    "dest_hazard_forecast_any": "destination_nws_hazard_valid_at_departure",
    "dest_hazard_active_at_prediction": "destination_nws_hazard_active_at_cutoff",
    "dest_hazard_max_severity": "destination_nws_hazard_max_severity",
    "dest_hazard_convective": "destination_nws_convective_hazard",
    "dest_hazard_flood": "destination_nws_flood_hazard",
    "dest_hazard_tropical": "destination_nws_tropical_hazard",
    "dest_hazard_wind_fog": "destination_nws_wind_or_fog_hazard",
    "dest_hazard_winter": "destination_nws_winter_hazard",
    "ArrDel15": "arrival_delay_15min",
}


def processed_pattern() -> str:
    return (PROCESSED / "flights" / "year=*" / "*.parquet").as_posix()


def build_lagged_performance(overwrite: bool = False) -> Path:
    """Seven-day airport and 28-day route delay rates, all ending at day -2."""
    output = INTERIM / "lagged_performance.parquet"
    if output.exists() and not overwrite:
        return output
    pattern = processed_pattern()
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET preserve_insertion_order=false")
    query = f"""
    WITH base AS (
      SELECT FlightDate::DATE AS flight_date, Origin AS origin_airport,
             Dest AS destination_airport, ArrDel15::DOUBLE AS delayed
      FROM read_parquet('{pattern}', hive_partitioning=true)
      WHERE primary_eligible=1
    ), origin_daily AS (
      SELECT flight_date, origin_airport AS airport, sum(delayed) AS delayed, count(*) AS n
      FROM base GROUP BY 1,2
    ), origin_roll AS (
      SELECT flight_date, airport,
        sum(delayed) OVER w / nullif(sum(n) OVER w, 0) AS origin_recent_delay_rate
      FROM origin_daily
      WINDOW w AS (PARTITION BY airport ORDER BY epoch(flight_date)
                   RANGE BETWEEN 691200 PRECEDING AND 172800 PRECEDING)
    ), destination_daily AS (
      SELECT flight_date, destination_airport AS airport, sum(delayed) AS delayed, count(*) AS n
      FROM base GROUP BY 1,2
    ), destination_roll AS (
      SELECT flight_date, airport,
        sum(delayed) OVER w / nullif(sum(n) OVER w, 0) AS destination_recent_delay_rate
      FROM destination_daily
      WINDOW w AS (PARTITION BY airport ORDER BY epoch(flight_date)
                   RANGE BETWEEN 691200 PRECEDING AND 172800 PRECEDING)
    ), route_daily AS (
      SELECT flight_date, origin_airport, destination_airport,
             sum(delayed) AS delayed, count(*) AS n
      FROM base GROUP BY 1,2,3
    ), route_roll AS (
      SELECT flight_date, origin_airport, destination_airport,
        sum(delayed) OVER w / nullif(sum(n) OVER w, 0) AS route_recent_delay_rate
      FROM route_daily
      WINDOW w AS (PARTITION BY origin_airport, destination_airport ORDER BY epoch(flight_date)
                   RANGE BETWEEN 2505600 PRECEDING AND 172800 PRECEDING)
    )
    SELECT r.flight_date, r.origin_airport, r.destination_airport,
           o.origin_recent_delay_rate, d.destination_recent_delay_rate,
           r.route_recent_delay_rate
    FROM route_roll r
    LEFT JOIN origin_roll o ON r.flight_date=o.flight_date AND r.origin_airport=o.airport
    LEFT JOIN destination_roll d ON r.flight_date=d.flight_date AND r.destination_airport=d.airport
    """
    con.execute(f"COPY ({query}) TO '{output.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()
    print(f"wrote {output}", flush=True)
    return output


def build_monthly_centrality(overwrite: bool = False) -> Path:
    """Compute schedule-network measures from the preceding calendar month."""
    output = INTERIM / "lagged_monthly_centrality.parquet"
    if output.exists() and not overwrite:
        return output
    edges = duckdb.sql(
        f"""
        SELECT date_trunc('month', FlightDate)::DATE AS network_month,
               Origin AS origin_airport, Dest AS destination_airport, count(*)::DOUBLE AS flights
        FROM read_parquet('{processed_pattern()}', hive_partitioning=true)
        GROUP BY 1,2,3 ORDER BY 1,2,3
        """
    ).df()
    edge_month = pd.to_datetime(edges.network_month)
    records: list[dict] = []
    for source_month in sorted(edge_month.unique()):
        month_edges = edges[edge_month.eq(source_month)]
        graph = nx.DiGraph()
        for row in month_edges.itertuples(index=False):
            graph.add_edge(row.origin_airport, row.destination_airport,
                           weight=float(row.flights), distance=1.0 / math.sqrt(float(row.flights)))
        if not graph:
            continue
        pagerank = nx.pagerank(graph, weight="weight")
        sample_k = min(75, len(graph))
        betweenness = nx.betweenness_centrality(
            graph, k=sample_k if sample_k < len(graph) else None,
            weight="distance", normalized=True, seed=42
        )
        total_weight = max(sum(d["weight"] for _, _, d in graph.edges(data=True)), 1.0)
        target_month = pd.Timestamp(source_month) + pd.offsets.MonthBegin(1)
        for airport in graph.nodes:
            records.append({
                "flight_month": target_month,
                "airport": airport,
                "network_pagerank": pagerank.get(airport, 0.0),
                "network_betweenness": betweenness.get(airport, 0.0),
                "network_weighted_out_share": graph.out_degree(airport, weight="weight") / total_weight,
                "network_weighted_in_share": graph.in_degree(airport, weight="weight") / total_weight,
            })
        print(f"centrality from {pd.Timestamp(source_month):%Y-%m}: {len(graph):,} airports", flush=True)
    pd.DataFrame(records).to_parquet(output, index=False, compression="zstd")
    return output


def load_faa_airports() -> pd.DataFrame:
    path = RAW / "faa" / "cy23-all-enplanements.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing official FAA workbook: {path}")
    faa = pd.read_excel(path)
    faa.columns = [str(c).strip() for c in faa.columns]
    faa = faa.rename(columns={"Locid": "airport", "Hub": "faa_hub_category",
                              "CY 23 Enplanements": "faa_2023_enplanements"})
    faa = faa[["airport", "faa_hub_category", "faa_2023_enplanements"]].copy()
    faa["airport"] = faa.airport.astype(str).str.strip()
    faa["faa_hub_category"] = faa.faa_hub_category.fillna("Non-primary").astype(str).str.strip()
    faa["faa_2023_enplanements"] = pd.to_numeric(faa.faa_2023_enplanements, errors="coerce")
    faa["faa_log_enplanements"] = np.log1p(faa.faa_2023_enplanements)
    return faa.drop_duplicates("airport")


def enrich_sample(source: Path, destination: Path, performance: pd.DataFrame,
                  centrality: pd.DataFrame, faa: pd.DataFrame) -> None:
    sample = pd.read_parquet(source).rename(columns=RENAME)
    sample = sample.drop(columns=["day_of_year", "doy_sin", "doy_cos", "hour_sin", "hour_cos"], errors="ignore")
    sample.loc[pd.to_numeric(sample["scheduled_duration_minutes"], errors="coerce").le(0),
               "scheduled_duration_minutes"] = np.nan
    sample["flight_date"] = pd.to_datetime(sample.pop("FlightDate")).dt.normalize()
    sample["flight_month"] = sample.flight_date.dt.to_period("M").dt.to_timestamp()
    sample = sample.merge(performance, on=["flight_date", "origin_airport", "destination_airport"],
                          how="left", validate="many_to_one")
    network_cols = [c for c in centrality.columns if c.startswith("network_")]
    origin_net = centrality.rename(columns={"airport": "origin_airport", **{c: f"origin_{c}" for c in network_cols}})
    destination_net = centrality.rename(columns={"airport": "destination_airport", **{c: f"destination_{c}" for c in network_cols}})
    sample = sample.merge(origin_net, on=["flight_month", "origin_airport"], how="left", validate="many_to_one")
    sample = sample.merge(destination_net, on=["flight_month", "destination_airport"], how="left", validate="many_to_one")
    faa_cols = [c for c in faa.columns if c != "airport"]
    origin_faa = faa.rename(columns={"airport": "origin_airport", **{c: f"origin_{c}" for c in faa_cols}})
    destination_faa = faa.rename(columns={"airport": "destination_airport", **{c: f"destination_{c}" for c in faa_cols}})
    sample = sample.merge(origin_faa, on="origin_airport", how="left", validate="many_to_one")
    sample = sample.merge(destination_faa, on="destination_airport", how="left", validate="many_to_one")
    sample.to_parquet(destination, index=False, compression="zstd")
    print(f"wrote {destination}: {len(sample):,} rows, {len(sample.columns):,} columns", flush=True)


def main() -> None:
    ensure_dirs()
    performance = pd.read_parquet(build_lagged_performance())
    performance["flight_date"] = pd.to_datetime(performance.flight_date)
    centrality = pd.read_parquet(build_monthly_centrality())
    centrality["flight_month"] = pd.to_datetime(centrality.flight_month)
    faa = load_faa_airports()
    enrich_sample(INTERIM / "model_train_sample.parquet", INTERIM / "model_train_enriched.parquet",
                  performance, centrality, faa)
    enrich_sample(INTERIM / "model_test_sample.parquet", INTERIM / "model_test_enriched.parquet",
                  performance, centrality, faa)


if __name__ == "__main__":
    main()
