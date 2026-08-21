from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import airportsdata
import duckdb
import geopandas as gpd
import holidays
import numpy as np
import pandas as pd

from common import INTERIM, PROCESSED, RAW, ensure_dirs, load_config


BTS_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "Cancelled",
    "Diverted",
    "ArrDel15",
    "ArrDelay",
]

HAZARD_GROUPS = {
    "WS": "winter",
    "WW": "winter",
    "BZ": "winter",
    "IS": "winter",
    "HW": "wind_fog",
    "FG": "wind_fog",
    "FF": "flood",
    "FL": "flood",
    "FA": "flood",
    "SV": "convective",
    "TO": "convective",
    "HU": "tropical",
    "TR": "tropical",
}


def airport_lookup() -> pd.DataFrame:
    records = []
    for code, item in airportsdata.load("IATA").items():
        if item.get("country") in {"US", "PR", "VI"} and item.get("tz"):
            records.append(
                {
                    "airport": code,
                    "latitude": item["lat"],
                    "longitude": item["lon"],
                    "timezone": item["tz"],
                }
            )
    return pd.DataFrame(records).drop_duplicates("airport")


def active_bts_airports(years: list[int]) -> set[str]:
    """Capture the commercial network without scanning every wide CSV twice."""
    codes: set[str] = set()
    for year in years:
        for month in (1, 4, 7, 10):
            path = RAW / "bts" / str(year) / f"bts_{year}_{month:02d}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, usecols=["Origin", "Dest"], low_memory=False)
            codes.update(frame.Origin.dropna().astype(str).unique())
            codes.update(frame.Dest.dropna().astype(str).unique())
    return codes


def _read_hazard_zip(path: Path, year: int) -> gpd.GeoDataFrame:
    extract_dir = INTERIM / "hazards_extracted" / str(year)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(extract_dir)
    shp = next(extract_dir.glob("*.shp"))
    frame = gpd.read_file(shp)
    keep = ["ISSUED", "EXPIRED", "PHENOM", "SIG", "geometry"]
    return frame[keep].copy()


def build_hazard_hours(config: dict, overwrite: bool) -> Path:
    output = INTERIM / "airport_hazard_hours.parquet"
    if output.exists() and not overwrite:
        return output

    active_codes = active_bts_airports(config["years"])
    airports = airport_lookup()
    airports = airports[airports.airport.isin(active_codes)].copy()
    airport_geo = gpd.GeoDataFrame(
        airports,
        geometry=gpd.points_from_xy(airports.longitude, airports.latitude),
        crs="EPSG:4326",
    )
    rows: list[pd.DataFrame] = []

    for year in config["years"]:
        path = RAW / "nws_hazards" / f"nws_hazards_{year}.zip"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run download_nws_hazards.py")
        hazards = _read_hazard_zip(path, year)
        if hazards.crs is None:
            hazards = hazards.set_crs("EPSG:4326")
        else:
            hazards = hazards.to_crs("EPSG:4326")
        joined = gpd.sjoin(airport_geo, hazards, how="inner", predicate="within")
        joined["issued"] = pd.to_datetime(joined["ISSUED"], format="%Y%m%d%H%M", utc=True, errors="coerce")
        joined["expired"] = pd.to_datetime(joined["EXPIRED"], format="%Y%m%d%H%M", utc=True, errors="coerce")
        joined = joined.dropna(subset=["issued", "expired"])
        joined = joined[joined.expired >= joined.issued]
        joined["hazard_group"] = joined.PHENOM.map(HAZARD_GROUPS).fillna("other")
        joined["severity"] = joined.SIG.map({"Y": 1, "A": 2, "W": 3}).fillna(0).astype(int)

        intervals = joined[["airport", "issued", "expired", "hazard_group", "severity"]].copy()
        lead_hours = int(config["prediction_lead_hours"])
        con = duckdb.connect()
        con.register("intervals", intervals)
        # Generate only the relevant airport-hours in native code. Flooring to
        # the hour matches the flight join; the issue cutoff remains conservative
        # because the generated start is the ceiling of issue+lead.
        expanded = con.sql(
            f"""
            WITH base AS (
                SELECT *,
                    to_timestamp(ceil(epoch(issued + INTERVAL '{lead_hours} hours') / 3600) * 3600) AS forecast_start,
                    date_trunc('hour', expired) AS forecast_end,
                    to_timestamp(ceil(epoch(issued + INTERVAL '{lead_hours} hours') / 3600) * 3600) AS active_start,
                    date_trunc('hour', expired + INTERVAL '{lead_hours} hours') AS active_end
                FROM intervals
            ), forecast AS (
                SELECT airport, unnest(generate_series(forecast_start, forecast_end, INTERVAL '1 hour')) AS utc_hour,
                       hazard_group, 1 AS hazard_forecast, 0 AS hazard_active_at_prediction, severity
                FROM base WHERE forecast_start <= forecast_end
            ), active AS (
                SELECT airport, unnest(generate_series(active_start, active_end, INTERVAL '1 hour')) AS utc_hour,
                       hazard_group, 0 AS hazard_forecast, 1 AS hazard_active_at_prediction, severity
                FROM base WHERE active_start <= active_end
            )
            SELECT * FROM forecast UNION ALL SELECT * FROM active
            """
        ).df()
        con.close()
        if not expanded.empty:
            rows.append(expanded)
        print(f"hazards spatially joined {year}: {len(joined):,} airport-event pairs", flush=True)

    if not rows:
        raise RuntimeError("No airport hazard intersections were produced")
    long = pd.concat(rows, ignore_index=True)
    long["utc_hour"] = long.utc_hour.dt.tz_localize(None)
    agg = long.groupby(["airport", "utc_hour"], as_index=False).agg(
        hazard_forecast_any=("hazard_forecast", "max"),
        hazard_active_at_prediction=("hazard_active_at_prediction", "max"),
        hazard_max_severity=("severity", "max"),
    )
    for group in sorted(set(HAZARD_GROUPS.values())):
        subset = (
            long[long.hazard_group.eq(group)]
            .groupby(["airport", "utc_hour"], as_index=False)
            .hazard_forecast.max()
            .rename(columns={"hazard_forecast": f"hazard_{group}"})
        )
        agg = agg.merge(subset, on=["airport", "utc_hour"], how="left")
    hazard_cols = [c for c in agg.columns if c.startswith("hazard_")]
    agg[hazard_cols] = agg[hazard_cols].fillna(0).astype("int8")
    agg.to_parquet(output, index=False)
    print(f"wrote {output}: {len(agg):,} airport-hours", flush=True)
    return output


def scheduled_utc(frame: pd.DataFrame, tz_map: dict[str, str]) -> pd.Series:
    hhmm = pd.to_numeric(frame.CRSDepTime, errors="coerce").fillna(-1).astype(int)
    minutes = (hhmm // 100) * 60 + (hhmm % 100)
    local = frame.FlightDate + pd.to_timedelta(minutes, unit="m")
    result = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    tz_values = frame.Origin.map(tz_map)
    for tz_name, index in tz_values.groupby(tz_values).groups.items():
        localized = local.loc[index].dt.tz_localize(
            tz_name, ambiguous="NaT", nonexistent="shift_forward"
        ).dt.tz_convert("UTC").dt.tz_localize(None)
        result.loc[index] = localized
    return result


def holiday_features(dates: pd.Series, years: list[int]) -> pd.DataFrame:
    calendar = holidays.US(years=range(min(years) - 1, max(years) + 2))
    names = dates.dt.date.map(lambda d: calendar.get(d, "None"))
    holiday_dates = np.array(sorted(pd.Timestamp(d).to_datetime64() for d in calendar))
    day_values = dates.dt.normalize().values.astype("datetime64[D]")
    pos = np.searchsorted(holiday_dates.astype("datetime64[D]"), day_values)
    pos_lo = np.clip(pos - 1, 0, len(holiday_dates) - 1)
    pos_hi = np.clip(pos, 0, len(holiday_dates) - 1)
    lo = np.abs((day_values - holiday_dates[pos_lo].astype("datetime64[D]")).astype(int))
    hi = np.abs((holiday_dates[pos_hi].astype("datetime64[D]") - day_values).astype(int))
    distance = np.minimum(lo, hi)
    return pd.DataFrame(
        {
            "holiday_name": names.fillna("None").astype(str),
            "is_holiday": names.ne("None").astype("int8"),
            "within_3d_holiday": (distance <= 3).astype("int8"),
            "days_to_nearest_holiday": distance.astype("int16"),
        },
        index=dates.index,
    )


def process_month(path: Path, year: int, month: int, hazard_hours: pd.DataFrame, config: dict) -> Path:
    output_dir = PROCESSED / "flights" / f"year={year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"month={month:02d}.parquet"
    if output.exists():
        return output

    frame = pd.read_csv(path, usecols=BTS_COLUMNS, low_memory=False)
    frame["FlightDate"] = pd.to_datetime(frame.FlightDate)
    airports = airport_lookup()
    tz_map = airports.set_index("airport").timezone.to_dict()
    frame["scheduled_departure_utc"] = scheduled_utc(frame, tz_map)
    frame["utc_hour"] = frame.scheduled_departure_utc.dt.floor("h")
    frame["month"] = frame.FlightDate.dt.month.astype("int8")
    frame["day_of_week"] = frame.FlightDate.dt.dayofweek.astype("int8")
    frame["departure_hour"] = (pd.to_numeric(frame.CRSDepTime, errors="coerce").fillna(0).astype(int) // 100).clip(0, 23).astype("int8")

    frame["origin_daily_scheduled"] = frame.groupby(["FlightDate", "Origin"])["Origin"].transform("size").astype("int32")
    frame["dest_daily_scheduled"] = frame.groupby(["FlightDate", "Dest"])["Dest"].transform("size").astype("int32")
    frame["origin_hourly_scheduled"] = frame.groupby(["FlightDate", "Origin", "departure_hour"])["Origin"].transform("size").astype("int16")

    hfeatures = holiday_features(frame.FlightDate, config["years"])
    frame = pd.concat([frame, hfeatures], axis=1)

    hazard_cols = [c for c in hazard_hours.columns if c.startswith("hazard_")]
    origin_haz = hazard_hours.rename(columns={"airport": "Origin", **{c: f"origin_{c}" for c in hazard_cols}})
    dest_haz = hazard_hours.rename(columns={"airport": "Dest", **{c: f"dest_{c}" for c in hazard_cols}})
    frame = frame.merge(origin_haz, on=["Origin", "utc_hour"], how="left")
    frame = frame.merge(dest_haz, on=["Dest", "utc_hour"], how="left")
    merged_hazard_cols = [c for c in frame.columns if c.startswith(("origin_hazard_", "dest_hazard_"))]
    frame[merged_hazard_cols] = frame[merged_hazard_cols].fillna(0).astype("int8")

    frame["adverse_operation"] = (
        frame.Cancelled.eq(1) | frame.Diverted.eq(1) | frame.ArrDel15.eq(1)
    ).astype("int8")
    frame["primary_eligible"] = (
        frame.Cancelled.eq(0) & frame.Diverted.eq(0) & frame.ArrDel15.notna()
    ).astype("int8")
    frame["year"] = year
    frame.to_parquet(output, index=False, compression="zstd")
    print(f"wrote {year}-{month:02d}: {len(frame):,} rows", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-hazards", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    config = load_config()
    hazard_path = build_hazard_hours(config, args.overwrite_hazards)
    hazard_hours = pd.read_parquet(hazard_path)
    for year in config["years"]:
        for month in range(1, 13):
            path = RAW / "bts" / str(year) / f"bts_{year}_{month:02d}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}; run download_bts.py")
            process_month(path, year, month, hazard_hours, config)


if __name__ == "__main__":
    main()
