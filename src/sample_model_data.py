from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from common import INTERIM, PROCESSED, ensure_dirs, load_config


COLUMNS = [
    "Reporting_Airline", "Origin", "Dest", "month", "day_of_week", "departure_hour",
    "holiday_name", "CRSElapsedTime", "Distance", "origin_daily_scheduled",
    "dest_daily_scheduled", "origin_hourly_scheduled", "is_holiday", "within_3d_holiday",
    "days_to_nearest_holiday", "origin_hazard_forecast_any",
    "origin_hazard_active_at_prediction", "origin_hazard_max_severity",
    "origin_hazard_convective", "origin_hazard_flood", "origin_hazard_tropical",
    "origin_hazard_wind_fog", "origin_hazard_winter", "dest_hazard_forecast_any",
    "dest_hazard_active_at_prediction", "dest_hazard_max_severity",
    "dest_hazard_convective", "dest_hazard_flood", "dest_hazard_tropical",
    "dest_hazard_wind_fog", "dest_hazard_winter", "ArrDel15", "ArrDelay", "year", "FlightDate",
]


def sample_years(years: list[int], rows: int, seed: int, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        print(f"already present: {output}")
        return
    pattern = (PROCESSED / "flights" / "year=*" / "*.parquet").as_posix()
    selected = ", ".join(f'"{column}"' for column in COLUMNS)
    year_filter = ", ".join(map(str, years))
    query = f"""
        SELECT * FROM (
            SELECT {selected}
            FROM read_parquet('{pattern}', hive_partitioning=true)
            WHERE primary_eligible=1 AND year IN ({year_filter})
        )
        USING SAMPLE reservoir({rows} ROWS) REPEATABLE ({seed})
    """
    temporary = output.with_suffix(".parquet.part")
    duckdb.sql(query).write_parquet(str(temporary), compression="zstd")
    written = duckdb.sql(f"SELECT count(*) FROM read_parquet('{temporary.as_posix()}')").fetchone()[0]
    if written != rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Expected {rows:,} sampled rows, wrote {written:,}")
    temporary.replace(output)
    print(f"wrote {output}: {rows:,} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    config = load_config()
    sample_years(config["train_years"], config["max_train_rows"], config["random_seed"],
                 INTERIM / "model_train_sample.parquet", args.overwrite)
    sample_years([config["test_year"]], config["max_test_rows"], config["random_seed"] + 1,
                 INTERIM / "model_test_sample.parquet", args.overwrite)


if __name__ == "__main__":
    main()
