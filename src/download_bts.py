from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import time
import zipfile

import requests

from common import RAW, ensure_dirs, load_config


URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)


def download_one(year: int, month: int, overwrite: bool = False) -> str:
    target_dir = RAW / "bts" / str(year)
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / f"bts_{year}_{month:02d}.zip"
    marker = target_dir / f"bts_{year}_{month:02d}.complete"
    if marker.exists() and not overwrite:
        return f"skip {year}-{month:02d}"

    partial = zip_path.with_suffix(".zip.part")
    url = URL.format(year=year, month=month)
    for attempt in range(1, 5):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    shutil.copyfileobj(response.raw, stream, length=1024 * 1024)
            partial.replace(zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                bad = archive.testzip()
                if bad:
                    raise zipfile.BadZipFile(f"CRC failure: {bad}")
                csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                if len(csv_names) != 1:
                    raise RuntimeError(f"Expected one CSV in {zip_path}, got {csv_names}")
                output_csv = target_dir / f"bts_{year}_{month:02d}.csv"
                with archive.open(csv_names[0]) as src, output_csv.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            marker.write_text("ok\n", encoding="ascii")
            return f"done {year}-{month:02d} ({zip_path.stat().st_size / 1e6:.1f} MB zip)"
        except Exception as exc:
            partial.unlink(missing_ok=True)
            if attempt == 4:
                raise RuntimeError(f"Failed {year}-{month:02d}: {exc}") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    years = load_config()["years"]
    jobs = [(year, month) for year in years for month in range(1, 13)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, y, m, args.overwrite) for y, m in jobs]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
