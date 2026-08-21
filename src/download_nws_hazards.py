from __future__ import annotations

import argparse
import time
from urllib.parse import urlencode

import requests

from common import RAW, ensure_dirs, load_config


ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"


def download_year(year: int, phenomena: list[str], significance: list[str], overwrite: bool) -> None:
    target_dir = RAW / "nws_hazards"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"nws_hazards_{year}.zip"
    if target.exists() and not overwrite:
        print(f"skip hazards {year}", flush=True)
        return
    params = {
        "accept": "shapefile",
        "sts": f"{year}-01-01T00:00Z",
        "ets": f"{year + 1}-01-01T00:00Z",
        "limitps": "1",
        "simple": "1",
        "phenomena": ",".join(phenomena),
        "significance": ",".join(significance),
    }
    url = f"{ENDPOINT}?{urlencode(params)}"
    partial = target.with_suffix(".zip.part")
    for attempt in range(1, 5):
        try:
            with requests.get(url, stream=True, timeout=(30, 600)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            partial.replace(target)
            print(f"done hazards {year} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    config = load_config()
    for year in config["years"]:
        download_year(
            year,
            config["hazard_phenomena"],
            config["hazard_significance"],
            args.overwrite,
        )


if __name__ == "__main__":
    main()
