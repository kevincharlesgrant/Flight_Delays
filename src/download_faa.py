from __future__ import annotations

import requests

from common import RAW, ensure_dirs


FAA_URL = "https://www.faa.gov/sites/faa.gov/files/2024-10/cy23-all-enplanements.xlsx"


def main() -> None:
    ensure_dirs()
    destination = RAW / "faa" / "cy23-all-enplanements.xlsx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 100_000:
        print(f"already present: {destination}")
        return
    response = requests.get(FAA_URL, timeout=90)
    response.raise_for_status()
    destination.write_bytes(response.content)
    print(f"downloaded {FAA_URL} -> {destination}")


if __name__ == "__main__":
    main()
