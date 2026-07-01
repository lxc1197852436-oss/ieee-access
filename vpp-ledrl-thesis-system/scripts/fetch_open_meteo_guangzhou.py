"""Fetch real Guangzhou weather from the Open-Meteo Historical Weather API.

The thesis training data are synthetic 2025-07 scenarios. To support an
out-of-distribution (OOD) evaluation, this script pulls real Guangzhou
temperature and solar radiation from Open-Meteo for a different period and
saves them as hourly rows that the OOD dataset builder can resample to 15 min.

Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api
No API key is required. The endpoint returns hourly values by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw_sources" / "open_meteo"

# Guangdong coordinates. latitude 23.13 / longitude 113.26 is the Guangzhou
# area, which matches the thesis region ("广东省").
DEFAULT_LATITUDE = 23.13
DEFAULT_LONGITUDE = 113.26
DEFAULT_START = "2024-07-01"
DEFAULT_END = "2024-07-28"
DEFAULT_VARIABLES = "temperature_2m,shortwave_radiation,cloudcover"


def fetch_open_meteo(latitude: float, longitude: float, start: str, end: str, variables: str) -> dict:
    base = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": variables,
        "timezone": "Asia/Shanghai",
    }
    url = f"{base}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "vpp-ledrl-thesis/1.0"})
    with urlopen(req, timeout=60) as resp:  # noqa: S310 - public read-only API
        payload = json.loads(resp.read().decode("utf-8"))
    return payload


def to_dataframe(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"]), **{k: hourly[k] for k in hourly if k != "time"}})
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real Guangzhou weather from Open-Meteo.")
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--variables", type=str, default=DEFAULT_VARIABLES)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching Open-Meteo data for ({args.latitude}, {args.longitude}) {args.start}..{args.end}")
    payload = fetch_open_meteo(args.latitude, args.longitude, args.start, args.end, args.variables)
    df = to_dataframe(payload)

    raw_path = OUT_DIR / f"guangzhou_open_meteo_{args.start}_{args.end}.json"
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = OUT_DIR / f"guangzhou_open_meteo_{args.start}_{args.end}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Rows: {len(df)}  Range: {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved raw JSON: {raw_path}")
    print(f"Saved hourly CSV: {csv_path}")
    print(df[["timestamp", "temperature_2m", "shortwave_radiation", "cloudcover"]].describe().to_string())


if __name__ == "__main__":
    main()
