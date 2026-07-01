"""Build an out-of-distribution (OOD) VPP test set from real Guangzhou weather.

Training data are synthetic 2025-07 scenarios (see app/core/data.py). This
script builds a test set whose temperature and PV-driving signal come from
real Open-Meteo Guangzhou observations in 2024-07, which is a different year
and a different distribution. The load, price, and event channels are still
generated so that the reward and event semantics remain comparable with the
training set; only the weather-driven channels are replaced by real data.

The output CSV follows the load_vpp_dataset() schema so it can be fed directly
into VPPEnv and the existing policy/agent evaluation pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import VPPConfig

WEATHER_CSV = ROOT / "data" / "raw_sources" / "open_meteo" / "guangzhou_open_meteo_2024-07-01_2024-07-28.csv"
OUT_DIR = ROOT / "data" / "processed"

EVENTS = {
    "normal": ("正常运行", "Status: Normal"),
    "heat_load": ("高温预警", "广东气象台发布高温橙色预警，预计晚高峰空调负荷显著上升。"),
    "price_spike": ("价格尖峰", "现货市场公告提示日前价格异常波动，晚高峰可能出现尖峰电价。"),
    "renewable_curtailment": ("新能源消纳", "调度公告提示午间新能源消纳压力增大，建议提升储能充电能力。"),
}


def load_real_weather(path: Path, start: str, periods: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Open-Meteo timestamps are naive local time (Asia/Shanghai). Build a naive
    # 15-min index with the same convention so reindex aligns correctly.
    ts = pd.date_range(start, periods=periods, freq="15min")
    df = df.set_index("timestamp").reindex(ts)
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df["temperature_2m"] = df["temperature_2m"].interpolate(method="linear").ffill().bfill()
    df["shortwave_radiation"] = df["shortwave_radiation"].interpolate(method="linear").ffill().bfill().fillna(0.0)
    df["cloudcover"] = df["cloudcover"].interpolate(method="linear").ffill().bfill().fillna(50.0)
    return df[["timestamp", "temperature_2m", "shortwave_radiation", "cloudcover"]].head(periods)


def build_ood_scenario(weather: pd.DataFrame, stress_type: str, seed: int) -> pd.DataFrame:
    cfg = VPPConfig()
    n = len(weather)
    rng = np.random.default_rng(seed)
    hour = weather["timestamp"].dt.hour.to_numpy() + weather["timestamp"].dt.minute.to_numpy() / 60.0
    day = np.arange(n) / 96.0

    temperature_c = weather["temperature_2m"].to_numpy().astype(float)
    # PV from real shortwave radiation (W/m^2). 5 MW plant, ~1000 W/m^2 STC.
    ghi = weather["shortwave_radiation"].to_numpy().astype(float)
    cloud = weather["cloudcover"].to_numpy().astype(float) / 100.0
    pv_mw = np.maximum(0.0, 5.0 * (ghi / 1000.0) * (1.0 - 0.3 * cloud) + rng.normal(0, 0.05, n))

    summer_factor = 1.0 + 0.08 * np.sin(2 * np.pi * day / 14.0)
    evening_peak = np.exp(-0.5 * ((hour - 20.0) / 2.4) ** 2)
    noon_load = 0.35 * np.exp(-0.5 * ((hour - 14.0) / 3.2) ** 2)
    base_load = 3.4 + 1.2 * evening_peak + noon_load
    load_mw = np.maximum(1.2, base_load * summer_factor + rng.normal(0, 0.12, n))

    price_base = 380 + 140 * evening_peak - 90 * (ghi / 1000.0) + 5 * (temperature_c - 30)
    price_yuan_mwh = np.maximum(80, price_base + rng.normal(0, 25, n))

    event_type = np.array(["正常运行"] * n, dtype=object)
    event_text = np.array([EVENTS["normal"][1]] * n, dtype=object)

    if stress_type == "heat_load":
        mask = (temperature_c > 33) & (hour >= 11) & (hour <= 22)
        load_mw = np.where(mask, load_mw * 1.12, load_mw)
        emask = (hour >= 14) & (hour <= 20) & (np.arange(n) % 10 == 0)
        event_type[emask] = EVENTS["heat_load"][0]
        event_text[emask] = EVENTS["heat_load"][1]
    elif stress_type == "price_spike":
        mask = (hour >= 18) & (hour <= 22)
        price_yuan_mwh = np.where(mask, price_yuan_mwh * 1.35, price_yuan_mwh)
        price_yuan_mwh = np.clip(price_yuan_mwh, None, 650)
        emask = mask & (np.arange(n) % 8 == 0)
        event_type[emask] = EVENTS["price_spike"][0]
        event_text[emask] = EVENTS["price_spike"][1]
    elif stress_type == "renewable_curtailment":
        mask = (hour >= 10) & (hour <= 14)
        pv_mw = np.where(mask, pv_mw * 1.18, pv_mw)
        pv_mw = np.clip(pv_mw, None, 5.4)
        emask = mask & (np.arange(n) % 12 == 0)
        event_type[emask] = EVENTS["renewable_curtailment"][0]
        event_text[emask] = EVENTS["renewable_curtailment"][1]

    return pd.DataFrame(
        {
            "timestamp": weather["timestamp"],
            "region": "广东省",
            "load_mw": np.round(load_mw, 4),
            "pv_mw": np.round(pv_mw, 4),
            "price_yuan_mwh": np.round(price_yuan_mwh, 4),
            "temperature_c": np.round(temperature_c, 3),
            "event_type": event_type,
            "event_text": event_text,
        }
    )


OOD_SCENARIOS = [
    ("OOD-S1", "常规夏季运行场景(真实气象)", "normal", 2024, 96 * 7),
    ("OOD-S2", "高温负荷压力场景(真实气象)", "heat_load", 2031, 96 * 7),
    ("OOD-S3", "价格尖峰场景(真实气象)", "price_spike", 2042, 96 * 7),
    ("OOD-S4", "新能源消纳压力场景(真实气象)", "renewable_curtailment", 2053, 96 * 7),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OOD VPP test set from real Open-Meteo weather.")
    parser.add_argument("--start", type=str, default="2024-07-01")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    for scenario_id, name, stress, seed, periods in OOD_SCENARIOS:
        weather = load_real_weather(WEATHER_CSV, args.start, periods)
        df = build_ood_scenario(weather, stress, seed)
        out_path = OUT_DIR / f"ood_{scenario_id.lower()}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        df_out = df.copy()
        df_out["scenario_id"] = scenario_id
        df_out["scenario_name"] = name
        df_out["stress_type"] = stress
        combined_rows.append(df_out)
        print(
            f"{scenario_id} {name}: {len(df)} rows  "
            f"temp[{df['temperature_c'].min():.1f}-{df['temperature_c'].max():.1f}]  "
            f"pv[{df['pv_mw'].max():.2f}]  events={int((df['event_type']!='正常运行').sum())}"
        )
    combined = pd.concat(combined_rows, ignore_index=True)
    combined_path = OUT_DIR / "ood_vpp_scenarios.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    print(f"Saved combined OOD dataset: {combined_path}")


if __name__ == "__main__":
    main()
