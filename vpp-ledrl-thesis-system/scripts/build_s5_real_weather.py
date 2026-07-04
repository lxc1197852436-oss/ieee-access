"""Build S5-real: a real-weather negative-price week (PV and temperature from
Open-Meteo Archives; price dip calibrated because public real-time spot price
APIs are not reachable from this environment).

This is the "real-data" companion to S5. The PV-driving signal and temperature
come from real Open-Meteo historical observations for a high-renewable region
(Germany, 53.1N 13.1E) in July 2023, a period with documented negative-price
events in European spot markets. The price channel is constructed with the
same negative-price dip as S5 because public real-time spot-price APIs are
not reachable from this environment; this is disclosed explicitly in the
paper. The result is a hybrid: real weather-driven PV and temperature, with a
calibrated price channel that mirrors the S5 structure. It is stronger than a
purely synthetic scenario because the PV dynamics are real, but weaker than a
fully real price+weather scenario, which is left to future work.

Output: data/processed/s5_real_weather_ai_semantic.csv (DeepSeek-scored).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import VPPConfig
from app.core.llm_provider import LLMProvider

OUT_DIR = ROOT / "data" / "processed"
NEG_EVENT_TYPE = "负电价消纳"
NEG_EVENT_TEXT = (
    "现货市场公告：新能源大发，系统调峰空间不足，出清价格跌至负值，"
    "调度建议储能尽可能吸纳过剩新能源并避免在低压时段向电网反送电。"
)

LAT, LON = 53.1, 13.1  # Germany, high PV+wind, documented 2023-07 negative prices
START, END = "2023-07-01", "2023-07-07"


def fetch_open_meteo() -> pd.DataFrame:
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}"
           f"&start_date={START}&end_date={END}"
           f"&hourly=shortwave_radiation,temperature_2m,cloudcover&timezone=Europe%2FBerlin")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.load(urllib.request.urlopen(req, timeout=60))
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "shortwave_radiation": data["hourly"]["shortwave_radiation"],
        "temperature_2m": data["hourly"]["temperature_2m"],
        "cloudcover": data["hourly"]["cloudcover"],
    })
    return df


def resample_15min(hourly: pd.DataFrame) -> pd.DataFrame:
    # Reindex hourly to 15-min by interpolation.
    hourly = hourly.set_index("timestamp")
    ts = pd.date_range(hourly.index[0], periods=96 * 7, freq="15min")
    df = hourly.reindex(ts).interpolate(method="linear").ffill().bfill().reset_index()
    df = df.rename(columns={"index": "timestamp"}).head(96 * 7)
    return df


def build_scenario(weather: pd.DataFrame) -> pd.DataFrame:
    cfg = VPPConfig()
    n = len(weather)
    rng = np.random.default_rng(2026)
    hour = weather["timestamp"].dt.hour + weather["timestamp"].dt.minute / 60
    h = hour.to_numpy()

    ghi = weather["shortwave_radiation"].to_numpy().astype(float)
    cloud = weather["cloudcover"].to_numpy().astype(float) / 100.0
    pv_mw = np.maximum(0.0, 5.0 * (ghi / 1000.0) * (1.0 - 0.3 * cloud) + rng.normal(0, 0.05, n))

    temp_c = weather["temperature_2m"].to_numpy().astype(float)

    summer = 1.0 + 0.08 * np.sin(2 * np.pi * np.arange(n) / 96.0 / 14.0)
    eve = np.exp(-0.5 * ((h - 20.0) / 2.4) ** 2)
    noon_load = 0.35 * np.exp(-0.5 * ((h - 14.0) / 3.2) ** 2)
    load_mw = np.maximum(1.2, (3.4 + 1.2 * eve + noon_load) * summer + rng.normal(0, 0.12, n))

    # Flat evening (no high-price export) + noon negative-price dip (calibrated).
    price_base = 280 + 40 * eve
    dip = -460.0 * np.exp(-0.5 * ((h - 12.5) / 2.0) ** 2)
    price = price_base + dip + rng.normal(0, 20, n)

    event_type = np.array(["正常运行"] * n, dtype=object)
    event_text = np.array(["Status: Normal"] * n, dtype=object)
    emask = ((h >= 10) & (h <= 15)) & (np.arange(n) % 9 == 0)
    event_type[emask] = NEG_EVENT_TYPE
    event_text[emask] = NEG_EVENT_TEXT

    return pd.DataFrame({
        "timestamp": weather["timestamp"], "region": "德国(真实气象)",
        "load_mw": np.round(load_mw, 4), "pv_mw": np.round(pv_mw, 4),
        "price_yuan_mwh": np.round(price, 4), "temperature_c": np.round(temp_c, 3),
        "event_type": event_type, "event_text": event_text,
        "scenario_id": "S5R", "scenario_name": "真实气象负电价周(德国2023-07)",
        "stress_type": "neg_price_real_weather",
    })


def add_ai_semantics(df: pd.DataFrame) -> pd.DataFrame:
    provider = LLMProvider()
    events = df[["event_type", "event_text", "temperature_c", "price_yuan_mwh"]].drop_duplicates(
        subset=["event_type", "event_text"]).reset_index(drop=True)
    assessments = []
    for _, row in events.iterrows():
        ctx = {"scenario_id": "S5R", "event_type": str(row["event_type"]),
               "temperature_c": float(row["temperature_c"]), "price_yuan_mwh": float(row["price_yuan_mwh"])}
        a = provider.assess_event(str(row["event_text"]), context=ctx, allow_fallback=False)
        assessments.append({"event_type": row["event_type"], "event_text": row["event_text"],
            "ai_risk_score": a.risk_score, "ai_price_spike_score": a.price_spike_score,
            "ai_load_pressure_score": a.load_pressure_score,
            "ai_renewable_curtailment_score": a.renewable_curtailment_score,
            "ai_recommended_storage_bias": a.recommended_storage_bias,
            "ai_event_summary": a.event_summary, "ai_explanation": a.explanation,
            "ai_provider": a.provider, "ai_model": a.model})
    feat = pd.DataFrame(assessments)
    cols = ["event_type", "event_text", "ai_risk_score", "ai_price_spike_score",
            "ai_load_pressure_score", "ai_renewable_curtailment_score",
            "ai_recommended_storage_bias", "ai_event_summary", "ai_explanation",
            "ai_provider", "ai_model"]
    return df.merge(feat[cols], on=["event_type", "event_text"], how="left")


def main() -> None:
    print(f"Fetching Open-Meteo archives: lat={LAT} lon={LON} {START}..{END}")
    hourly = fetch_open_meteo()
    weather = resample_15min(hourly)
    df = build_scenario(weather)
    df = add_ai_semantics(df)
    out = OUT_DIR / "s5_real_weather_ai_semantic.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    h = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    neg = df["price_yuan_mwh"] < 0
    print(f"\nSaved S5-real: {out}  rows={len(df)}")
    print(f"  neg-price steps={int(neg.sum())}  price[min={df['price_yuan_mwh'].min():.0f} mean={df['price_yuan_mwh'].mean():.0f}]")
    print(f"  pv(real GHI-driven)[max={df['pv_mw'].max():.2f} mean={df['pv_mw'].mean():.2f}]")
    print(f"  temp(real)[mean={df['temperature_c'].mean():.1f}C max={df['temperature_c'].max():.1f}C]")


if __name__ == "__main__":
    main()
