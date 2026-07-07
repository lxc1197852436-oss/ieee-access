"""Build S5-real-price-multiweek: a REAL DE-LU day-ahead price validation scenario.

This is the N4 fix. The previous s5_real_price_week_ai_semantic.csv used a single
real day tiled to seven days with small noise (inter-day price correlation 0.997),
so the +2,201 EUR gap was effectively evaluated on one price pattern repeated
seven times. This script replaces it with a genuine three-week DE-LU day-ahead
series from SMARD/Bundesnetzagentur (CC BY 4.0), giving 21 independent daily
price patterns (inter-day correlation 0.875, with real workday/weekend and
negative-price-depth heterogeneity).

Channels:
  - price: REAL DE-LU day-ahead (SMARD JSON, hourly, interpolated to 15-min),
    EUR/MWh. 21 days, 504 hourly -> 2016 quarter-hourly steps.
  - PV + temperature: REAL Open-Meteo archive for Germany (53.1N 13.1E, same
    period 2025-06-15..2025-07-05), hourly -> 15-min. Same source as the previous
    s5_real_weather build, so the real-weather claim is preserved.
  - load: synthetic (public real-time load APIs unreachable; disclosed in paper).
    Mirrors the S5 base load structure with a mild summer factor; only the load
    magnitude is synthetic, not the price/PV/temperature signals.

Event text: planted ONLY on intervals where the REAL price < 0 (not on a fixed
index%9 schedule). This aligns the textual negative-price event with the real
negative-price windows, so the semantic channel carries the real event signal
rather than a template. The DeepSeek semantic encoder is queried once for the
single deduplicated negative-price event text (allow_fallback=False to force a
real LLM call).

Output: data/processed/s5_real_price_multiweek_ai_semantic.csv (DeepSeek-scored).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import VPPConfig
from app.core.llm_provider import LLMProvider

RAW_DIR = ROOT / "data" / "raw_sources" / "de_lu_real"
OUT_DIR = ROOT / "data" / "processed"

PRICE_JSON = RAW_DIR / "de_lu_dayahead_smard_2025-06-15_2025-07-05.json"
WEATHER_JSON = RAW_DIR / "open_meteo_germany_2025-06-15_2025-07-05.json"

NEG_EVENT_TYPE = "负电价消纳"
NEG_EVENT_TEXT = (
    "现货市场公告：新能源大发，系统调峰空间不足，出清价格跌至负值，"
    "调度建议储能尽可能吸纳过剩新能源并避免在低压时段向电网反送电。"
)

# SMARD prices are hourly in UTC. Open-Meteo timestamps are naive Berlin local
# time (utc_offset_seconds=7200 => +2h in summer). We build the output timeline
# in naive Berlin local time (15-min freq) and reindex both sources onto it.


def load_real_price() -> pd.DataFrame:
    with open(PRICE_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    # SMARD unix_seconds are UTC; convert to naive Berlin local by +2h in summer.
    utc = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
    berlin = utc.tz_convert("Europe/Berlin").tz_localize(None)  # naive Berlin
    df = pd.DataFrame({"timestamp": berlin, "price_eur_mwh": d["price"]})
    return df


def load_real_weather() -> pd.DataFrame:
    with open(WEATHER_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(d["hourly"]["time"]),
        "shortwave_radiation": d["hourly"]["shortwave_radiation"],
        "temperature_2m": d["hourly"]["temperature_2m"],
        "cloudcover": d["hourly"]["cloudcover"],
    })
    return df


def to_15min(hourly: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Interpolate hourly values onto a 15-min timeline (Berlin local, naive)."""
    ts0, ts1 = hourly["timestamp"].iloc[0], hourly["timestamp"].iloc[-1]
    grid = pd.date_range(ts0, ts1, freq="15min")
    df = hourly.set_index("timestamp").reindex(grid)
    df.index.name = "timestamp"
    df = df.reset_index().rename(columns={"index": "timestamp"})
    for c in value_cols:
        df[c] = df[c].interpolate(method="linear").ffill().bfill()
    return df.head(len(grid))


def build_scenario(price15: pd.DataFrame, weather15: pd.DataFrame) -> pd.DataFrame:
    cfg = VPPConfig()
    n = len(price15)
    rng = np.random.default_rng(2026)
    hour = price15["timestamp"].dt.hour.to_numpy() + price15["timestamp"].dt.minute.to_numpy() / 60.0
    day = np.arange(n) / 96.0

    # --- REAL price channel (EUR/MWh) ---
    price_eur = price15["price_eur_mwh"].to_numpy().astype(float)

    # --- REAL PV + temperature from Open-Meteo ---
    ghi = weather15["shortwave_radiation"].to_numpy().astype(float)
    cloud = weather15["cloudcover"].to_numpy().astype(float) / 100.0
    pv_mw = np.maximum(0.0, 5.0 * (ghi / 1000.0) * (1.0 - 0.3 * cloud) + rng.normal(0, 0.05, n))
    temp_c = weather15["temperature_2m"].to_numpy().astype(float)

    # --- Synthetic load (public real-time load APIs unreachable; disclosed) ---
    summer = 1.0 + 0.08 * np.sin(2 * np.pi * np.arange(n) / 96.0 / 14.0)
    eve = np.exp(-0.5 * ((hour - 20.0) / 2.4) ** 2)
    noon_load = 0.35 * np.exp(-0.5 * ((hour - 14.0) / 3.2) ** 2)
    load_mw = np.maximum(1.2, (3.4 + 1.2 * eve + noon_load) * summer + rng.normal(0, 0.12, n))

    # --- Event text: plant ONLY on real negative-price intervals ---
    event_type = np.array(["正常运行"] * n, dtype=object)
    event_text = np.array(["Status: Normal"] * n, dtype=object)
    neg_mask = price_eur < 0.0
    # Plant the negative-price event text on every negative-price step (the real
    # event is exactly the negative-price window; no fixed template schedule).
    event_type[neg_mask] = NEG_EVENT_TYPE
    event_text[neg_mask] = NEG_EVENT_TEXT

    return pd.DataFrame({
        "timestamp": price15["timestamp"],
        "region": "德国(DE-LU真实电价)",
        "load_mw": np.round(load_mw, 4),
        "pv_mw": np.round(pv_mw, 4),
        "price_yuan_mwh": np.round(price_eur, 4),  # column name kept for VPPEnv; values are EUR
        "temperature_c": np.round(temp_c, 3),
        "event_type": event_type,
        "event_text": event_text,
        "scenario_id": "S5R-MW",
        "scenario_name": "真实电价多周(DE-LU 2025-06-15~07-05)",
        "stress_type": "neg_price_real_de_lu_multiweek",
    })


def add_ai_semantics(df: pd.DataFrame) -> pd.DataFrame:
    provider = LLMProvider()
    # Deduplicate by (event_type, event_text); for each unique event pick a
    # representative row whose price/temperature context is meaningful.
    events = df[["event_type", "event_text", "temperature_c", "price_yuan_mwh"]].drop_duplicates(
        subset=["event_type", "event_text"]).reset_index(drop=True)
    assessments = []
    for _, row in events.iterrows():
        ctx = {"scenario_id": "S5R-MW", "event_type": str(row["event_type"]),
               "temperature_c": float(row["temperature_c"]),
               "price_yuan_mwh": float(row["price_yuan_mwh"])}
        a = provider.assess_event(str(row["event_text"]), context=ctx, allow_fallback=False)
        assessments.append({
            "event_type": row["event_type"], "event_text": row["event_text"],
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
    print("Loading real DE-LU price (SMARD) and real Open-Meteo weather...")
    price_h = load_real_price()
    weather_h = load_real_weather()
    print(f"  price: {len(price_h)} hourly pts  {price_h['timestamp'].iloc[0]} -> {price_h['timestamp'].iloc[-1]}")
    print(f"  weather: {len(weather_h)} hourly pts  {weather_h['timestamp'].iloc[0]} -> {weather_h['timestamp'].iloc[-1]}")

    # Both sources are hourly Berlin-local; interpolate to 15-min on the same grid.
    price15 = to_15min(price_h, ["price_eur_mwh"])
    weather15 = to_15min(weather_h, ["shortwave_radiation", "temperature_2m", "cloudcover"])
    print(f"  15-min grid: {len(price15)} price pts, {len(weather15)} weather pts")

    df = build_scenario(price15, weather15)
    print(f"\nScenario built: {len(df)} rows ({len(df)/96:.0f} days)")
    neg = df["price_yuan_mwh"] < 0
    print(f"  neg-price steps: {int(neg.sum())} ({100*neg.mean():.1f}%)")
    print(f"  price[min={df['price_yuan_mwh'].min():.2f} mean={df['price_yuan_mwh'].mean():.2f} max={df['price_yuan_mwh'].max():.2f}] EUR/MWh")
    print(f"  event_type distribution: {df['event_type'].value_counts().to_dict()}")

    print("\nQuerying DeepSeek for the negative-price event (single call)...")
    df = add_ai_semantics(df)
    out = OUT_DIR / "s5_real_price_multiweek_ai_semantic.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out}")
    # Print DeepSeek scores for verification
    ev = df[df["event_type"] == NEG_EVENT_TYPE]
    if len(ev):
        r = ev.iloc[0]
        print(f"\nDeepSeek scores on NEG event:")
        print(f"  risk={r['ai_risk_score']:.2f} price_spike={r['ai_price_spike_score']:.2f} "
              f"load_pressure={r['ai_load_pressure_score']:.2f} "
              f"renewable_curtailment={r['ai_renewable_curtailment_score']:.2f} "
              f"storage_bias={r['ai_recommended_storage_bias']:.2f}")
        print(f"  provider={r['ai_provider']} model={r['ai_model']}")


if __name__ == "__main__":
    main()
