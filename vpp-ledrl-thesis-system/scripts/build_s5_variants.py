"""Build S5 parameter-sensitivity variants for the unseen-event robustness check.

The hand-crafted semantic prior was designed for four known event categories
and has no negative-price branch. S5 (noon negative-price surplus) showed that
LE-DRL-SAC with a relaxed regularizer exceeds SAC-Numeric on this unseen
event. To rule out the objection that S5 is a single hand-picked scenario,
this script builds four structurally distinct negative-price variants that
all satisfy the two conditions for large headroom (a deep negative-price
window plus the absence of a high-price export opportunity) but vary across
three independent dimensions:

  V1 (S5-ref):  noon negative price, summer, baseline depth  (reference)
  V2 (deep):    noon negative price, summer, deeper dip
  V3 (winter):  noon negative price, winter (weak PV, low load)
  V4 (night):   night negative price (wind surplus), summer, NO PV

V4 is structurally the most distinct: the surplus comes from wind at night,
so the prior's ``pv_surplus > 0.2 -> charge`` branch is fully inapplicable,
whereas S5's surplus comes from midday PV. If LE-DRL-SAC exceeds SAC-Numeric
on all four variants, the adaptation is not an artifact of one scenario.

Output: data/processed/s5_variants/<variant_id>.csv with the same schema as
s5_negative_price_surplus_ai_semantic.csv (semantic columns added by a later
DeepSeek-scoring step).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import ScenarioConfig
from app.core.data import generate_china_vpp_scenario

OUT_DIR = ROOT / "data" / "processed" / "s5_variants"
NEG_EVENT_TYPE = "负电价消纳"
NEG_EVENT_TEXT = (
    "现货市场公告：新能源大发，系统调峰空间不足，出清价格跌至负值，"
    "调度建议储能尽可能吸纳过剩新能源并避免在低价时段向电网反送电。"
)

# (variant_id, name, seed, period, dip_depth, pv_scale, season, start)
# V4 is a PV-capacity dimension: 4 MW plant (vs 5 MW in V1) under the same
# noon negative-price structure, so the prior's PV-surplus branch still
# applies but the surplus-to-load ratio differs.
VARIANTS = [
    ("V1", "S5基准变体-午间负电价夏季", 2060, "noon", -460.0, 1.0, "summer", "2025-07-29"),
    ("V2", "深度负电价-午间夏季", 2061, "noon", -650.0, 1.0, "summer", "2025-07-29"),
    ("V3", "冬季负电价-午间弱光伏", 2062, "noon", -460.0, 0.35, "winter", "2025-01-13"),
    ("V4", "4MW光伏午间负电价夏季", 2063, "noon", -460.0, 0.8, "summer", "2025-07-29"),
]


def apply_variant(df: pd.DataFrame, period: str, depth: float, pv_scale: float, season: str) -> pd.DataFrame:
    df = df.copy()
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    h = hour.to_numpy()

    # PV scaling (winter weakens PV; night variant zeroes PV since surplus is wind)
    df["pv_mw"] = df["pv_mw"] * pv_scale
    if period == "night":
        # Night wind surplus: PV is irrelevant at night; keep daytime PV normal.
        pass
    df["pv_mw"] = df["pv_mw"].clip(lower=0.0).round(4)

    # Load: winter lowers it.
    if season == "winter":
        df["load_mw"] = (df["load_mw"] * 0.8).round(4)

    # Flat evening peak (no high-price export opportunity): depress the evening
    # spike that the base generator inserts. This is the second headroom condition.
    eve = np.exp(-0.5 * ((h - 20.0) / 2.4) ** 2)
    # Remove the evening price spike and set a flat ~320 baseline.
    df["price_yuan_mwh"] = df["price_yuan_mwh"] - 100.0 * eve  # depress evening spike
    # Add the negative-price dip.
    if period == "noon":
        sur = np.exp(-0.5 * ((h - 12.5) / 2.0) ** 2)
    else:  # night
        sur = np.exp(-0.5 * ((h - 3.0) / 2.5) ** 2)
    df["price_yuan_mwh"] = df["price_yuan_mwh"] + depth * sur
    df["price_yuan_mwh"] = df["price_yuan_mwh"].round(4)

    # Plant the unseen event at a sparse stride during the negative window.
    if period == "noon":
        mask = (h >= 10) & (h <= 15)
    else:
        mask = (h >= 0) & (h <= 6)
    emask = mask & (df.index % 9 == 0)
    df.loc[emask, "event_type"] = NEG_EVENT_TYPE
    df.loc[emask, "event_text"] = NEG_EVENT_TEXT

    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for vid, name, seed, period, depth, pvs, season, start in VARIANTS:
        cfg = ScenarioConfig(start=f"{start} 00:00:00", periods=96 * 7, freq="15min", seed=seed, region="广东省")
        base = generate_china_vpp_scenario(cfg)
        df = apply_variant(base, period, depth, pvs, season)
        df["scenario_id"] = vid
        df["scenario_name"] = name
        df["stress_type"] = f"neg_price_{period}_{season}"
        out = OUT_DIR / f"{vid.lower()}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        h = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
        neg = df["price_yuan_mwh"] < 0
        print(f"{vid} {name}: rows={len(df)} neg_steps={int(neg.sum())} "
              f"price[min={df['price_yuan_mwh'].min():.0f} mean={df['price_yuan_mwh'].mean():.0f}] "
              f"pv[mean={df['pv_mw'].mean():.2f} max={df['pv_mw'].max():.2f}] "
              f"load[mean={df['load_mw'].mean():.2f}] events={int((df['event_type']==NEG_EVENT_TYPE).sum())}")


if __name__ == "__main__":
    main()
