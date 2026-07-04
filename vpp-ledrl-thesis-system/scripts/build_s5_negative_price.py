"""Build scenario S5: negative-price deep-surplus event (unseen event category).

Motivation
----------
The hand-crafted semantic safety prior in ledrl_agent._semantic_prior_action
was designed before this event category existed. It has four branches keyed to
price thresholds (260 / 520 yuan/MWh), PV surplus, evening peak (18-22h), and
the four known event types. It has NO branch for negative prices.

Under negative-price deep surplus, the correct response is:
  - charge aggressively during the midday negative-price window to absorb PV
    surplus (reduces curtailment penalty AND avoids selling at negative price),
  - STOP charging once SOC is high (the prior keeps charging toward the 260
    threshold and hits the SOC ceiling, producing violation penalties),
  - do NOT discharge during the evening peak, because on a surplus day the
    evening price is also depressed (the prior's 18-22h discharge branch fires
    regardless of the actual price and loses money).

This scenario therefore exposes a structural limitation of a pre-designed
event prior: it cannot be extended to an unanticipated event category without
rewriting its branches, whereas a learning agent can be fine-tuned on the new
event's semantic scores.

The scenario reuses generate_china_vpp_scenario for the base load/PV/price
shapes (seed 2065, 7 days) and then applies a negative-price deep-surplus
stress transform. Output schema matches chapter6_ai_semantic_scenarios.csv so
the existing evaluation pipeline consumes it directly.
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

OUT_DIR = ROOT / "data" / "processed"

# A distinct event label/text pair that did NOT appear in training. DeepSeek
# has never scored this template in the cached ai_event_semantic_features.csv.
NEG_EVENT_TYPE = "负电价消纳"
NEG_EVENT_TEXT = (
    "现货市场公告：午间新能源大发，系统调峰空间不足，日前及实时出清价格跌至负值，"
    "调度建议储能尽可能吸纳过剩新能源并避免在低价时段向电网反送电。"
)


def apply_negative_price_stress(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    hour_arr = hour.to_numpy()

    # Midday deep surplus: boost PV toward plant ceiling during 10-15h.
    noon_mask = (hour_arr >= 10) & (hour_arr <= 15)
    df.loc[noon_mask, "pv_mw"] = df.loc[noon_mask, "pv_mw"] * 1.25
    df["pv_mw"] = df["pv_mw"].clip(upper=5.4)

    # Negative-price window: midday surplus collapses the spot price to
    # negative values (realistic for high-renewable penetration markets where
    # 5.4 MW PV faces ~3.9 MW load for ~5 hours, ~1.5 MW sustained net injection).
    # A smooth dip centered at 12.5h; depth chosen so the core window clears
    # below zero, matching observed negative-price events in high-RE markets.
    price_dip = -460.0 * np.exp(-0.5 * ((hour_arr - 12.5) / 2.0) ** 2)
    df["price_yuan_mwh"] = df["price_yuan_mwh"] + price_dip
    # Evening peak on a surplus day is depressed (no spike): pull 18-22h down.
    eve_mask = (hour_arr >= 18) & (hour_arr <= 22)
    df.loc[eve_mask, "price_yuan_mwh"] = df.loc[eve_mask, "price_yuan_mwh"] * 0.55
    df["price_yuan_mwh"] = df["price_yuan_mwh"].round(4)

    # Plant the unseen negative-price event at a sparse stride during the
    # negative-price window, mirroring apply_stress's event-planting style.
    event_mask = noon_mask & (df.index % 9 == 0)
    df.loc[event_mask, "event_type"] = NEG_EVENT_TYPE
    df.loc[event_mask, "event_text"] = NEG_EVENT_TEXT

    df["scenario_id"] = "S5"
    df["scenario_name"] = "负电价深度过剩场景"
    df["stress_type"] = "negative_price_surplus"
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ScenarioConfig(
        start="2025-07-29 00:00:00",
        periods=96 * 7,
        freq="15min",
        seed=2065,
        region="广东省",
    )
    base = generate_china_vpp_scenario(cfg)
    df = apply_negative_price_stress(base)

    out_path = OUT_DIR / "s5_negative_price_surplus.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Diagnostics: prove the scenario actually exhibits the intended structure.
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    noon = (hour >= 10) & (hour <= 15)
    eve = (hour >= 18) & (hour <= 22)
    neg = df["price_yuan_mwh"] < 0
    print(f"Saved S5: {out_path}  rows={len(df)}")
    print(f"  negative-price steps: {int(neg.sum())} (midday: {int((neg & noon).sum())})")
    print(f"  midday price  min={df.loc[noon,'price_yuan_mwh'].min():.1f}  "
          f"mean={df.loc[noon,'price_yuan_mwh'].mean():.1f}")
    print(f"  evening price mean={df.loc[eve,'price_yuan_mwh'].mean():.1f}  "
          f"max={df.loc[eve,'price_yuan_mwh'].max():.1f}")
    print(f"  midday pv     mean={df.loc[noon,'pv_mw'].mean():.2f}  max={df.loc[noon,'pv_mw'].max():.2f}")
    print(f"  midday load   mean={df.loc[noon,'load_mw'].mean():.2f}")
    print(f"  unseen events planted: {int((df['event_type']==NEG_EVENT_TYPE).sum())}")


if __name__ == "__main__":
    main()
