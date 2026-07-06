"""Build scenario S7: export-curtailed local absorption event (unseen event category).

Motivation
----------
The keyword encoder (LocalSemanticEncoder, app/core/semantic.py) matches the literal
four-character string "新能源消纳" to trigger its renewable-curtailment branch. The
semantically equivalent expression "就地消纳" (local absorption due to export
curtailment) does NOT match, so the keyword encoder returns all-zero scores for an
event that should trigger strong charging. This is the keyword encoder's precision-
match brittleness: a synonymous phrasing falls into its blind spot.

DeepSeek, by contrast, should recognize "就地消纳" as semantically equivalent to
renewable-curtailment and return renewable_curtailment ~0.7 and storage_bias ~+0.6,
steering the actor to charge during the midday surplus window.

Scenario structure (distinct from S5 negative-price):
  - Midday PV surplus (10-15h, PV boosted) creates local over-supply.
  - Export curtailment depresses the LOCAL price (not to negative, just low, ~200-280
    yuan/MWh, near the 260 threshold so the price<260 numeric branch is marginal and
    the semantic renewable signal becomes decisive).
  - Evening peak price is NOT depressed (unlike S5): there is still a high-price
    export opportunity in the evening, so the correct policy is "charge midday to
    absorb surplus, discharge evening at high price" -- the semantic signal must
    push midday charging over what the numeric price<260 branch alone would do.

The hand-crafted semantic safety prior (ledrl_agent._semantic_prior_action) was
designed for the four known event types. It has no branch keyed to "export
curtailment" / "就地消纳". Under this event the prior's renewable and storage_bias
inputs are zero (when fed keyword scores), so its midday charging strength is only
0.55 (from pv_surplus) instead of the 0.89 it reaches with correct renewable=0.75.
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

# A distinct event label/text that did NOT appear in training and that the keyword
# encoder cannot match. NOTE: the text deliberately AVOIDS the literal tokens
# "新能源消纳" / "弃光" / "光伏" / "curtailment" -- the keyword encoder's branch D
# matches any of these four-character tokens, so even "光伏" alone would trigger it
# and give the right answer by accident. By using "发电" / "就地消纳" / "联络线"
# / "外送" instead, the keyword encoder returns all-zero for an event whose correct
# response is strong charging -- exposing the precision-match brittleness cleanly.
S7_EVENT_TYPE = "外送受限消纳"
S7_EVENT_TEXT = (
    "调度中心通知：220kV联络线计划检修，外送通道受限，午间发电过剩时段优先就地消纳，"
    "避免反送电网造成线路过载。"
)


def apply_export_curtailed_stress(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    hour_arr = hour.to_numpy()

    # Midday surplus: boost PV toward plant ceiling during 10-15h ONLY.
    # Outside this window, PV/load/price stay at base-generator values so that
    # "Status: Normal" events landing outside 10-15h have a non-surplus, non-low-
    # price physical context -- preventing DeepSeek from mis-scoring them as
    # curtailment events based on context alone.
    noon_mask = (hour_arr >= 10) & (hour_arr <= 15)
    df.loc[noon_mask, "pv_mw"] = df.loc[noon_mask, "pv_mw"] * 1.22
    df["pv_mw"] = df["pv_mw"].clip(upper=5.4)

    # Export curtailment depresses LOCAL midday price only in the 10-15h window.
    # Base midday price ~290; *0.80 -> ~232, just below 260 (marginal, making the
    # semantic renewable signal decisive). Outside the window, price is unchanged.
    df.loc[noon_mask, "price_yuan_mwh"] = df.loc[noon_mask, "price_yuan_mwh"] * 0.80
    # Evening peak (18-22h) is NOT depressed: export curtailment is a midday
    # phenomenon; evening still has high-price export opportunity (contrast S5).

    # Reset ALL event fields to "Status: Normal" first, then plant the export-
    # curtailment event ONLY in the midday surplus window. This guarantees that
    # every "Status: Normal" row outside 10-15h carries a genuinely normal
    # physical context (non-surplus, non-low-price), so DeepSeek scores it as
    # normal rather than misreading the scene-wide physics as curtailment.
    df["event_type"] = "正常运行"
    df["event_text"] = "Status: Normal"
    event_mask = noon_mask & (df.index % 9 == 0)
    df.loc[event_mask, "event_type"] = S7_EVENT_TYPE
    df.loc[event_mask, "event_text"] = S7_EVENT_TEXT

    df["scenario_id"] = "S7"
    df["scenario_name"] = "外送受限就地消纳场景"
    df["stress_type"] = "export_curtailed"
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ScenarioConfig(
        start="2025-08-05 00:00:00",
        periods=96 * 7,
        freq="15min",
        seed=2066,
        region="广东省",
    )
    base = generate_china_vpp_scenario(cfg)
    df = apply_export_curtailed_stress(base)

    out_path = OUT_DIR / "s7_export_curtailed.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Diagnostics: prove the scenario exhibits the intended structure.
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    noon = (hour >= 10) & (hour <= 15)
    eve = (hour >= 18) & (hour <= 22)
    pv_surplus_noon = (df.loc[noon, "pv_mw"] - df.loc[noon, "load_mw"])
    print(f"Saved S7: {out_path}  rows={len(df)}")
    print(f"  midday pv     mean={df.loc[noon,'pv_mw'].mean():.2f}  max={df.loc[noon,'pv_mw'].max():.2f}")
    print(f"  midday load   mean={df.loc[noon,'load_mw'].mean():.2f}")
    print(f"  midday surplus(pv-load) mean={pv_surplus_noon.mean():.2f}  >0.2 steps={int((pv_surplus_noon>0.2).sum())}")
    print(f"  midday price  mean={df.loc[noon,'price_yuan_mwh'].mean():.1f}  "
          f"min={df.loc[noon,'price_yuan_mwh'].min():.1f}  max={df.loc[noon,'price_yuan_mwh'].max():.1f}")
    print(f"  midday price<260 steps={int((df.loc[noon,'price_yuan_mwh']<260).sum())}/{int(noon.sum())}")
    print(f"  evening price mean={df.loc[eve,'price_yuan_mwh'].mean():.1f}  "
          f"max={df.loc[eve,'price_yuan_mwh'].max():.1f}  (should be HIGH, not depressed)")
    print(f"  export-curtailment events planted: {int((df['event_type']==S7_EVENT_TYPE).sum())}")
    print(f"  any negative price? {int((df['price_yuan_mwh']<0).sum())} (should be 0)")


if __name__ == "__main__":
    main()
