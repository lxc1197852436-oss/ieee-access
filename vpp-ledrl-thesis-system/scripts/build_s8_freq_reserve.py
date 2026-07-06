"""Build scenario S8: frequency-regulation reserve event (unseen event category).

Motivation
----------
S5 (negative price) and S7 (export curtailment) both showed that when the physical
signal (price/PV surplus) is strong enough to drive the correct action on its own,
DeepSeek's semantic advantage over the keyword encoder is washed out -- the SAC
actor learns from price/PV alone. S8 is designed to break this: the correct
response (keep SOC mid-band for bidirectional reserve) is OPPOSITE to what the
physical signal alone suggests (PV surplus -> charge hard -> fill up), and the
reserve request applies at SCATTERED, non-fixed intervals so the actor cannot
learn WHEN to reserve from the hour encoding alone -- it must read the textual
event via the semantic channel.

Environment support (config.py + environment.py): a reserve-capacity penalty
`reserve_penalty_yuan_per_dev * (soc - 0.5)^2` is applied ONLY during intervals
whose event_type contains "调频". This makes "keep SOC mid-band" a reward payoff
contingent on the textual event, so a policy must know WHEN the reserve request
applies (via semantics) to avoid the penalty.

Keyword encoder behavior on S8: the event text deliberately AVOIDS the literal
tokens "新能源消纳/弃光/光伏/curtailment" (uses "发电/频率/调频/预留"), so the
keyword encoder returns all-zero scores -- it neither recognizes the reserve
request nor steers the actor away from filling up. DeepSeek should recognize
"调频预留" and return renewable=0, storage_bias=0 (no charge bias, keep mid-band).

Scenario structure:
  - PV surplus scattered across multiple windows (morning, midday, evening) so
    "charge during surplus" is a plausible numeric-only heuristic that FAILS
    during reserve intervals.
  - Reserve-request events planted at a scattered stride across DIFFERENT hours
    (not just midday) so the hour encoding alone cannot predict them.
  - Evening price spikes remain (discharge opportunity), so the policy must
    distinguish "reserve interval" from "discharge interval" using the event.
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

# A distinct event label/text that did NOT appear in training. The text avoids
# "新能源消纳/弃光/光伏/curtailment" so the keyword encoder returns all-zero.
# It uses "发电/频率/调频/预留" which DeepSeek should recognize as a reserve
# request (renewable=0, storage_bias=0 -> keep mid-band).
S8_EVENT_TYPE = "调频预留需求"
S8_EVENT_TEXT = (
    "系统频率短时波动，调度要求储能保留双向可调容量参与一次调频，"
    "本周期不宜满充或深放，需将SOC维持在中段以备上下调节。"
)

# Reserve penalty strength (per-unit SOC deviation squared per 15-min step).
# Tuned so that filling up (SOC=0.9, dev=0.4) costs ~0.4^2*penalty per step;
# over a 7-day scenario with ~30 reserve steps this is a meaningful but not
# overwhelming cost. The environment reads reserve_soc_target=0.5 by default.
RESERVE_PENALTY = 600.0


def apply_freq_reserve_stress(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    hour_arr = hour.to_numpy()

    # Reset all events to "Status: Normal" first, then plant reserve requests
    # at scattered intervals across DIFFERENT hours (not just midday) so the
    # hour encoding alone cannot predict them.
    df["event_type"] = "正常运行"
    df["event_text"] = "Status: Normal"

    # Plant reserve-request events at a sparse stride spanning morning, midday,
    # AND evening hours -- deliberately non-periodic so SAC cannot learn the
    # pattern from the hour-of-day feature alone.
    reserve_mask = (df.index % 17 == 0)  # ~ every 4.25h, non-aligned to day
    df.loc[reserve_mask, "event_type"] = S8_EVENT_TYPE
    df.loc[reserve_mask, "event_text"] = S8_EVENT_TEXT

    # PV surplus: keep base generator's PV (already has midday surplus) and
    # ADD scattered morning/evening surplus via small boosts, so "charge during
    # surplus" is a plausible numeric heuristic that conflicts with reserve.
    # Do NOT over-boost -- the point is that surplus exists, tempting charging.
    # Leave PV largely as-is so the reserve conflict comes from the event, not
    # extreme physics.

    df["scenario_id"] = "S8"
    df["scenario_name"] = "调频预留需求场景"
    df["stress_type"] = "freq_reserve"
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ScenarioConfig(
        start="2025-09-02 00:00:00",
        periods=96 * 7,
        freq="15min",
        seed=2067,
        region="广东省",
    )
    base = generate_china_vpp_scenario(cfg)
    df = apply_freq_reserve_stress(base)

    out_path = OUT_DIR / "s8_freq_reserve.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Diagnostics
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    print(f"Saved S8: {out_path}  rows={len(df)}")
    print(f"  reserve-request events planted: {int((df['event_type']==S8_EVENT_TYPE).sum())} "
          f"(~{int((df['event_type']==S8_EVENT_TYPE).sum())/7:.0f}/day)")
    # Show the hours at which reserve events land -- they should be scattered.
    reserve_hours = hour[df["event_type"] == S8_EVENT_TYPE].unique()
    print(f"  reserve event hours (should be scattered): {sorted(reserve_hours.round(2))}")
    print(f"  midday pv mean (10-15h)={df.loc[(hour>=10)&(hour<=15),'pv_mw'].mean():.2f}")
    print(f"  evening price mean (18-22h)={df.loc[(hour>=18)&(hour<=22),'price_yuan_mwh'].mean():.1f}")
    # Expected penalty if a policy fills to SOC=0.9 during all reserve steps:
    n_res = int((df["event_type"]==S8_EVENT_TYPE).sum())
    dev_full = 0.9 - 0.5
    print(f"  if SOC=0.9 during all {n_res} reserve steps: penalty~{RESERVE_PENALTY*dev_full*dev_full*n_res:.0f} yuan")
    print(f"  if SOC=0.5 (mid): penalty=0 yuan")
    print(f"  RESERVE_PENALTY={RESERVE_PENALTY} (set in train script, not here)")


if __name__ == "__main__":
    main()
