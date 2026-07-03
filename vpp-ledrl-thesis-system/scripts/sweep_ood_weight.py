"""Sweep the safety-layer weight w on the OOD real-weather set.

The proposed controller (w=0.9) underperforms Rule-Based on OOD-S1 (normal
operation), likely because the semantic prior keeps the battery active even
without event pressure. This script sweeps w in {0.5, 0.75, 0.9, 1.0} on the
OOD set to see whether a lower w mitigates the benign-scenario over-cycling
while preserving the event-scenario advantage. It loads the DeepSeek-trained
LE-DRL-SAC checkpoint (seed 2026) and evaluates on the AI-semantic OOD data.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.environment import VPPEnv
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig
from app.core.rl.sac import SACAgent
from app.core.simulation import calculate_metrics

OOD_PATH = ROOT / "data" / "processed" / "ood_vpp_scenarios_ai_semantic.csv"
CKPT = ROOT / "outputs" / "chapter6_long" / "checkpoints" / "LE-DRL-SAC_seed2026.pt"
OUT_PATH = ROOT / "outputs" / "chapter6_long" / "ood_weight_sweep.csv"

WEIGHTS = [0.5, 0.75, 0.85, 0.9, 1.0]
POWER = 2.0


class ProgressBar:
    def __init__(self, total, label="", width=28):
        self.total = max(1, total)
        self.label = label
        self.width = width
        self.count = 0
        self.start = time.time()

    def update(self, n=1):
        self.count = min(self.total, self.count + n)
        elapsed = time.time() - self.start
        frac = self.count / self.total
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self.count / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.count) / rate if rate > 0 else 0.0
        sys.stdout.write(
            f"\r{self.label} [{bar}] {self.count}/{self.total} "
            f"({frac*100:5.1f}%) {elapsed:5.1f}s elapsed, eta {eta:5.1f}s  "
        )
        sys.stdout.flush()

    def finish(self):
        elapsed = time.time() - self.start
        sys.stdout.write(f"\r{self.label} [{'#'*self.width}] {self.total}/{self.total} (100.0%) {elapsed:5.1f}s done\n")
        sys.stdout.flush()


def load_scenarios():
    df = pd.read_csv(OOD_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out = []
    for sid, sub in df.groupby("scenario_id"):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        out.append((sid, str(sub["scenario_name"].iloc[0]), str(sub["stress_type"].iloc[0]), sub))
    return out


def evaluate(weight):
    sac = SACAgent.load(CKPT)
    agent = LEDRLAgent(
        LEDRLConfig(
            include_semantic=True,
            semantic_mode="native",
            name="LE-DRL-SAC + semantic safety layer",
            semantic_guidance_weight=weight,
            semantic_guidance_power=POWER,
        )
    )
    agent.sac = sac
    rows = []
    for sid, name, stress, data in load_scenarios():
        env = VPPEnv(data)
        state = env.reset(initial_soc=0.5)
        while not env.done():
            action = agent.act(state, deterministic=True)
            state, _, _, _ = env.step(action)
        metrics = calculate_metrics(pd.DataFrame(env.history))
        rows.append({"weight": weight, "scenario_id": sid, "scenario_name": name, **metrics})
    return rows


def main():
    all_rows = []
    bar = ProgressBar(len(WEIGHTS), label="OOD w-sweep")
    for w in WEIGHTS:
        all_rows.extend(evaluate(w))
        bar.update()
        avg = float(np.mean([r["total_reward_yuan"] for r in all_rows if r["weight"] == w]))
        print(f"\n  w={w:.2f}: avg_reward={avg:.1f}", flush=True)
    bar.finish()

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    summary = df.groupby("weight", as_index=False)[
        ["total_reward_yuan", "cvar_5_yuan", "battery_throughput_mwh", "high_price_discharge_rate"]
    ].mean()
    print("\nOOD weight sweep summary (cross-scenario mean):")
    print(summary.to_string(index=False))
    pivot = df.pivot(index="scenario_id", columns="weight", values="total_reward_yuan")
    print("\nOOD scenario-level total reward by w:")
    print(pivot.to_string())
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
