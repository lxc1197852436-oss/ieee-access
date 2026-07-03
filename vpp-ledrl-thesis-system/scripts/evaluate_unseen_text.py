"""Evaluate the proposed controller on unseen event expressions.

The training set uses five fixed event templates. This script evaluates the
DeepSeek-trained LE-DRL-SAC + semantic safety layer (w=0.9) on the unseen-text
scenarios, whose event expressions are different from training but share the
same operational meaning. The goal is to test whether the language-enhanced
policy generalizes to unseen event wording rather than memorizing templates.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.environment import VPPEnv
from app.core.policies import RuleBasedPolicy
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig
from app.core.rl.sac import SACAgent
from app.core.rolling_optimizer import EnhancedRollingHorizonPolicy, RollingHorizonOptimizerPolicy
from app.core.simulation import calculate_metrics

UNSEEN_PATH = ROOT / "data" / "processed" / "unseen_text_scenarios.csv"
CKPT = ROOT / "outputs" / "chapter6_long" / "checkpoints" / "LE-DRL-SAC_seed2026.pt"
OUT_PATH = ROOT / "outputs" / "chapter6_long" / "unseen_text_evaluation.csv"

METRIC_KEYS = [
    "total_reward_yuan",
    "mean_reward_yuan",
    "cvar_5_yuan",
    "final_soc",
    "battery_throughput_mwh",
    "high_price_discharge_rate",
    "low_price_charge_rate",
    "event_count",
]


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
    df = pd.read_csv(UNSEEN_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out = []
    for sid, sub in df.groupby("scenario_id"):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        out.append((sid, str(sub["scenario_name"].iloc[0]), str(sub["stress_type"].iloc[0]), sub))
    return out


def evaluate(policy, policy_name, data, sid, name, stress):
    env = VPPEnv(data)
    state = env.reset(initial_soc=0.5)
    while not env.done():
        if hasattr(policy, "encode"):
            action = policy.act(state, deterministic=True)
        else:
            action = policy.act(state)
        state, _, _, _ = env.step(action)
    metrics = calculate_metrics(pd.DataFrame(env.history))
    return {"scenario_id": sid, "scenario_name": name, "stress_type": stress, "policy": policy_name, **{k: metrics[k] for k in METRIC_KEYS}}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    scenarios = load_scenarios()
    sac = SACAgent.load(CKPT)
    proposed = LEDRLAgent(
        LEDRLConfig(
            include_semantic=True,
            semantic_mode="native",
            name="LE-DRL-SAC + semantic safety layer",
            semantic_guidance_weight=0.9,
            semantic_guidance_power=2.0,
        )
    )
    proposed.sac = sac

    tasks = []
    for sid, name, stress, data in scenarios:
        for pname in ["Rule-Based", "Rolling-Horizon", "Enhanced Rolling-Horizon", "LE-DRL-SAC + semantic safety layer"]:
            tasks.append((sid, name, stress, data, pname))

    rows = []
    bar = ProgressBar(len(tasks), label="Unseen-text eval")
    for sid, name, stress, data, pname in tasks:
        if pname == "Rule-Based":
            policy = RuleBasedPolicy()
        elif pname == "Rolling-Horizon":
            policy = RollingHorizonOptimizerPolicy(data=data)
        elif pname == "Enhanced Rolling-Horizon":
            policy = EnhancedRollingHorizonPolicy(data=data)
        else:
            policy = proposed
        row = evaluate(policy, pname, data, sid, name, stress)
        rows.append(row)
        bar.update()
        print(f"\n  {sid} {pname}: reward={row['total_reward_yuan']:.1f} throughput={row['battery_throughput_mwh']:.2f}", flush=True)
    bar.finish()

    write_csv(OUT_PATH, rows)
    summary = pd.DataFrame(rows).groupby("policy", as_index=False)[
        ["total_reward_yuan", "cvar_5_yuan", "battery_throughput_mwh", "high_price_discharge_rate", "low_price_charge_rate"]
    ].mean()
    print("\nUnseen-text cross-scenario summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
