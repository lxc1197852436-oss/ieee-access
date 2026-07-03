"""Sweep the semantic safety-layer weight w on the freshly trained LE-DRL-SAC.

The previous prior_weight_sweep_summary.csv was produced by an older LE-DRL-SAC
checkpoint and its numbers no longer match the retrained model. This script
loads the new 3-seed LE-DRL-SAC checkpoints and re-evaluates them with
test-time semantic guidance weights w in {0, 0.25, 0.5, 0.75, 0.9, 1.0}, then
writes a new prior_weight_sweep_summary.csv that the IEEE table builder picks
up as the proposed-controller result.
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
from app.core.experiment_design import SCENARIOS
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig
from app.core.rl.sac import SACAgent
from app.core.simulation import calculate_metrics
from scripts.run_chapter6_experiments import scenario_data

CKPT_DIR = ROOT / "outputs" / "chapter6_long" / "checkpoints"
OUT_BY_SEED = ROOT / "outputs" / "chapter6_long" / "prior_weight_sweep_by_seed.csv"
OUT_SUMMARY = ROOT / "outputs" / "chapter6_long" / "prior_weight_sweep_summary.csv"

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
SEEDS = [2026, 2031, 2042]
GUIDANCE_POWER = 1.6

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
    def __init__(self, total: int, label: str = "", width: int = 28):
        self.total = max(1, total)
        self.label = label
        self.width = width
        self.count = 0
        self.start = time.time()

    def update(self, n: int = 1) -> None:
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

    def finish(self) -> None:
        elapsed = time.time() - self.start
        sys.stdout.write(f"\r{self.label} [{'#'*self.width}] {self.total}/{self.total} (100.0%) {elapsed:5.1f}s done\n")
        sys.stdout.flush()


def load_ledrl_agent(seed: int, weight: float) -> LEDRLAgent:
    ckpt = CKPT_DIR / f"LE-DRL-SAC_seed{seed}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    sac = SACAgent.load(ckpt)
    agent = LEDRLAgent(
        LEDRLConfig(
            include_semantic=True,
            semantic_mode="native",
            name="LE-DRL-SAC + semantic safety layer",
            semantic_guidance_weight=weight,
            semantic_guidance_power=GUIDANCE_POWER,
        )
    )
    agent.sac = sac
    return agent


def evaluate(agent: LEDRLAgent, seed: int, weight: float) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        data = scenario_data(scenario)
        env = VPPEnv(data)
        state = env.reset(initial_soc=0.5)
        while not env.done():
            action = agent.act(state, deterministic=True)
            state, _, _, _ = env.step(action)
        history = pd.DataFrame(env.history)
        metrics = calculate_metrics(history)
        rows.append(
            {
                "seed": seed,
                "weight": weight,
                "scenario": scenario.scenario_id,
                **{k: metrics[k] for k in METRIC_KEYS},
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_rows: list[dict] = []
    total = len(WEIGHTS) * len(SEEDS)
    bar = ProgressBar(total, label="Safety-layer sweep")
    for weight in WEIGHTS:
        for seed in SEEDS:
            agent = load_ledrl_agent(seed, weight)
            rows = evaluate(agent, seed, weight)
            all_rows.extend(rows)
            bar.update()
            avg_reward = float(np.mean([r["total_reward_yuan"] for r in rows]))
            print(f"\n  w={weight:.2f} seed={seed}: avg_reward={avg_reward:.1f}", flush=True)
    bar.finish()

    write_csv(OUT_BY_SEED, all_rows)

    df = pd.DataFrame(all_rows)
    summary = df.groupby("weight", as_index=False).agg(
        total_reward_yuan_mean=("total_reward_yuan", "mean"),
        total_reward_yuan_std=("total_reward_yuan", "std"),
        cvar_5_yuan_mean=("cvar_5_yuan", "mean"),
        battery_throughput_mwh_mean=("battery_throughput_mwh", "mean"),
        high_price_discharge_rate_mean=("high_price_discharge_rate", "mean"),
        low_price_charge_rate_mean=("low_price_charge_rate", "mean"),
    )
    write_csv(OUT_SUMMARY, summary.to_dict("records"))

    print("\nSemantic safety-layer weight sweep (new checkpoints):")
    print(summary.to_string(index=False))
    print(f"\nSaved by-seed: {OUT_BY_SEED}")
    print(f"Saved summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
