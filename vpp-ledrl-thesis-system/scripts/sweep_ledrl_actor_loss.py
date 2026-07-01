"""Quick diagnostic: find a semantic_actor_loss_weight that prevents LE-DRL-SAC
policy collapse (throughput stuck at 3.04, no discharge).

The full 80-episode 3-seed run recently collapsed to the no-action policy,
matching SAC-Numeric instead of the earlier -202345 result. This script tries
several actor-loss weights on a single seed with 30 episodes to find a
setting where LE-DRL-SAC throughput rises above 3.04 MWh, before committing to
a full re-run.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_chapter6_long_sac.py"
OUT_PATH = ROOT / "outputs" / "chapter6_long" / "ledrl_actor_loss_sweep.csv"


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


WEIGHTS = ["0.25", "0.75", "1.50", "3.00"]


def run_one(weight: str) -> list[dict]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--models",
        "LE-DRL-SAC",
        "--episodes",
        "30",
        "--seeds",
        "2026",
        "--train-periods-per-scenario",
        "288",
        "--update-interval",
        "4",
        "--batch-size",
        "64",
        "--warmup-steps",
        "256",
        "--hidden-dim",
        "64",
        "--reward-mode",
        "advantage",
        "--reward-scale",
        "0.01",
        "--semantic-aux-reward-scale",
        "0.35",
        "--semantic-actor-loss-weight",
        weight,
        "--numeric-actor-loss-weight",
        "0.0",
        "--numeric-guidance-weight",
        "0.0",
        "--semantic-guidance-weight",
        "0.0",
        "--log-every",
        "30",
    ]
    print(f"\nRunning semantic_actor_loss_weight={weight}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    eval_path = ROOT / "outputs" / "chapter6_long" / "evaluation_by_seed.csv"
    df = pd.read_csv(eval_path)
    rows = []
    for _, row in df.iterrows():
        rows.append({"semantic_actor_loss_weight": weight, **row.to_dict()})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


import csv


def summarize(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    summary = df.groupby("semantic_actor_loss_weight", as_index=False)[
        ["total_reward_yuan", "battery_throughput_mwh", "high_price_discharge_rate", "low_price_charge_rate"]
    ].mean()
    print("\nLE-DRL-SAC actor-loss diagnostic summary:")
    print(summary.to_string(index=False))
    print("\nPass criterion: throughput > 3.04 and high_price_discharge_rate > 0")


def main() -> None:
    all_rows: list[dict] = []
    bar = ProgressBar(len(WEIGHTS), label="Actor-loss sweep")
    for w in WEIGHTS:
        rows = run_one(w)
        all_rows.extend(rows)
        write_csv(OUT_PATH, all_rows)
        bar.update()
        print(f"\n  saved -> {OUT_PATH}", flush=True)
        summarize(all_rows)
    bar.finish()
    print(f"Saved final: {OUT_PATH}")


if __name__ == "__main__":
    main()
