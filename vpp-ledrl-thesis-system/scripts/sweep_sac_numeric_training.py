from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_chapter6_long_sac.py"
OUT_PATH = ROOT / "outputs" / "chapter6_long" / "sac_numeric_guidance_sweep.csv"

CONFIGS = [
    {
        "tag": "numeric_guidance_075",
        "reward_mode": "advantage",
        "reward_scale": "0.01",
        "dispatch_aux_reward_scale": "0.00",
        "numeric_actor_loss_weight": "0.00",
        "numeric_guidance_weight": "0.75",
    },
    {
        "tag": "numeric_guidance_090",
        "reward_mode": "advantage",
        "reward_scale": "0.01",
        "dispatch_aux_reward_scale": "0.00",
        "numeric_actor_loss_weight": "0.00",
        "numeric_guidance_weight": "0.90",
    },
    {
        "tag": "numeric_guidance_100",
        "reward_mode": "advantage",
        "reward_scale": "0.01",
        "dispatch_aux_reward_scale": "0.00",
        "numeric_actor_loss_weight": "0.00",
        "numeric_guidance_weight": "1.00",
    },
]


def run_config(config: dict) -> list[dict]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--models",
        "SAC-Numeric + numeric safety layer",
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
        config["reward_mode"],
        "--reward-scale",
        config["reward_scale"],
        "--dispatch-aux-reward-scale",
        config["dispatch_aux_reward_scale"],
        "--semantic-aux-reward-scale",
        "0.0",
        "--semantic-actor-loss-weight",
        "0.0",
        "--numeric-actor-loss-weight",
        config["numeric_actor_loss_weight"],
        "--numeric-guidance-weight",
        config["numeric_guidance_weight"],
        "--log-every",
        "30",
    ]
    print("Running", config["tag"], " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    eval_path = ROOT / "outputs" / "chapter6_long" / "evaluation_by_seed.csv"
    df = pd.read_csv(eval_path)
    rows = []
    for _, row in df.iterrows():
        payload = {"tag": config["tag"], **row.to_dict()}
        rows.append(payload)
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


def summarize(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    summary = df.groupby("tag", as_index=False)[
        ["total_reward_yuan", "cvar_5_yuan", "battery_throughput_mwh", "high_price_discharge_rate", "low_price_charge_rate"]
    ].mean()
    print("\nSAC-Numeric numeric-guidance sweep summary:")
    print(summary.to_string(index=False))


def main() -> None:
    all_rows: list[dict] = []
    for config in CONFIGS:
        rows = run_config(config)
        all_rows.extend(rows)
        write_csv(OUT_PATH, all_rows)
        print(f"Incrementally saved: {OUT_PATH}", flush=True)
        summarize(all_rows)
    print(f"Saved final sweep: {OUT_PATH}")


if __name__ == "__main__":
    main()
