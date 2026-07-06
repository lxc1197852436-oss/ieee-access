"""Drive the full M5 ablation: 5 seeds x 3 sources x 3 models, main config (S1-S4, w=0.9, actor_loss=3.0).

This is the final, paper-fillable M5 ablation. It confirms the core claim ("SAC reads
LLM semantics") by comparing native LE-DRL-SAC against w/o Text (zeroed semantic dims)
and SAC-Numeric (no semantic channel) under the main config where semantic_actor_loss
is active -- the only config that actually forces the actor to read the semantic state.

Outputs per source go to outputs/chapter6_long/m5_<source>_s1to4/ and a combined
5-seed CSV is assembled at the end for the paper's M5 ablation table.

Usage:
  python scripts/run_m5_full_ablation.py            # all 3 sources
  python scripts/run_m5_full_ablation.py --sources deepseek keyword   # subset
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = "2026,2031,2042,2047,2053"
MODELS = "LE-DRL-SAC,LE-DRL w/o Text,SAC-Numeric"


def run_source(source: str, seeds: str, episodes: int) -> int:
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train_m5_ablation.py"),
        "--semantic-source", source,
        "--no-include-s5",
        "--semantic-actor-loss-weight", "3.0",
        "--semantic-guidance-weight", "0.9",
        "--semantic-guidance-power", "2.0",
        "--seeds", seeds,
        "--episodes", str(episodes),
        "--models", MODELS,
        "--log-every", "80",
    ]
    print(f"\n{'='*70}\n[run] source={source} seeds={seeds} models={MODELS}\n{'='*70}", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=["deepseek", "keyword", "noisy"])
    p.add_argument("--seeds", default=SEEDS)
    p.add_argument("--episodes", type=int, default=80)
    args = p.parse_args()

    for src in args.sources:
        rc = run_source(src, args.seeds, args.episodes)
        if rc != 0:
            print(f"[warn] source={src} exited with rc={rc}, continuing to next source")

    # Assemble combined 5-seed eval CSV across sources for the paper table.
    import pandas as pd
    rows = []
    for src in args.sources:
        f = ROOT / "outputs" / "chapter6_long" / f"m5_{src}_s1to4" / "s5_eval.csv"
        if f.exists():
            rows.append(pd.read_csv(f))
    if rows:
        combined = pd.concat(rows, ignore_index=True)
        out = ROOT / "outputs" / "chapter6_long" / "m5_full_ablation_5seed.csv"
        combined.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n[done] combined 5-seed eval -> {out}")
        # Quick per-scenario mean summary
        print("\n=== mean total_reward by source x model x scenario (5 seeds) ===")
        piv = combined.groupby(["semantic_source", "model", "scenario_id"])["total_reward_yuan"].mean().reset_index()
        print(piv.pivot_table(index=["scenario_id", "model"], columns="semantic_source", values="total_reward_yuan").round(1).to_string())
    else:
        print("[warn] no s5_eval.csv files found to combine")


if __name__ == "__main__":
    main()
