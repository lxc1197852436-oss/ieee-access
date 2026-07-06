"""Summarize the 5-seed M5 full ablation into paper-fillable tables with bootstrap CIs.

Reads the three per-source s5_eval.csv files (each written by train_m5_ablation.py),
merges them, and computes:
  - per (source, model, scenario) 5-seed mean reward
  - native vs w/o Text gap per source x scenario with 95% paired-bootstrap CI
  - DeepSeek vs keyword and DeepSeek vs noisy gaps (native) with CI
  - a flat LaTeX-ready table

Bootstrap method matches the main paper: seed-paired paired bootstrap over the 5
matched seed-level reward differences, 20000 resamples, 2.5/97.5 percentiles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "chapter6_long"
SCENARIOS = ["S1", "S2", "S3", "S4"]
SOURCES = ["deepseek", "keyword", "noisy"]
MODELS = ["LE-DRL-SAC", "LE-DRL w/o Text", "SAC-Numeric"]
RNG_SEED = 20260705


def load_all() -> pd.DataFrame:
    frames = []
    for src in SOURCES:
        f = OUT / f"m5_{src}_s1to4" / "s5_eval.csv"
        if not f.exists():
            print(f"missing {f}", file=sys.stderr)
            continue
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df = df[df.scenario_id.isin(SCENARIOS)].copy()
    return df


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 20000, seed: int = RNG_SEED) -> tuple[float, float, float]:
    """Return (mean, lo, hi) of the mean of paired differences."""
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def main() -> None:
    df = load_all()
    if df.empty:
        print("no data"); return

    # Persist merged CSV.
    merged_path = OUT / "m5_full_ablation_5seed.csv"
    df.to_csv(merged_path, index=False, encoding="utf-8-sig")

    # 1) Mean reward table: rows (scenario, model), cols source.
    print("=" * 80)
    print("TABLE 1: 5-seed mean total_reward by (scenario, model) x source")
    print("=" * 80)
    mean_tab = df.groupby(["scenario_id", "model", "semantic_source"])["total_reward_yuan"].mean().reset_index()
    piv = mean_tab.pivot_table(index=["scenario_id", "model"], columns="semantic_source", values="total_reward_yuan")
    piv = piv[SOURCES]
    print(piv.round(1).to_string())

    # 2) native vs w/o Text gap per source x scenario with CI.
    print("\n" + "=" * 80)
    print("TABLE 2: native(LE-DRL-SAC) minus w/o Text, per source x scenario, with 95% CI")
    print("=" * 80)
    rows2 = []
    for src in SOURCES:
        for sc in SCENARIOS:
            sub = df[(df.semantic_source == src) & (df.scenario_id == sc)]
            native = sub[sub.model == "LE-DRL-SAC"].sort_values("seed")["total_reward_yuan"].values
            wot = sub[sub.model == "LE-DRL w/o Text"].sort_values("seed")["total_reward_yuan"].values
            # w/o Text and SAC-Numeric are identical across sources (zeroed/no channel),
            # but for correctness we pair within source.
            if len(native) != len(wot) or len(native) == 0:
                continue
            diffs = native - wot
            mean, lo, hi = paired_bootstrap_ci(diffs)
            favorable = int((diffs > 0).sum())
            rows2.append({"source": src, "scenario": sc, "gap": mean, "ci_lo": lo, "ci_hi": hi,
                          "favorable": f"{favorable}/{len(diffs)}"})
    t2 = pd.DataFrame(rows2)
    print(t2.round(1).to_string(index=False))

    # 3) DeepSeek vs keyword / noisy (native only) with CI, per scenario.
    print("\n" + "=" * 80)
    print("TABLE 3: DeepSeek native minus keyword/noisy native, per scenario, 95% CI")
    print("=" * 80)
    rows3 = []
    for sc in SCENARIOS:
        ds = df[(df.semantic_source == "deepseek") & (df.model == "LE-DRL-SAC") & (df.scenario_id == sc)].sort_values("seed")["total_reward_yuan"].values
        for cmp_src in ["keyword", "noisy"]:
            other = df[(df.semantic_source == cmp_src) & (df.model == "LE-DRL-SAC") & (df.scenario_id == sc)].sort_values("seed")["total_reward_yuan"].values
            if len(ds) != len(other) or len(ds) == 0:
                continue
            diffs = ds - other
            mean, lo, hi = paired_bootstrap_ci(diffs)
            favorable = int((diffs > 0).sum())
            rows3.append({"scenario": sc, "comparison": f"deepseek-{cmp_src}", "gap": mean,
                          "ci_lo": lo, "ci_hi": hi, "favorable": f"{favorable}/{len(diffs)}"})
    t3 = pd.DataFrame(rows3)
    print(t3.round(1).to_string(index=False))

    # 4) Pooled-across-scenario gaps (the headline numbers).
    print("\n" + "=" * 80)
    print("TABLE 4: pooled (all scenarios) paired gaps, 95% CI")
    print("=" * 80)
    rows4 = []
    # native vs w/o Text, pooled across 5 seeds x 4 scenarios = 20 pairs per source
    for src in SOURCES:
        sub = df[df.semantic_source == src]
        native = sub[sub.model == "LE-DRL-SAC"].sort_values(["scenario_id", "seed"])["total_reward_yuan"].values
        wot = sub[sub.model == "LE-DRL w/o Text"].sort_values(["scenario_id", "seed"])["total_reward_yuan"].values
        diffs = native - wot
        mean, lo, hi = paired_bootstrap_ci(diffs)
        rows4.append({"comparison": f"{src}: native-w/oText", "gap": mean, "ci_lo": lo, "ci_hi": hi,
                      "favorable": f"{int((diffs>0).sum())}/{len(diffs)}"})
    # deepseek vs keyword/noisy, pooled native
    ds_all = df[(df.semantic_source == "deepseek") & (df.model == "LE-DRL-SAC")].sort_values(["scenario_id", "seed"])["total_reward_yuan"].values
    for cmp_src in ["keyword", "noisy"]:
        other = df[(df.semantic_source == cmp_src) & (df.model == "LE-DRL-SAC")].sort_values(["scenario_id", "seed"])["total_reward_yuan"].values
        diffs = ds_all - other
        mean, lo, hi = paired_bootstrap_ci(diffs)
        rows4.append({"comparison": f"deepseek-{cmp_src} (native)", "gap": mean, "ci_lo": lo, "ci_hi": hi,
                      "favorable": f"{int((diffs>0).sum())}/{len(diffs)}"})
    t4 = pd.DataFrame(rows4)
    print(t4.round(1).to_string(index=False))

    # Save all tables.
    with pd.ExcelWriter(OUT / "m5_ablation_summary.xlsx") as w:
        piv.to_excel(w, sheet_name="mean_reward")
        t2.to_excel(w, sheet_name="native_vs_wotext", index=False)
        t3.to_excel(w, sheet_name="deepseek_vs_others", index=False)
        t4.to_excel(w, sheet_name="pooled_gaps", index=False)
    print(f"\nSaved: {OUT / 'm5_ablation_summary.xlsx'}")
    print(f"Merged CSV: {merged_path}")


if __name__ == "__main__":
    main()
