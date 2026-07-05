"""Statistical robustness check for LE-DRL-SAC vs baselines.

Fixes the issue in bootstrap_significance.py (which resampled 3 seed-level means,
yielding implausibly narrow CIs). Here we use:

  * paired bootstrap on the per-seed scenario-level reward differences
    (proposed - baseline, one difference per seed per scenario), which preserves
    the seed-matched pairing and reflects scenario-level variability;
  * a paired Wilcoxon signed-rank test as a non-parametric cross-check (does not
    assume a normal difference distribution and works at small n).

For the 3-seed S1-S4 main comparison the paired differences are 3 seeds x 4 scenarios
= 12 points (the unit of resampling). For the 5-seed S5/V*/real-price comparisons
the unit is 5 seeds x (scenarios) per variant.

Usage:  python scripts/bootstrap_stats_paired.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "chapter6_long"
N_BOOT = 20000
RNG_SEED = 2026
ALPHA = 0.05

# Proposed controller name as it appears in the 3-seed evaluation file
PROPOSED_3S = "LE-DRL-SAC + semantic safety layer (w=0.9)"
BASELINES_3S = ["SAC-Numeric", "LE-DRL w/o Text", "LE-DRL-SAC"]


def paired_bootstrap_diff(diffs: np.ndarray, n_boot: int, rng: np.random.Generator):
    """ diffs: paired differences (proposed - baseline), one per (seed,scenario).
    Returns point estimate, 95% percentile CI, and the bootstrap SE. """
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diffs[idx].mean()
    lo = float(np.quantile(means, ALPHA / 2))
    hi = float(np.quantile(means, 1 - ALPHA / 2))
    return float(diffs.mean()), lo, hi, float(means.std(ddof=1))


def wilcoxon_check(diffs: np.ndarray):
    n = len(diffs)
    if n < 5:
        # Wilcoxon is unreliable below n>=5 (ties dominate); report but flag.
        w, p = stats.wilcoxon(diffs) if n >= 3 else (float("nan"), float("nan"))
    else:
        w, p = stats.wilcoxon(diffs)
    return float(w) if not np.isnan(w) else float("nan"), float(p)


def run_3seed_main():
    print("=" * 70)
    print("3-seed S1-S4 main comparison (paired bootstrap, 20000 resamples)")
    print("=" * 70)
    eval_path = OUT_DIR / "evaluation_by_seed.csv"
    sweep_path = OUT_DIR / "prior_weight_sweep_by_seed.csv"
    df = pd.read_csv(eval_path)
    # rename for uniform join keys
    df = df.rename(columns={"scenario_id": "scenario"})
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        proposed = sweep[sweep["weight"].round(2) == 0.9].copy()
        prop_rows = pd.DataFrame({
            "seed": proposed["seed"].astype(int),
            "scenario": proposed["scenario"],
            "model": PROPOSED_3S,
            "total_reward_yuan": proposed["total_reward_yuan"],
        })
        df = pd.concat([df[["seed", "scenario", "model", "total_reward_yuan"]],
                        prop_rows], ignore_index=True)

    # Pivot: rows = (seed, scenario), cols = model, values = total_reward
    pv = df.pivot_table(index=["seed", "scenario"], columns="model",
                        values="total_reward_yuan", aggfunc="first")
    if PROPOSED_3S not in pv.columns:
        print(f"  !! proposed model '{PROPOSED_3S}' not found; available:")
        print(pv.columns.tolist())
        return
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for base in BASELINES_3S:
        if base not in pv.columns:
            continue
        sub = pv[[PROPOSED_3S, base]].dropna()
        diffs = (sub[PROPOSED_3S] - sub[base]).to_numpy()
        point, lo, hi, se = paired_bootstrap_diff(diffs, N_BOOT, rng)
        n = len(diffs)
        if n >= 3:
            try:
                w, p = wilcoxon_check(diffs)
            except ValueError:
                w, p = float("nan"), float("nan")
        else:
            w, p = float("nan"), float("nan")
        rows.append({
            "comparison": f"proposed - {base}",
            "n_paired": n,
            "diff_mean_yuan": round(point, 2),
            "ci_low": round(lo, 2),
            "ci_high": round(hi, 2),
            "ci_width": round(hi - lo, 2),
            "bootstrap_se": round(se, 2),
            "wilcoxon_p": None if np.isnan(p) else f"{p:.4f}",
            "significant": "yes" if (lo > 0 or hi < 0) else "no",
        })
        print(f"  proposed - {base:18s}  n={n:2d}  diff={point:8.1f}  "
              f"95% CI [{lo:8.1f}, {hi:8.1f}]  width={hi-lo:6.1f}  "
              f"p={'  n/a' if np.isnan(p) else f'{p:.4f}'}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "bootstrap_paired_3seed.csv", index=False)
    print(f"  saved -> {OUT_DIR / 'bootstrap_paired_3seed.csv'}")


def run_5seed_s5_variants():
    print("\n" + "=" * 70)
    print("5-seed S5 + V1-V4 (paired bootstrap per variant, 20000 resamples)")
    print("=" * 70)
    path = OUT_DIR / "s5_and_variants_5seed.csv"
    df = pd.read_csv(path)
    # Models in this file: LE-DRL-SAC (relaxed reg, w=0) vs SAC-Numeric
    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    for scen in ["S5", "V1", "V2", "V3", "V4"]:
        sub = df[df["scenario"] == scen]
        pv = sub.pivot_table(index="seed", columns="model",
                             values="total_reward_yuan", aggfunc="first")
        if "LE-DRL-SAC" not in pv.columns or "SAC-Numeric" not in pv.columns:
            print(f"  {scen}: missing model columns {pv.columns.tolist()}")
            continue
        d = pv[["LE-DRL-SAC", "SAC-Numeric"]].dropna()
        diffs = (d["LE-DRL-SAC"] - d["SAC-Numeric"]).to_numpy()
        point, lo, hi, se = paired_bootstrap_diff(diffs, N_BOOT, rng)
        n = len(diffs)
        try:
            _, p = wilcoxon_check(diffs)
        except ValueError:
            p = float("nan")
        rows.append({
            "scenario": scen,
            "n_seeds": n,
            "diff_mean_yuan": round(point, 2),
            "ci_low": round(lo, 2),
            "ci_high": round(hi, 2),
            "ci_width": round(hi - lo, 2),
            "wilcoxon_p": None if np.isnan(p) else f"{p:.4f}",
            "significant": "yes" if (lo > 0 or hi < 0) else "no",
        })
        print(f"  {scen}  n={n}  diff={point:8.1f}  95% CI [{lo:8.1f}, {hi:8.1f}]  "
              f"width={hi-lo:6.1f}  p={'  n/a' if np.isnan(p) else f'{p:.4f}'}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "bootstrap_paired_5seed_s5.csv", index=False)
    print(f"  saved -> {OUT_DIR / 'bootstrap_paired_5seed_s5.csv'}")


def run_5seed_real_price():
    print("\n" + "=" * 70)
    print("5-seed real DE-LU price validation (paired bootstrap)")
    print("=" * 70)
    # locate the real-price 5-seed csv if present
    cand = list(OUT_DIR.glob("*real_price*5seed*.csv")) + \
           list(OUT_DIR.glob("*real*price*5*.csv"))
    if not cand:
        print("  (no 5-seed real-price csv found under outputs/chapter6_long)")
        return
    df = pd.read_csv(cand[0])
    print(f"  loaded {cand[0].name}, columns: {df.columns.tolist()}")
    rng = np.random.default_rng(RNG_SEED + 2)
    if "model" in df.columns and "scenario" in df.columns and "seed" in df.columns:
        for scen in df["scenario"].unique():
            sub = df[df["scenario"] == scen]
            pv = sub.pivot_table(index="seed", columns="model",
                                 values="total_reward_yuan", aggfunc="first")
            cols = [c for c in pv.columns if "LE-DRL" in c or "ledrl" in c.lower()]
            base = [c for c in pv.columns if "SAC-Numeric" in c]
            if not cols or not base:
                continue
            d = pv[[cols[0], base[0]]].dropna()
            diffs = (d[cols[0]] - d[base[0]]).to_numpy()
            point, lo, hi, se = paired_bootstrap_diff(diffs, N_BOOT, rng)
            try:
                _, p = wilcoxon_check(diffs)
            except ValueError:
                p = float("nan")
            print(f"  {scen}  n={len(diffs)}  diff={point:8.1f}  "
                  f"95% CI [{lo:8.1f}, {hi:8.1f}]  p={'n/a' if np.isnan(p) else f'{p:.4f}'}")


if __name__ == "__main__":
    run_3seed_main()
    run_5seed_s5_variants()
    run_5seed_real_price()
