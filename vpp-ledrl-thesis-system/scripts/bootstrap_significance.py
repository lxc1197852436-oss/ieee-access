"""Bootstrap confidence intervals for the proposed controller vs baselines.

The IEEE Access draft currently reports only mean rewards across three seeds.
Reviewers can challenge the proposed-vs-Rule-Based gap of about 205 yuan as
within noise. This script resamples the per-seed scenario rewards with
replacement to estimate 95% bootstrap confidence intervals for the average
total reward of each model and for the pairwise difference (proposed minus
baseline), so the manuscript can report significance honestly.

Inputs:
  - ieee_pkg/.../figures/evaluation_by_seed.csv  (per-seed scenario rewards)

Outputs:
  - outputs/chapter6_long/bootstrap_ci.csv        (per-model CI)
  - outputs/chapter6_long/bootstrap_diff.csv      (pairwise difference CI)
  - console summary
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
EVAL_PATH = REPO_ROOT / "ieee_pkg" / "ieee_access_vpp_ledrl_20260630" / "figures" / "evaluation_by_seed.csv"
FALLBACK_EVAL_PATH = ROOT / "outputs" / "chapter6_long" / "evaluation_by_seed.csv"
OUT_DIR = ROOT / "outputs" / "chapter6_long"

N_BOOT = 10000
SEED = 2026
ALPHA = 0.05


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


def load_eval() -> pd.DataFrame:
    path = EVAL_PATH if EVAL_PATH.exists() else FALLBACK_EVAL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation_by_seed.csv: {EVAL_PATH} or {FALLBACK_EVAL_PATH}")
    df = pd.read_csv(path)
    return df


def model_average_rewards(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """For each model, return an array of per-seed cross-scenario average rewards."""
    out: dict[str, list[float]] = {}
    for (seed, model), sub in df.groupby(["seed", "model"]):
        out.setdefault(model, []).append(float(sub["total_reward_yuan"].mean()))
    return {m: np.asarray(v, dtype=float) for m, v in out.items()}


def bootstrap_ci(samples: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator) -> tuple[float, float, float]:
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = samples[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(samples.mean()), lo, hi


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator) -> tuple[float, float, float]:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, len(a), size=len(a))
        ib = rng.integers(0, len(b), size=len(b))
        diffs[i] = a[ia].mean() - b[ib].mean()
    point = float(a.mean() - b.mean())
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return point, lo, hi


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    df = load_eval()
    models = model_average_rewards(df)
    rng = np.random.default_rng(SEED)

    print(f"Models: {list(models.keys())}")
    print(f"Seeds per model: { {m: len(v) for m, v in models.items()} }")
    print(f"Bootstrap samples: {N_BOOT}\n")

    # Per-model CI.
    ci_rows: list[dict] = []
    bar = ProgressBar(len(models), label="Bootstrap CI")
    for model, samples in models.items():
        mean, lo, hi = bootstrap_ci(samples, N_BOOT, ALPHA, rng)
        ci_rows.append(
            {
                "model": model,
                "n_seeds": len(samples),
                "mean_reward": round(mean, 2),
                "ci_low": round(lo, 2),
                "ci_high": round(hi, 2),
            }
        )
        bar.update()
    bar.finish()
    write_csv(OUT_DIR / "bootstrap_ci.csv", ci_rows)

    # Pairwise difference vs baselines, using the proposed controller as anchor.
    proposed_name = None
    for cand in [
        "LE-DRL-SAC + semantic safety layer (w=0.75)",
        "LE-DRL-SAC",
        "SAC-Numeric + numeric safety layer",
    ]:
        if cand in models:
            proposed_name = cand
            break
    if proposed_name is None:
        print("No proposed controller found in evaluation_by_seed.csv; skipping pairwise diffs.")
        print("\nPer-model 95% bootstrap CI:")
        print(pd.DataFrame(ci_rows).to_string(index=False))
        return

    proposed = models[proposed_name]
    diff_rows: list[dict] = []
    bar = ProgressBar(len(models) - 1, label="Pairwise diff")
    for model, samples in models.items():
        if model == proposed_name:
            continue
        point, lo, hi = bootstrap_diff_ci(proposed, samples, N_BOOT, ALPHA, rng)
        diff_rows.append(
            {
                "proposed": proposed_name,
                "baseline": model,
                "diff_mean": round(point, 2),
                "ci_low": round(lo, 2),
                "ci_high": round(hi, 2),
                "significant": "yes" if (lo > 0 or hi < 0) else "no",
            }
        )
        bar.update()
    bar.finish()
    write_csv(OUT_DIR / "bootstrap_diff.csv", diff_rows)

    print(f"\nProposed controller: {proposed_name}")
    print("\nPer-model 95% bootstrap CI:")
    print(pd.DataFrame(ci_rows).to_string(index=False))
    print("\nPairwise difference (proposed - baseline), 95% bootstrap CI:")
    print(pd.DataFrame(diff_rows).to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'bootstrap_ci.csv'}")
    print(f"Saved: {OUT_DIR / 'bootstrap_diff.csv'}")


if __name__ == "__main__":
    main()
