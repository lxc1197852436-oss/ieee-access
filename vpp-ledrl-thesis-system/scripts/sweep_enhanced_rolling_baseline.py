from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.experiment_design import SCENARIOS
from app.core.rolling_optimizer import EnhancedRollingHorizonPolicy
from app.core.simulation import run_policy
from scripts.run_chapter6_experiments import scenario_data

OUT_PATH = ROOT / "outputs" / "chapter6" / "enhanced_rolling_sweep.csv"


class ProgressBar:
    """Tiny stdlib-only progress bar (no tqdm dependency)."""

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
            f"({frac*100:5.1f}%) {elapsed:6.1f}s elapsed, eta {eta:6.1f}s  "
        )
        sys.stdout.flush()

    def finish(self) -> None:
        elapsed = time.time() - self.start
        sys.stdout.write(
            f"\r{self.label} [{'#'*self.width}] {self.total}/{self.total} (100.0%) {elapsed:6.1f}s done\n"
        )
        sys.stdout.flush()

PARAMS = [
    {
        "tag": "light_40_10",
        "extra_cycling_cost_yuan_per_mwh": 40.0,
        "smoothing_penalty_yuan_per_mw": 10.0,
        "terminal_soc_penalty_yuan": 5000.0,
    },
    {
        "tag": "light_70_15",
        "extra_cycling_cost_yuan_per_mwh": 70.0,
        "smoothing_penalty_yuan_per_mw": 15.0,
        "terminal_soc_penalty_yuan": 6000.0,
    },
    {
        "tag": "balanced_100_20",
        "extra_cycling_cost_yuan_per_mwh": 100.0,
        "smoothing_penalty_yuan_per_mw": 20.0,
        "terminal_soc_penalty_yuan": 6500.0,
    },
    {
        "tag": "balanced_130_25",
        "extra_cycling_cost_yuan_per_mwh": 130.0,
        "smoothing_penalty_yuan_per_mw": 25.0,
        "terminal_soc_penalty_yuan": 7000.0,
    },
]

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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> None:
    print("\nSweep summary:")
    for tag in [p["tag"] for p in PARAMS]:
        items = [row for row in rows if row["tag"] == tag]
        reward = sum(float(row["total_reward_yuan"]) for row in items) / len(items)
        cvar = sum(float(row["cvar_5_yuan"]) for row in items) / len(items)
        throughput = sum(float(row["battery_throughput_mwh"]) for row in items) / len(items)
        final_soc = sum(float(row["final_soc"]) for row in items) / len(items)
        print(f"{tag}: reward={reward:.1f} cvar={cvar:.1f} throughput={throughput:.2f} final_soc={final_soc:.3f}")


def main() -> None:
    rows: list[dict] = []
    total = len(PARAMS) * len(SCENARIOS)
    bar = ProgressBar(total, label="Rolling sweep")
    for params in PARAMS:
        for scenario in SCENARIOS:
            data = scenario_data(scenario)
            policy = EnhancedRollingHorizonPolicy(data=data, **{k: v for k, v in params.items() if k != "tag"})
            result = run_policy(policy, data)
            metrics = result["metrics"]
            row = {
                "tag": params["tag"],
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "stress_type": scenario.stress_type,
                "policy": result["policy"],
                **{k: metrics[k] for k in METRIC_KEYS},
            }
            rows.append(row)
            bar.update()
            print(
                f"\n  {params['tag']} {scenario.scenario_id} "
                f"reward={row['total_reward_yuan']:.1f} "
                f"cvar={row['cvar_5_yuan']:.1f} "
                f"throughput={row['battery_throughput_mwh']:.2f} "
                f"final_soc={row['final_soc']:.3f}"
            )
    bar.finish()
    write_csv(OUT_PATH, rows)
    summarize(rows)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
