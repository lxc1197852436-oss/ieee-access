from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.experiment_design import SCENARIOS
from app.core.policies import RuleBasedPolicy
from app.core.rolling_optimizer import EnhancedRollingHorizonPolicy, RollingHorizonOptimizerPolicy
from app.core.simulation import run_policy
from scripts.run_chapter6_experiments import scenario_data

OUT_PATH = ROOT / "outputs" / "chapter6" / "enhanced_rolling_validation.csv"


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


def validate_metrics(row: dict) -> None:
    for key in METRIC_KEYS:
        value = float(row[key])
        if not math.isfinite(value):
            raise AssertionError(f"Non-finite metric {key} for {row['scenario_id']} {row['policy']}: {value}")
    throughput = float(row["battery_throughput_mwh"])
    if row["policy"] == "Enhanced Rolling-Horizon" and throughput > 80.0:
        print(
            f"WARNING: Enhanced Rolling-Horizon throughput is still high in {row['scenario_id']}: {throughput:.3f} MWh"
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    policies = sorted({row["policy"] for row in rows})
    print("\nCross-scenario summary:")
    for policy in policies:
        items = [row for row in rows if row["policy"] == policy]
        reward = sum(float(row["total_reward_yuan"]) for row in items) / len(items)
        cvar = sum(float(row["cvar_5_yuan"]) for row in items) / len(items)
        throughput = sum(float(row["battery_throughput_mwh"]) for row in items) / len(items)
        final_soc = sum(float(row["final_soc"]) for row in items) / len(items)
        print(
            f"{policy}: reward={reward:.1f} cvar={cvar:.1f} "
            f"throughput={throughput:.2f} final_soc={final_soc:.3f}"
        )


def main() -> None:
    rows: list[dict] = []
    for scenario in SCENARIOS:
        data = scenario_data(scenario)
        policies = [
            RuleBasedPolicy(),
            RollingHorizonOptimizerPolicy(data=data),
            EnhancedRollingHorizonPolicy(data=data),
        ]
        for policy in policies:
            result = run_policy(policy, data)
            metrics = result["metrics"]
            row = {
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "stress_type": scenario.stress_type,
                "policy": result["policy"],
                **{k: metrics[k] for k in METRIC_KEYS},
            }
            validate_metrics(row)
            rows.append(row)
            print(
                scenario.scenario_id,
                result["policy"],
                f"reward={row['total_reward_yuan']:.1f}",
                f"cvar={row['cvar_5_yuan']:.1f}",
                f"throughput={row['battery_throughput_mwh']:.2f}",
                f"final_soc={row['final_soc']:.3f}",
            )
    write_csv(OUT_PATH, rows)
    print_summary(rows)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
