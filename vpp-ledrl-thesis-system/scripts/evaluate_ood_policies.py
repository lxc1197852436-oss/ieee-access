"""Evaluate trained policies on the real-weather OOD test set.

The training scenarios are synthetic 2025-07 data. This script evaluates the
trained SAC-Numeric + numeric safety layer checkpoint on the 2024-07 real-
weather OOD scenarios, together with the rule-based and rolling-horizon
baselines that need no training. The goal is to test whether policies learned
on synthetic data still behave reasonably when temperature and PV come from
real Guangzhou observations.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.environment import VPPEnv
from app.core.policies import RuleBasedPolicy
from app.core.rolling_optimizer import EnhancedRollingHorizonPolicy, RollingHorizonOptimizerPolicy
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig
from app.core.simulation import calculate_metrics

OOD_PATH = ROOT / "data" / "processed" / "ood_vpp_scenarios.csv"
CKPT_DIR = ROOT / "outputs" / "chapter6_long" / "checkpoints"
OUT_PATH = ROOT / "outputs" / "chapter6_long" / "ood_evaluation.csv"

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


def load_ood_scenarios() -> list[tuple[str, str, str, pd.DataFrame]]:
    df = pd.read_csv(OOD_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out = []
    for scenario_id, sub in df.groupby("scenario_id"):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        name = str(sub["scenario_name"].iloc[0])
        stress = str(sub["stress_type"].iloc[0])
        out.append((scenario_id, name, stress, sub))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_policy(policy, data: pd.DataFrame, scenario_id: str, name: str, stress: str, policy_name: str) -> dict:
    env = VPPEnv(data)
    state = env.reset(initial_soc=0.5)
    while not env.done():
        action = policy.act(state) if hasattr(policy, "act") and not hasattr(policy, "encode") else policy.act(state, deterministic=True)
        state, _, _, _ = env.step(action)
    history = pd.DataFrame(env.history)
    metrics = calculate_metrics(history)
    return {
        "scenario_id": scenario_id,
        "scenario_name": name,
        "stress_type": stress,
        "policy": policy_name,
        **{k: metrics[k] for k in METRIC_KEYS},
    }


def main() -> None:
    scenarios = load_ood_scenarios()
    rows: list[dict] = []

    # Trained checkpoint (numeric safety layer).
    ckpt = CKPT_DIR / "SAC-Numeric_numeric_safety_layer_seed2026.pt"
    trained = None
    if ckpt.exists():
        try:
            from app.core.rl.sac import SACAgent

            sac = SACAgent.load(ckpt)
            agent = LEDRLAgent(
                LEDRLConfig(
                    include_semantic=False,
                    name="SAC-Numeric + numeric safety layer",
                    numeric_guidance_weight=1.0,
                    numeric_guidance_power=1.6,
                )
            )
            agent.sac = sac
            trained = agent
            print(f"Loaded checkpoint: {ckpt}")
        except Exception as exc:  # pragma: no cover
            print(f"WARN: could not load checkpoint {ckpt}: {exc}")
    else:
        print(f"WARN: checkpoint not found: {ckpt}")

    for scenario_id, name, stress, data in scenarios:
        for policy, policy_name in [
            (RuleBasedPolicy(), "Rule-Based"),
            (RollingHorizonOptimizerPolicy(data=data), "Rolling-Horizon"),
            (EnhancedRollingHorizonPolicy(data=data), "Enhanced Rolling-Horizon"),
        ]:
            row = evaluate_policy(policy, data, scenario_id, name, stress, policy_name)
            rows.append(row)
            print(
                f"{scenario_id} {policy_name}: reward={row['total_reward_yuan']:.1f} "
                f"throughput={row['battery_throughput_mwh']:.2f} "
                f"high_discharge={row['high_price_discharge_rate']:.3f}"
            )
        if trained is not None:
            row = evaluate_policy(trained, data, scenario_id, name, stress, "SAC-Numeric + numeric safety layer")
            rows.append(row)
            print(
                f"{scenario_id} SAC-Numeric + numeric safety layer: reward={row['total_reward_yuan']:.1f} "
                f"throughput={row['battery_throughput_mwh']:.2f} "
                f"high_discharge={row['high_price_discharge_rate']:.3f}"
            )

    write_csv(OUT_PATH, rows)
    print(f"\nSaved OOD evaluation: {OUT_PATH}")

    print("\nOOD cross-scenario summary:")
    summary = pd.DataFrame(rows).groupby("policy", as_index=False)[
        ["total_reward_yuan", "cvar_5_yuan", "battery_throughput_mwh", "high_price_discharge_rate", "low_price_charge_rate"]
    ].mean()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
