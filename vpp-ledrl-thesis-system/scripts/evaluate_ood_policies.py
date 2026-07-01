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
import time
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
            f"({frac*100:5.1f}%) {elapsed:5.1f}s elapsed, eta {eta:5.1f}s"
        )
        sys.stdout.flush()

    def finish(self) -> None:
        elapsed = time.time() - self.start
        sys.stdout.write(f"\r{self.label} [{'#'*self.width}] {self.total}/{self.total} (100.0%) {elapsed:5.1f}s done\n")
        sys.stdout.flush()


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

    # Build the full evaluation task list so the progress bar covers everything.
    tasks: list[tuple] = []
    for scenario_id, name, stress, data in scenarios:
        for policy_name in ["Rule-Based", "Rolling-Horizon", "Enhanced Rolling-Horizon"] + (
            ["SAC-Numeric + numeric safety layer"] if trained is not None else []
        ):
            tasks.append((scenario_id, name, stress, data, policy_name))

    bar = ProgressBar(len(tasks), label="OOD eval")
    for scenario_id, name, stress, data, policy_name in tasks:
        if policy_name == "Rule-Based":
            policy = RuleBasedPolicy()
        elif policy_name == "Rolling-Horizon":
            policy = RollingHorizonOptimizerPolicy(data=data)
        elif policy_name == "Enhanced Rolling-Horizon":
            policy = EnhancedRollingHorizonPolicy(data=data)
        else:
            policy = trained
        row = evaluate_policy(policy, data, scenario_id, name, stress, policy_name)
        rows.append(row)
        bar.update()
        print(
            f"\n  {scenario_id} {policy_name}: reward={row['total_reward_yuan']:.1f} "
            f"throughput={row['battery_throughput_mwh']:.2f} "
            f"high_discharge={row['high_price_discharge_rate']:.3f}"
        )
    bar.finish()

    write_csv(OUT_PATH, rows)
    print(f"\nSaved OOD evaluation: {OUT_PATH}")

    print("\nOOD cross-scenario summary:")
    summary = pd.DataFrame(rows).groupby("policy", as_index=False)[
        ["total_reward_yuan", "cvar_5_yuan", "battery_throughput_mwh", "high_price_discharge_rate", "low_price_charge_rate"]
    ].mean()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
