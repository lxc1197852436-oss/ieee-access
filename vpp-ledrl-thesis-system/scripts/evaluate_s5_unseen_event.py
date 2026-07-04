"""Evaluate LE-DRL-SAC (retrained on S1-S5) on the S5 unseen-event scenario.

Compares, on S5 only:
  - LE-DRL-SAC actor only (w=0.0): pure learned policy, no safety prior.
  - LE-DRL-SAC + semantic safety layer (w=0.9): the proposed controller.
  - LE-DRL-SAC + semantic safety layer (w=1.0): pure hand-crafted prior
    (no learned component). This is the upper bound the main paper worries
    about. On S5 the prior has NO negative-price branch, so this is where
    the learned component should matter most.
  - SAC-Numeric (retrained on S1-S5): learned policy without text.
  - Rule-Based: deterministic baseline.

Reports total reward, CVaR 5%, throughput, high-price discharge rate,
negative-price charge rate (a behavioral indicator specific to S5), and
final SOC. Three seeds; bootstrap CI on the proposed-vs-prior gap.

Honesty rule: this script runs ONCE. Results are written to CSV exactly as
produced. No hyperparameter tuning after seeing the numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.environment import VPPEnv
from app.core.policies import RuleBasedPolicy
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig
from app.core.rl.sac import SACAgent
from app.core.simulation import calculate_metrics

S5_CSV = ROOT / "data" / "processed" / "s5_negative_price_surplus_ai_semantic.csv"
CKPT_DIR = ROOT / "outputs" / "chapter6_long" / "checkpoints_with_s5"
OUT_CSV = ROOT / "outputs" / "chapter6_long" / "s5_unseen_event_evaluation.csv"
SEEDS = [2026, 2031, 2042]
WEIGHTS = [0.0, 0.9, 1.0]


def load_s5() -> pd.DataFrame:
    df = pd.read_csv(S5_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def neg_price_charge_rate(history: pd.DataFrame) -> float:
    neg = history["price_yuan_mwh"] < 0
    charge = history["actual_action_mw"] < -0.1
    if neg.sum() == 0:
        return 0.0
    return float((charge & neg).sum() / neg.sum())


def run_episode(agent, data) -> dict:
    env = VPPEnv(data)
    state = env.reset(initial_soc=0.5)
    while not env.done():
        action = agent.act(state, deterministic=True)
        state, _, _, _ = env.step(action)
    history = pd.DataFrame(env.history)
    m = calculate_metrics(history)
    m["neg_price_charge_rate"] = neg_price_charge_rate(history)
    return m


def run_rulebased(data) -> dict:
    env = VPPEnv(data)
    state = env.reset(initial_soc=0.5)
    pol = RuleBasedPolicy()
    while not env.done():
        action = pol.act(state)
        state, _, _, _ = env.step(action)
    history = pd.DataFrame(env.history)
    m = calculate_metrics(history)
    m["neg_price_charge_rate"] = neg_price_charge_rate(history)
    return m


def make_agent(model_name: str, weight: float, sac_ckpt: Path) -> LEDRLAgent:
    sac = SACAgent.load(sac_ckpt)
    if model_name == "LE-DRL-SAC":
        agent = LEDRLAgent(LEDRLConfig(
            include_semantic=True, semantic_mode="native",
            name=f"LE-DRL-SAC (w={weight})",
            semantic_guidance_weight=weight, semantic_guidance_power=2.0,
            use_ai_semantics=True,
        ))
    elif model_name == "SAC-Numeric":
        agent = LEDRLAgent(LEDRLConfig(
            include_semantic=False, name="SAC-Numeric",
            use_ai_semantics=True,
        ))
    else:
        raise ValueError(model_name)
    agent.sac = sac
    return agent


def main() -> None:
    data = load_s5()
    rows = []

    # Rule-Based (deterministic, no seed)
    m = run_rulebased(data)
    rows.append({"model": "Rule-Based", "seed": 0, "weight": None, **m})
    print(f"Rule-Based: reward={m['total_reward_yuan']:.1f}")

    for seed in SEEDS:
        # SAC-Numeric (retrained)
        ck_num = CKPT_DIR / f"SAC-Numeric_seed{seed}.pt"
        if ck_num.exists():
            ag = make_agent("SAC-Numeric", 0.0, ck_num)
            m = run_episode(ag, data)
            rows.append({"model": "SAC-Numeric", "seed": seed, "weight": 0.0, **m})
            print(f"SAC-Numeric seed={seed}: reward={m['total_reward_yuan']:.1f}")

        # LE-DRL-SAC at three weights (same checkpoint, different blend)
        ck_led = CKPT_DIR / f"LE-DRL-SAC_seed{seed}.pt"
        if not ck_led.exists():
            print(f"  missing {ck_led}, skipping")
            continue
        for w in WEIGHTS:
            ag = make_agent("LE-DRL-SAC", w, ck_led)
            m = run_episode(ag, data)
            rows.append({"model": "LE-DRL-SAC", "seed": seed, "weight": w, **m})
            print(f"LE-DRL-SAC seed={seed} w={w}: reward={m['total_reward_yuan']:.1f} "
                  f"CVaR={m['cvar_5_yuan']:.1f} thrpt={m['battery_throughput_mwh']:.2f} "
                  f"negChg={m['neg_price_charge_rate']:.3f}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_CSV}")

    # Summary: mean over seeds
    print("\n=== S5 summary (mean over 3 seeds) ===")
    summary = df.groupby(["model", "weight"], dropna=False)[
        ["total_reward_yuan", "cvar_5_yuan", "battery_throughput_mwh",
         "high_price_discharge_rate", "neg_price_charge_rate", "final_soc"]
    ].mean().round(1)
    print(summary.to_string())

    # Bootstrap CI on proposed (w=0.9) vs pure prior (w=1.0), seed-level means
    prop = df[(df["model"] == "LE-DRL-SAC") & (df["weight"] == 0.9)]["total_reward_yuan"].values
    prior = df[(df["model"] == "LE-DRL-SAC") & (df["weight"] == 1.0)]["total_reward_yuan"].values
    if len(prop) >= 2 and len(prior) >= 2:
        rng = np.random.default_rng(20260704)
        diffs = prop - prior
        n_boot = 10000
        boot_means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)])
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        print(f"\nProposed(w=0.9) - Prior(w=1.0) on S5: mean={diffs.mean():.1f} "
              f"95% CI=[{lo:.1f}, {hi:.1f}]  (positive = SAC component helps)")


if __name__ == "__main__":
    main()
