"""Evaluate 5-seed LE-DRL-SAC (w=0) and SAC-Numeric on S5 + V1-V4 using the
already-trained checkpoints in checkpoints_with_s5/. No retraining; pure eval.
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
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig
from app.core.rl.sac import SACAgent
from app.core.simulation import calculate_metrics

CKPT_DIR = ROOT / "outputs" / "chapter6_long" / "checkpoints_with_s5"
S5_CSV = ROOT / "data" / "processed" / "s5_negative_price_surplus_ai_semantic.csv"
VARIANT_DIR = ROOT / "data" / "processed" / "s5_variants"
OUT_CSV = ROOT / "outputs" / "chapter6_long" / "s5_and_variants_5seed.csv"
SEEDS = [2026, 2031, 2042, 2047, 2053]


def neg_chg(h):
    neg = h["price_yuan_mwh"] < 0
    return float(((h["actual_action_mw"] < -0.1) & neg).sum() / max(1, neg.sum()))


def load(name):
    df = pd.read_csv(name)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def eval_agent(ckpt, model, data, w=0.0):
    sac = SACAgent.load(ckpt)
    if model == "LE-DRL-SAC":
        ag = LEDRLAgent(LEDRLConfig(include_semantic=True, semantic_mode="native",
            name="x", semantic_guidance_weight=w, semantic_guidance_power=2.0, use_ai_semantics=True))
    else:
        ag = LEDRLAgent(LEDRLConfig(include_semantic=False, name="x", use_ai_semantics=True))
    ag.sac = sac
    env = VPPEnv(data); s = env.reset(initial_soc=0.5)
    while not env.done():
        s, _, _, _ = env.step(ag.act(s, deterministic=True))
    h = pd.DataFrame(env.history)
    m = calculate_metrics(h); m["neg_price_charge_rate"] = neg_chg(h)
    return m


def main():
    s5 = load(S5_CSV)
    variants = {vid: load(VARIANT_DIR / f"{vid.lower()}_ai_semantic.csv") for vid in ["V1","V2","V3","V4"]}
    rows = []
    for seed in SEEDS:
        for model in ["LE-DRL-SAC", "SAC-Numeric"]:
            ck = CKPT_DIR / f"{model}_seed{seed}.pt"
            if not ck.exists():
                print(f"missing {ck}"); continue
            # S5
            m = eval_agent(ck, model, s5)
            rows.append({"scenario": "S5", "model": model, "seed": seed, **m})
            # Variants
            for vid, df in variants.items():
                m = eval_agent(ck, model, df)
                rows.append({"scenario": vid, "model": model, "seed": seed, **m})
        print(f"seed {seed} done")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_CSV}")

    # Summary: LE-DRL vs SAC-Numeric per scenario
    print("\n=== 5-seed summary (mean reward) ===")
    summ = out.groupby(["scenario", "model"])["total_reward_yuan"].mean().unstack()
    summ["gap"] = summ["LE-DRL-SAC"] - summ["SAC-Numeric"]
    print(summ.round(1).to_string())

    # Bootstrap CI per scenario
    print("\n=== Bootstrap CI (LE-DRL - SAC-Numeric), 5 seeds ===")
    rng = np.random.default_rng(20260705)
    for sc in ["S5", "V1", "V2", "V3", "V4"]:
        led = out[(out.scenario == sc) & (out.model == "LE-DRL-SAC")]["total_reward_yuan"].values
        num = out[(out.scenario == sc) & (out.model == "SAC-Numeric")]["total_reward_yuan"].values
        d = led - num
        boot = np.array([rng.choice(d, size=5, replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {sc}: gap={d.mean():+.1f}  CI=[{lo:+.1f}, {hi:+.1f}]  favorable={int((d>0).sum())}/5")


if __name__ == "__main__":
    main()
