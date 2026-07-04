"""Train LE-DRL-SAC (relaxed regularizer) and SAC-Numeric on S1-S5, then
evaluate on the four S5 parameter variants (V1-V4) as held-out unseen events.

This is the robustness check for the S5 unseen-event result. The training set
is S1-S5 (the original five scenarios, including the S5 noon negative-price
event). The four variants V1-V4 are held out: they differ from S5 in negative-
price depth, season, or time-of-day, and V4 additionally replaces PV surplus
with night wind surplus. If LE-DRL-SAC exceeds SAC-Numeric on all four
variants, the unseen-event adaptation is not an artifact of the S5 scenario.

Honesty rule: single run per seed. No tuning after seeing numbers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(1) from exc

from app.core.environment import VPPEnv
from app.core.experiment_design import SCENARIOS
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig
from app.core.simulation import calculate_metrics
from scripts.train_chapter6_long_sac import (
    learning_reward, semantic_auxiliary_reward, dispatch_alignment_reward,
    set_seed, ProgressBar,
)

OUT_DIR = ROOT / "outputs" / "chapter6_long"
CKPT_DIR = OUT_DIR / "checkpoints_s5_variants"
AI_SCENARIOS = ROOT / "data" / "processed" / "chapter6_ai_semantic_scenarios.csv"
AI_S5 = ROOT / "data" / "processed" / "s5_negative_price_surplus_ai_semantic.csv"
VARIANT_DIR = ROOT / "data" / "processed" / "s5_variants"
OUT_CSV = OUT_DIR / "s5_variants_evaluation.csv"
SEEDS = [2026, 2031, 2042]


def load_train_set(periods: int) -> list[tuple[str, pd.DataFrame]]:
    rows = []
    ai = pd.read_csv(AI_SCENARIOS); ai["timestamp"] = pd.to_datetime(ai["timestamp"])
    for sc in SCENARIOS:
        rows.append((sc.scenario_id, ai[ai["scenario_id"] == sc.scenario_id].head(periods).reset_index(drop=True)))
    s5 = pd.read_csv(AI_S5); s5["timestamp"] = pd.to_datetime(s5["timestamp"])
    rows.append(("S5", s5.head(periods).reset_index(drop=True)))
    return rows


def load_variants() -> list[tuple[str, pd.DataFrame]]:
    out = []
    for vid in ["V1", "V2", "V3", "V4"]:
        df = pd.read_csv(VARIANT_DIR / f"{vid.lower()}_ai_semantic.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        out.append((vid, df))
    return out


def neg_price_charge_rate(history: pd.DataFrame) -> float:
    neg = history["price_yuan_mwh"] < 0
    charge = history["actual_action_mw"] < -0.1
    return float((charge & neg).sum() / max(1, neg.sum()))


def train_one(agent, train_data, seed, args) -> None:
    bar = ProgressBar(args.episodes, label=f"train {agent.name} seed={seed}")
    for ep in range(1, args.episodes + 1):
        sid, rows = train_data[(ep + seed) % len(train_data)]
        env = VPPEnv(rows)
        state = env.reset(initial_soc=0.35 + 0.3 * ((ep - 1) % 5) / 4)
        while not env.done():
            sv = agent.encode(state)
            a = agent.act(state, deterministic=False)
            ns, r, done, info = env.step(a)
            shaped = learning_reward(r, info, env, args)
            shaped += dispatch_alignment_reward(state, info["actual_action_mw"], agent, args)
            shaped += semantic_auxiliary_reward(state, info["actual_action_mw"], agent, args)
            agent.sac.add_transition(sv, a, shaped, agent.encode(ns), done)
            if agent.sac.total_steps % args.update_interval == 0:
                for _ in range(args.updates_per_step):
                    agent.sac.update()
            agent.sac.total_steps += 1
            state = ns
        bar.update()
    bar.finish()


def eval_variants(agent) -> list[dict]:
    out = []
    for vid, data in load_variants():
        env = VPPEnv(data); state = env.reset(initial_soc=0.5)
        while not env.done():
            a = agent.act(state, deterministic=True)
            state, _, _, _ = env.step(a)
        h = pd.DataFrame(env.history)
        m = calculate_metrics(h); m["neg_price_charge_rate"] = neg_price_charge_rate(h)
        out.append({"variant_id": vid, **m})
    return out


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--train-periods-per-scenario", type=int, default=288)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--update-interval", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--warmup-steps", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--reward-mode", default="advantage")
    p.add_argument("--reward-scale", type=float, default=0.01)
    p.add_argument("--dispatch-aux-reward-scale", type=float, default=0.0)
    p.add_argument("--low-price-threshold", type=float, default=260.0)
    p.add_argument("--high-price-threshold", type=float, default=520.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--semantic-aux-reward-scale", type=float, default=0.35)
    p.add_argument("--semantic-actor-loss-weight", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train_data = load_train_set(args.train_periods_per_scenario)
    print(f"Training on S1-S5 (variants V1-V4 held out). seeds={SEEDS}")

    all_eval = []
    for seed in SEEDS:
        for model_name, include_sem in [("LE-DRL-SAC", True), ("SAC-Numeric", False)]:
            set_seed(seed)
            common = dict(hidden_dim=args.hidden_dim, batch_size=args.batch_size,
                          warmup_steps=args.warmup_steps, lr=args.lr, gamma=args.gamma,
                          device=args.device, use_ai_semantics=True,
                          semantic_guidance_weight=0.0, semantic_guidance_power=2.0,
                          numeric_guidance_weight=0.0, numeric_guidance_power=1.6)
            if include_sem:
                agent = LEDRLAgent(LEDRLConfig(include_semantic=True, semantic_mode="native",
                    name=model_name, semantic_actor_loss_weight=args.semantic_actor_loss_weight, **common))
            else:
                agent = LEDRLAgent(LEDRLConfig(include_semantic=False, name=model_name, **common))
            train_one(agent, train_data, seed, args)
            ckpt = CKPT_DIR / f"{safe(model_name)}_seed{seed}.pt"
            agent.sac.save(ckpt)
            ev = eval_variants(agent)
            for r in ev:
                r.update({"model": model_name, "seed": seed})
                all_eval.append(r)
            print(f"  {model_name} seed={seed}: " +
                  " ".join([f"{r['variant_id']}={r['total_reward_yuan']:.0f}" for r in ev]))

    df = pd.DataFrame(all_eval)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_CSV}")

    print("\n=== Variants summary (mean over 3 seeds) ===")
    summ = df.groupby(["model", "variant_id"])["total_reward_yuan"].mean().unstack()
    print(summ.round(1).to_string())

    # Bootstrap CI per variant: LE-DRL - SAC-Numeric
    print("\n=== LE-DRL vs SAC-Numeric per variant (3 seeds) ===")
    rng = np.random.default_rng(20260704)
    for vid in ["V1", "V2", "V3", "V4"]:
        led = df[(df["model"] == "LE-DRL-SAC") & (df["variant_id"] == vid)]["total_reward_yuan"].values
        num = df[(df["model"] == "SAC-Numeric") & (df["variant_id"] == vid)]["total_reward_yuan"].values
        diffs = led - num
        boot = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        wins = int((diffs > 0).sum())
        print(f"  {vid}: gap={diffs.mean():+.1f}  CI=[{lo:+.1f}, {hi:+.1f}]  seeds favorable={wins}/3")


if __name__ == "__main__":
    main()
