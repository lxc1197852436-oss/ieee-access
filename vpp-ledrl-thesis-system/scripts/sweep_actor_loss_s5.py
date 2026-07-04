"""Sweep semantic_actor_loss_weight on S1-S5 training.

Diagnosis showed that weight=3.0 turns the SAC actor into an imitator of
semantic_target_actions, drowning the critic's Q signal (actor_loss~3.1 vs
critic_loss~0.24). On S5 (unseen negative-price event), this prevents the
actor from learning the cross-interval SOC management that SAC-Numeric learns
freely. This script sweeps the regularizer weight to find whether a lower
value lets LE-DRL-SAC recover adaptivity without losing the known-scenario
performance.

Trains on S1-S5 for each weight in {0.0, 0.5, 1.5, 3.0}, three seeds, and
evaluates on ALL FIVE scenarios (not just S5) so we can see the trade-off
between regularizer strength and adaptivity. Checkpoints go to a separate
directory; original results remain reproducible.

Honesty rule: single run per (weight, seed). No tuning after seeing numbers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
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
CKPT_BASE = OUT_DIR / "checkpoints_actor_loss_sweep"
AI_SCENARIOS = ROOT / "data" / "processed" / "chapter6_ai_semantic_scenarios.csv"
AI_S5 = ROOT / "data" / "processed" / "s5_negative_price_surplus_ai_semantic.csv"

WEIGHTS = [0.0, 0.5, 1.5, 3.0]
SEEDS = [2026, 2031, 2042]


def load_all_scenarios(periods: int) -> list[tuple[str, pd.DataFrame]]:
    rows = []
    ai = pd.read_csv(AI_SCENARIOS)
    ai["timestamp"] = pd.to_datetime(ai["timestamp"])
    for sc in SCENARIOS:
        sub = ai[ai["scenario_id"] == sc.scenario_id].head(periods).reset_index(drop=True)
        rows.append((sc.scenario_id, sub))
    s5 = pd.read_csv(AI_S5)
    s5["timestamp"] = pd.to_datetime(s5["timestamp"])
    rows.append(("S5", s5.head(periods).reset_index(drop=True)))
    return rows


def eval_scenarios(agent) -> list[dict]:
    out = []
    # S1-S4 from AI scenarios
    ai = pd.read_csv(AI_SCENARIOS)
    ai["timestamp"] = pd.to_datetime(ai["timestamp"])
    for sc in SCENARIOS:
        data = ai[ai["scenario_id"] == sc.scenario_id].reset_index(drop=True)
        env = VPPEnv(data); state = env.reset(initial_soc=0.5)
        while not env.done():
            a = agent.act(state, deterministic=True)
            state, _, _, _ = env.step(a)
        m = calculate_metrics(pd.DataFrame(env.history))
        out.append({"scenario_id": sc.scenario_id, **m})
    # S5
    s5 = pd.read_csv(AI_S5); s5["timestamp"] = pd.to_datetime(s5["timestamp"])
    env = VPPEnv(s5); state = env.reset(initial_soc=0.5)
    while not env.done():
        a = agent.act(state, deterministic=True)
        state, _, _, _ = env.step(a)
    m = calculate_metrics(pd.DataFrame(env.history))
    out.append({"scenario_id": "S5", **m})
    return out


def train_one(agent, train_data, seed, args) -> None:
    bar = ProgressBar(args.episodes, label=f"train {agent.name} seed={seed}")
    for ep in range(1, args.episodes + 1):
        scenario_id, rows = train_data[(ep + seed) % len(train_data)]
        env = VPPEnv(rows)
        initial_soc = 0.35 + 0.3 * ((ep - 1) % 5) / 4
        state = env.reset(initial_soc=initial_soc)
        while not env.done():
            sv = agent.encode(state)
            action = agent.act(state, deterministic=False)
            ns, reward, done, info = env.step(action)
            shaped = learning_reward(reward, info, env, args)
            shaped += dispatch_alignment_reward(state, info["actual_action_mw"], agent, args)
            shaped += semantic_auxiliary_reward(state, info["actual_action_mw"], agent, args)
            agent.sac.add_transition(sv, action, shaped, agent.encode(ns), done)
            if agent.sac.total_steps % args.update_interval == 0:
                for _ in range(args.updates_per_step):
                    agent.sac.update()
            agent.sac.total_steps += 1
            state = ns
        bar.update()
    bar.finish()


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


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
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()

    CKPT_BASE.mkdir(parents=True, exist_ok=True)
    train_data = load_all_scenarios(args.train_periods_per_scenario)
    print(f"Training on: {[sid for sid,_ in train_data]}  weights={WEIGHTS}  seeds={SEEDS}")

    all_eval = []
    for w in WEIGHTS:
        for seed in SEEDS:
            set_seed(seed)
            name = f"LE-DRL-SAC_w{w}"
            agent = LEDRLAgent(LEDRLConfig(
                include_semantic=True, semantic_mode="native", name=name,
                semantic_actor_loss_weight=w, use_ai_semantics=True,
                semantic_guidance_weight=0.0, semantic_guidance_power=2.0,
                hidden_dim=args.hidden_dim, batch_size=args.batch_size,
                warmup_steps=args.warmup_steps, lr=args.lr, gamma=args.gamma,
                device=args.device,
            ))
            train_one(agent, train_data, seed, args)
            ckpt = CKPT_BASE / f"{safe(name)}_seed{seed}.pt"
            agent.sac.save(ckpt)
            ev = eval_scenarios(agent)
            for r in ev:
                r.update({"weight": w, "seed": seed, "model": name})
                all_eval.append(r)
            s5_r = [r for r in ev if r["scenario_id"] == "S5"][0]
            avg = np.mean([r["total_reward_yuan"] for r in ev])
            print(f"  w={w} seed={seed}: S5={s5_r['total_reward_yuan']:.1f}  5-scenario avg={avg:.1f}")

    df = pd.DataFrame(all_eval)
    out_csv = OUT_DIR / "actor_loss_sweep_evaluation.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_csv}")

    print("\n=== Sweep summary (mean over 3 seeds) ===")
    summ = df.groupby(["weight", "scenario_id"])["total_reward_yuan"].mean().unstack()
    summ["avg"] = summ.mean(axis=1)
    print(summ.round(1).to_string())


if __name__ == "__main__":
    main()
