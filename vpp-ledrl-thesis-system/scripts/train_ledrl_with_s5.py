"""Retrain LE-DRL-SAC on S1-S5 (adds the unseen S5 negative-price event).

This mirrors scripts/train_chapter6_long_sac.py but extends the training set
to include S5 (negative-price deep surplus), so the learning agent is exposed
to the unseen event category during fine-tuning. The hand-crafted semantic
safety prior in ledrl_agent._semantic_prior_action is NOT modified: it keeps
its four original branches, so on S5 it has no negative-price branch and falls
through to its default thresholds.

Checkpoints are written to a separate directory so the original Chapter 6
results remain reproducible.

Narrative: a deployed prior is fixed at design time; when an unanticipated
event category appears in operation, the prior cannot be rewritten overnight
but the learning agent can be fine-tuned on the new event's cached semantic
scores. This script realizes that fine-tuning step.
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
    print(exc)
    raise SystemExit(1) from exc

from app.core.environment import VPPEnv
from app.core.experiment_design import SCENARIOS
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig
from app.core.simulation import calculate_metrics
from scripts.run_chapter6_experiments import scenario_data
from scripts.train_chapter6_long_sac import (
    learning_reward,
    semantic_auxiliary_reward,
    dispatch_alignment_reward,
    set_seed,
    ProgressBar,
)


OUT_DIR = ROOT / "outputs" / "chapter6_long"
CKPT_DIR = OUT_DIR / "checkpoints_with_s5"
AI_S5 = ROOT / "data" / "processed" / "s5_negative_price_surplus_ai_semantic.csv"
AI_SCENARIOS = ROOT / "data" / "processed" / "chapter6_ai_semantic_scenarios.csv"


def load_s5_training_data() -> pd.DataFrame:
    df = pd.read_csv(AI_S5)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def training_sets_with_s5(periods_per_scenario: int) -> list[tuple[str, pd.DataFrame]]:
    rows = []
    # S1-S4 from the AI-semantic scenario file (same as --use-ai-semantics).
    ai = pd.read_csv(AI_SCENARIOS)
    ai["timestamp"] = pd.to_datetime(ai["timestamp"])
    for sc in SCENARIOS:
        sub = ai[ai["scenario_id"] == sc.scenario_id].head(periods_per_scenario).reset_index(drop=True)
        rows.append((sc.scenario_id, sub))
    # S5 (the unseen event).
    s5 = load_s5_training_data().head(periods_per_scenario).reset_index(drop=True)
    rows.append(("S5", s5))
    return rows


def train_one(agent, train_data, seed, args) -> list[dict]:
    logs = []
    bar = ProgressBar(args.episodes, label=f"train {agent.name} seed={seed}")
    for ep in range(1, args.episodes + 1):
        scenario_id, rows = train_data[(ep + seed) % len(train_data)]
        env = VPPEnv(rows)
        initial_soc = 0.35 + 0.3 * ((ep - 1) % 5) / 4
        state = env.reset(initial_soc=initial_soc)
        ep_reward = 0.0
        losses = []
        while not env.done():
            state_vec = agent.encode(state)
            action = agent.act(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            shaped = learning_reward(reward, info, env, args)
            shaped += dispatch_alignment_reward(state, info["actual_action_mw"], agent, args)
            shaped += semantic_auxiliary_reward(state, info["actual_action_mw"], agent, args)
            agent.sac.add_transition(state_vec, action, shaped, agent.encode(next_state), done)
            if agent.sac.total_steps % args.update_interval == 0:
                for _ in range(args.updates_per_step):
                    linfo = agent.sac.update()
                    if linfo:
                        losses.append(linfo)
            agent.sac.total_steps += 1
            state = next_state
            ep_reward += reward
        mean_critic = float(np.mean([x["critic_loss"] for x in losses])) if losses else float("nan")
        mean_actor = float(np.mean([x["actor_loss"] for x in losses])) if losses else float("nan")
        logs.append({
            "seed": seed, "model": agent.name, "episode": ep,
            "training_scenario_id": scenario_id,
            "episode_reward_yuan": ep_reward,
            "mean_critic_loss": mean_critic, "mean_actor_loss": mean_actor,
        })
        bar.update()
        if ep == 1 or ep % args.log_every == 0 or ep == args.episodes:
            print(f"\n  {agent.name} seed={seed} ep={ep}/{args.episodes} raw={ep_reward:.1f} "
                  f"updates={len(losses)} replay={agent.sac.replay.size}")
    bar.finish()
    return logs


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--seeds", type=str, default="2026,2031,2042")
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
    p.add_argument("--semantic-guidance-weight", type=float, default=0.0)
    p.add_argument("--semantic-guidance-power", type=float, default=2.0)
    p.add_argument("--numeric-guidance-weight", type=float, default=0.0)
    p.add_argument("--numeric-guidance-power", type=float, default=1.6)
    p.add_argument("--semantic-aux-reward-scale", type=float, default=0.35)
    p.add_argument("--semantic-actor-loss-weight", type=float, default=3.0)
    p.add_argument("--models", default="LE-DRL-SAC,SAC-Numeric")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    train_data = training_sets_with_s5(args.train_periods_per_scenario)
    print(f"Training scenarios: {[sid for sid, _ in train_data]}  seeds={seeds}")

    all_logs = []
    checkpoints = []
    for seed in seeds:
        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            set_seed(seed)
            common = dict(
                hidden_dim=args.hidden_dim, batch_size=args.batch_size,
                warmup_steps=args.warmup_steps, lr=args.lr, gamma=args.gamma,
                device=args.device, use_ai_semantics=True,
                numeric_actor_loss_weight=0.0,
                semantic_guidance_weight=args.semantic_guidance_weight,
                semantic_guidance_power=args.semantic_guidance_power,
                numeric_guidance_weight=args.numeric_guidance_weight,
                numeric_guidance_power=args.numeric_guidance_power,
            )
            if model == "SAC-Numeric":
                agent = LEDRLAgent(LEDRLConfig(include_semantic=False, name="SAC-Numeric", **common))
            elif model == "LE-DRL-SAC":
                agent = LEDRLAgent(LEDRLConfig(
                    include_semantic=True, semantic_mode="native", name="LE-DRL-SAC",
                    semantic_actor_loss_weight=args.semantic_actor_loss_weight, **common,
                ))
            else:
                raise ValueError(model)
            logs = train_one(agent, train_data, seed, args)
            all_logs.extend(logs)
            ckpt = CKPT_DIR / f"{safe(model)}_seed{seed}.pt"
            agent.sac.save(ckpt)
            checkpoints.append({"seed": seed, "model": model, "checkpoint": str(ckpt)})
            print(f"saved {ckpt}")

    payload = {"args": vars(args), "seeds": seeds,
               "models": args.models.split(","),
               "training_scenarios": [sid for sid, _ in train_data],
               "checkpoints": checkpoints}
    (CKPT_DIR.parent / "train_with_s5_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Checkpoints in {CKPT_DIR}")


if __name__ == "__main__":
    main()
