"""Train LE-DRL-SAC + SAC-Numeric on the multi-week real DE-LU price scenario (N4 fix).

Trains 5 seeds x 3 sources (deepseek/keyword/noisy) x 2 models under the
RELAXED new-event-adaptation config (w=0, semantic_actor_loss=0), matching the
S5 config in the main paper. The training set is the full 21-day real DE-LU
series (s5_real_price_multiweek, 2013 steps). Checkpoints are written to
outputs/chapter6_long/m9_multiweek_<source>_relaxed/checkpoints/ for
eval_m9_multiweek_real_price.py to consume.

Usage:
  python scripts/train_m9_multiweek.py --semantic-source deepseek
  python scripts/train_m9_multiweek.py --semantic-source keyword
  python scripts/train_m9_multiweek.py --semantic-source noisy
"""
from __future__ import annotations

import argparse
import csv as _csv
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
    print(exc)
    raise SystemExit(1) from exc

from app.core.config import VPPConfig
from app.core.environment import VPPEnv
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig
from scripts.train_chapter6_long_sac import (  # noqa: E402
    ProgressBar, learning_reward, semantic_auxiliary_reward,
    dispatch_alignment_reward, set_seed,
)

PROCESSED = ROOT / "data" / "processed"
OUT_BASE = ROOT / "outputs" / "chapter6_long"

SOURCE_FILES = {
    "deepseek": "s5_real_price_multiweek_ai_semantic.csv",
    "keyword": "s5_real_price_multiweek_ai_semantic_keyword.csv",
    "noisy": "s5_real_price_multiweek_ai_semantic_noisy.csv",
}


def load_data(source: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / SOURCE_FILES[source])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def train_one(agent, data, seed, args, env_config=None) -> list:
    logs = []
    bar = ProgressBar(args.episodes, label=f"train {agent.name} seed={seed}")
    for ep in range(1, args.episodes + 1):
        env = VPPEnv(data, config=env_config)
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
        logs.append({"seed": seed, "model": agent.name, "episode": ep,
                     "episode_reward_eur": ep_reward,
                     "mean_critic_loss": mean_critic, "mean_actor_loss": mean_actor})
        bar.update()
        if ep == 1 or ep % args.log_every == 0 or ep == args.episodes:
            print(f"\n  {agent.name} seed={seed} ep={ep}/{args.episodes} raw={ep_reward:.1f} "
                  f"updates={len(losses)} replay={agent.sac.replay.size}")
    bar.finish()
    return logs


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def main() -> None:
    p = argparse.ArgumentParser(description="Train LE-DRL-SAC on multi-week real DE-LU price (N4 fix).")
    p.add_argument("--semantic-source", choices=list(SOURCE_FILES.keys()), default="deepseek")
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--seeds", type=str, default="2026,2031,2042,2047,2053")
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
    p.add_argument("--semantic-actor-loss-weight", type=float, default=0.0)
    p.add_argument("--models", default="LE-DRL-SAC,SAC-Numeric")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    out_dir = OUT_BASE / f"m9_multiweek_{args.semantic_source}_relaxed"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    data = load_data(args.semantic_source)
    print(f"source={args.semantic_source}  data_rows={len(data)}  seeds={seeds}")

    env_config = VPPConfig()
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
                    semantic_actor_loss_weight=args.semantic_actor_loss_weight, **common))
            else:
                raise ValueError(model)
            logs = train_one(agent, data, seed, args, env_config=env_config)
            all_logs.extend(logs)
            ckpt = ckpt_dir / f"{safe(model)}_seed{seed}.pt"
            agent.sac.save(ckpt)
            checkpoints.append({"seed": seed, "model": model, "checkpoint": str(ckpt)})
            print(f"  EVAL {args.semantic_source} {model} seed={seed}: last_ep_reward={logs[-1]['episode_reward_eur']:.1f}")

    with (out_dir / "training_logs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=list(all_logs[0].keys()))
        w.writeheader(); w.writerows(all_logs)
    payload = {"args": vars(args), "seeds": seeds, "semantic_source": args.semantic_source,
               "models": args.models.split(","), "checkpoints": checkpoints,
               "source_file": SOURCE_FILES[args.semantic_source]}
    (out_dir / "m9_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
