"""Train the B3 gated mixture-of-experts SAC on S1-S5 and evaluate on variants.

Training loop:
  1. Encode state s.
  2. SAC-prior samples a_prior; SAC-free samples a_free.
  3. Gate produces w; blended action a = (1-w)*a_free + w*a_prior applied to env.
  4. Each SAC stores its own (s, its_action, reward, s', done) in its own buffer.
     Reward is the shaped reward; both SACs see the same environmental reward
     so their Q functions are comparable.
  5. Update each SAC (critic + actor with its own regularizer).
  6. Update gate with BCE on soft target = sigmoid(0.05 * (Q_prior - Q_free)).

Evaluation: on S1-S5 (known) and V1-V4 (unseen variants), report reward and
the mean gate weight w, to see whether the gate learns high-w on known events
and low-w on unseen events. The key diagnostic is the gate-weight pattern
across scenarios, not just the reward.
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
from app.core.rl.gated_moe import GatedMoEAgent, GatedMoEConfig
from app.core.rl.state_encoder import StateEncoder, StateEncoderConfig
from app.core.simulation import calculate_metrics
from scripts.train_chapter6_long_sac import (
    learning_reward, semantic_auxiliary_reward, dispatch_alignment_reward,
    set_seed, ProgressBar,
)

OUT_DIR = ROOT / "outputs" / "chapter6_long"
CKPT_DIR = OUT_DIR / "checkpoints_gated_moe"
AI_SCENARIOS = ROOT / "data" / "processed" / "chapter6_ai_semantic_scenarios.csv"
AI_S5 = ROOT / "data" / "processed" / "s5_negative_price_surplus_ai_semantic.csv"
VARIANT_DIR = ROOT / "data" / "processed" / "s5_variants"
OUT_CSV = OUT_DIR / "gated_moe_evaluation.csv"
SEEDS = [2026, 2031, 2042]


def load_train(periods: int) -> list[tuple[str, pd.DataFrame]]:
    rows = []
    ai = pd.read_csv(AI_SCENARIOS); ai["timestamp"] = pd.to_datetime(ai["timestamp"])
    for sc in SCENARIOS:
        rows.append((sc.scenario_id, ai[ai["scenario_id"] == sc.scenario_id].head(periods).reset_index(drop=True)))
    s5 = pd.read_csv(AI_S5); s5["timestamp"] = pd.to_datetime(s5["timestamp"])
    rows.append(("S5", s5.head(periods).reset_index(drop=True)))
    return rows


def load_eval_set() -> list[tuple[str, str, pd.DataFrame]]:
    """Return (scenario_id, known/unseen, data) for evaluation."""
    out = []
    ai = pd.read_csv(AI_SCENARIOS); ai["timestamp"] = pd.to_datetime(ai["timestamp"])
    for sc in SCENARIOS:
        out.append((sc.scenario_id, "known", ai[ai["scenario_id"] == sc.scenario_id].reset_index(drop=True)))
    s5 = pd.read_csv(AI_S5); s5["timestamp"] = pd.to_datetime(s5["timestamp"])
    out.append(("S5", "known_train", s5.reset_index(drop=True)))
    for vid in ["V1", "V2", "V3", "V4"]:
        df = pd.read_csv(VARIANT_DIR / f"{vid.lower()}_ai_semantic.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        out.append((vid, "unseen", df))
    return out


def neg_price_charge_rate(history: pd.DataFrame) -> float:
    neg = history["price_yuan_mwh"] < 0
    charge = history["actual_action_mw"] < -0.1
    return float((charge & neg).sum() / max(1, neg.sum()))


def train_one(agent: GatedMoEAgent, train_data, seed: int, args) -> None:
    bar = ProgressBar(args.episodes, label=f"train {agent.name} seed={seed}")
    for ep in range(1, args.episodes + 1):
        sid, rows = train_data[(ep + seed) % len(train_data)]
        env = VPPEnv(rows)
        state = env.reset(initial_soc=0.35 + 0.3 * ((ep - 1) % 5) / 4)
        while not env.done():
            sv = agent.encode(state)
            sv_t = torch.as_tensor(sv, dtype=torch.float32, device=agent.device).unsqueeze(0)
            # each expert samples its own action
            with torch.no_grad():
                a_prior, _ = agent.sac_prior.actor.sample(sv_t)
                a_free, _ = agent.sac_free.actor.sample(sv_t)
                w = float(agent.gate(sv_t).cpu().numpy().reshape(-1)[0])
            a_prior_f = float(a_prior.cpu().numpy().reshape(-1)[0])
            a_free_f = float(a_free.cpu().numpy().reshape(-1)[0])
            blended = (1.0 - w) * a_free_f + w * a_prior_f
            blended = float(np.clip(blended, -2.0, 2.0))
            ns, reward, done, info = env.step(blended)
            shaped = learning_reward(reward, info, env, args)
            shaped += dispatch_alignment_reward(state, info["actual_action_mw"], agent, args)
            shaped += semantic_auxiliary_reward(state, info["actual_action_mw"], agent, args)
            # each expert stores its own action as exploration
            agent.sac_prior.add_transition(sv, a_prior_f, shaped, agent.encode(ns), done)
            agent.sac_free.add_transition(sv, a_free_f, shaped, agent.encode(ns), done)
            if agent.sac_prior.total_steps % args.update_interval == 0:
                for _ in range(args.updates_per_step):
                    agent.sac_prior.update()
                    agent.sac_free.update()
                    # gate update uses a batch from prior's buffer (both see same states)
                    if agent.sac_prior.replay.size >= 64:
                        b = agent.sac_prior.replay.sample(64)
                        agent.update_gate(b.states)
            agent.sac_prior.total_steps += 1
            agent.sac_free.total_steps += 1
            state = ns
        bar.update()
        if ep == 1 or ep % args.log_every == 0 or ep == args.episodes:
            print(f"\n  {agent.name} seed={seed} ep={ep}/{args.episodes} replay={agent.sac_prior.replay.size}")
    bar.finish()


def eval_one(agent: GatedMoEAgent, data: pd.DataFrame) -> dict:
    env = VPPEnv(data); state = env.reset(initial_soc=0.5)
    ws = []
    while not env.done():
        sv = agent.encode(state)
        a, w = agent.act(state, deterministic=True, return_w=True)
        ws.append(w)
        state, _, _, _ = env.step(a)
    h = pd.DataFrame(env.history)
    m = calculate_metrics(h); m["neg_price_charge_rate"] = neg_price_charge_rate(h)
    m["mean_gate_w"] = float(np.mean(ws))
    return m


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--train-periods-per-scenario", type=int, default=288)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--update-interval", type=int, default=8)
    p.add_argument("--reward-mode", default="advantage")
    p.add_argument("--reward-scale", type=float, default=0.01)
    p.add_argument("--dispatch-aux-reward-scale", type=float, default=0.0)
    p.add_argument("--low-price-threshold", type=float, default=260.0)
    p.add_argument("--high-price-threshold", type=float, default=520.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--semantic-aux-reward-scale", type=float, default=0.35)
    p.add_argument("--prior-actor-loss-weight", type=float, default=3.0)
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train_data = load_train(args.train_periods_per_scenario)
    encoder = StateEncoder(StateEncoderConfig(include_semantic=True, semantic_mode="native"))
    print(f"Training Gated-MoE on S1-S5. seeds={SEEDS}")

    all_eval = []
    for seed in SEEDS:
        set_seed(seed)
        cfg = GatedMoEConfig(
            state_dim=encoder.feature_dim, hidden_dim=64, device=args.device,
            prior_actor_loss_weight=args.prior_actor_loss_weight,
            free_actor_loss_weight=0.0, name="Gated-MoE",
        )
        agent = GatedMoEAgent(cfg, encoder)
        train_one(agent, train_data, seed, args)
        ckpt = CKPT_DIR / f"gated_moe_seed{seed}.pt"
        agent.save(ckpt)
        print(f"saved {ckpt}")
        for sid, kind, data in load_eval_set():
            m = eval_one(agent, data)
            m.update({"scenario_id": sid, "kind": kind, "seed": seed})
            all_eval.append(m)
            print(f"  seed={seed} {sid}({kind}): reward={m['total_reward_yuan']:.1f} w={m['mean_gate_w']:.3f}")

    df = pd.DataFrame(all_eval)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_CSV}")

    print("\n=== Gated-MoE summary (mean over 3 seeds) ===")
    summ = df.groupby(["scenario_id", "kind"]).agg(
        reward=("total_reward_yuan", "mean"), w=("mean_gate_w", "mean"),
        negChg=("neg_price_charge_rate", "mean")).round({"reward": 1, "w": 3, "negChg": 2})
    print(summ.to_string())


if __name__ == "__main__":
    main()
