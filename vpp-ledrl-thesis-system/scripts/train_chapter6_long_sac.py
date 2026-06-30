from __future__ import annotations

import argparse
import csv
import json
import random
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
except ModuleNotFoundError as exc:
    print(exc)
    print("Install ML dependencies first: pip install -r requirements-ml.txt")
    raise SystemExit(1)

from app.core.environment import VPPEnv
from app.core.experiment_design import SCENARIOS
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig
from app.core.simulation import calculate_metrics
from scripts.run_chapter6_experiments import scenario_data


OUT_DIR = ROOT / "outputs" / "chapter6_long"
AI_SCENARIOS = ROOT / "data" / "processed" / "chapter6_ai_semantic_scenarios.csv"


def parse_seeds(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def build_agent(model_name: str, args: argparse.Namespace) -> LEDRLAgent:
    common = {
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "lr": args.lr,
        "gamma": args.gamma,
        "device": args.device,
        "use_ai_semantics": args.use_ai_semantics,
    }
    if model_name == "SAC-Numeric":
        return LEDRLAgent(LEDRLConfig(include_semantic=False, name="SAC-Numeric", **common))
    if model_name == "LE-DRL-SAC":
        return LEDRLAgent(
            LEDRLConfig(
                include_semantic=True,
                semantic_mode="native",
                name="LE-DRL-SAC",
                semantic_guidance_weight=args.semantic_guidance_weight,
                semantic_guidance_power=args.semantic_guidance_power,
                semantic_actor_loss_weight=args.semantic_actor_loss_weight,
                **common,
            )
        )
    if model_name == "LE-DRL w/o Text":
        return LEDRLAgent(
            LEDRLConfig(include_semantic=True, semantic_mode="zero", name="LE-DRL w/o Text", **common)
        )
    raise ValueError(f"Unknown model: {model_name}")


def load_scenario_for_training(scenario, use_ai_semantics: bool) -> pd.DataFrame:
    if use_ai_semantics:
        if not AI_SCENARIOS.exists():
            raise FileNotFoundError(
                f"AI semantic scenario data not found: {AI_SCENARIOS}. "
                "Run scripts/build_ai_semantic_features.py first."
            )
        data = pd.read_csv(AI_SCENARIOS)
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        return data[data["scenario_id"] == scenario.scenario_id].reset_index(drop=True)
    return scenario_data(scenario)


def training_sets(periods_per_scenario: int, use_ai_semantics: bool) -> list[tuple[str, pd.DataFrame]]:
    rows = []
    for scenario in SCENARIOS:
        data = load_scenario_for_training(scenario, use_ai_semantics).head(periods_per_scenario).reset_index(drop=True)
        rows.append((scenario.scenario_id, data))
    return rows


def no_action_reward(info: dict, env: VPPEnv) -> float:
    cfg = env.config
    pv = float(info["pv_mw"])
    load = float(info["load_mw"])
    price = float(info["price_yuan_mwh"])
    revenue = (pv - load) * cfg.dt_hours * price
    curtailment_mwh = max(0.0, pv - load) * cfg.dt_hours
    curtailment_cost = curtailment_mwh * cfg.curtailment_penalty_yuan_per_mwh
    return revenue - curtailment_cost


def learning_reward(raw_reward: float, info: dict, env: VPPEnv, args: argparse.Namespace) -> float:
    if args.reward_mode == "raw":
        value = raw_reward
    elif args.reward_mode == "advantage":
        value = raw_reward - no_action_reward(info, env)
    else:
        raise ValueError(f"Unknown reward mode: {args.reward_mode}")
    return float(value * args.reward_scale)


def semantic_auxiliary_reward(state: dict, action_mw: float, agent: LEDRLAgent, args: argparse.Namespace) -> float:
    """Risk-aware auxiliary reward for LE-DRL-SAC training only.

    The final dispatch action is still produced by the SAC actor. This term only
    gives the actor a denser learning signal when text events indicate price
    spike, load pressure, or renewable-curtailment risk.
    """
    if (
        args.semantic_aux_reward_scale <= 0.0
        or not agent.config.include_semantic
        or agent.config.semantic_mode != "native"
    ):
        return 0.0

    sem = state["semantic"]
    hour = float(state["hour"])
    price = float(state["price_yuan_mwh"])
    pv_surplus = float(state["pv_mw"]) - float(state["load_mw"])
    action_norm = float(np.clip(action_mw / agent.config.action_limit, -1.0, 1.0))

    charge_need = 0.0
    discharge_need = 0.0
    if price < 260.0:
        charge_need = max(charge_need, 0.7)
    if 10.0 <= hour <= 15.5 and (pv_surplus > 0.0 or sem.renewable_curtailment_score > 0.55):
        charge_need = max(charge_need, 0.35 + 0.65 * sem.renewable_curtailment_score)

    if price > 520.0:
        discharge_need = max(discharge_need, 0.8)
    if 18.0 <= hour <= 22.0 and max(sem.price_spike_score, sem.load_pressure_score) > 0.45:
        discharge_need = max(discharge_need, 0.35 + 0.65 * max(sem.price_spike_score, sem.load_pressure_score))

    target = discharge_need - charge_need
    if abs(target) < 1e-6:
        return 0.0
    alignment = target * action_norm
    return float(args.semantic_aux_reward_scale * alignment)


def train_one(
    agent: LEDRLAgent,
    train_data: list[tuple[str, pd.DataFrame]],
    seed: int,
    args: argparse.Namespace,
) -> list[dict]:
    logs: list[dict] = []
    update_count = 0
    for ep in range(1, args.episodes + 1):
        scenario_id, rows = train_data[(ep + seed) % len(train_data)]
        env = VPPEnv(rows)
        initial_soc = 0.35 + 0.3 * ((ep - 1) % 5) / 4
        state = env.reset(initial_soc=initial_soc)
        ep_reward = 0.0
        ep_learning_reward = 0.0
        losses = []
        while not env.done():
            state_vec = agent.encode(state)
            action = agent.act(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            shaped_reward = learning_reward(reward, info, env, args)
            shaped_reward += semantic_auxiliary_reward(state, info["actual_action_mw"], agent, args)
            agent.sac.add_transition(state_vec, action, shaped_reward, agent.encode(next_state), done)
            if agent.sac.total_steps % args.update_interval == 0:
                for _ in range(args.updates_per_step):
                    info = agent.sac.update()
                    if info:
                        losses.append(info)
                        update_count += 1
            agent.sac.total_steps += 1
            state = next_state
            ep_reward += reward
            ep_learning_reward += shaped_reward
        mean_critic_loss = float(np.mean([x["critic_loss"] for x in losses])) if losses else np.nan
        mean_actor_loss = float(np.mean([x["actor_loss"] for x in losses])) if losses else np.nan
        row = {
            "seed": seed,
            "model": agent.name,
            "episode": ep,
            "training_scenario_id": scenario_id,
            "episode_reward_yuan": ep_reward,
            "episode_learning_reward": ep_learning_reward,
            "updates": len(losses),
            "replay_size": agent.sac.replay.size,
            "mean_critic_loss": mean_critic_loss,
            "mean_actor_loss": mean_actor_loss,
        }
        logs.append(row)
        if ep == 1 or ep % args.log_every == 0 or ep == args.episodes:
            print(
                f"{agent.name} seed={seed} episode={ep}/{args.episodes} "
                f"raw_reward={ep_reward:.1f} learning_reward={ep_learning_reward:.2f} "
                f"updates={len(losses)} replay={agent.sac.replay.size}"
            )
    return logs


def evaluate_agent(agent: LEDRLAgent, seed: int) -> tuple[list[dict], list[dict]]:
    metrics_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    for scenario in SCENARIOS:
        data = load_scenario_for_training(scenario, agent.config.use_ai_semantics)
        env = VPPEnv(data)
        state = env.reset(initial_soc=0.5)
        while not env.done():
            action = agent.act(state, deterministic=True)
            state, _, _, _ = env.step(action)
        history = pd.DataFrame(env.history)
        metrics = calculate_metrics(history)
        metrics_rows.append(
            {
                "seed": seed,
                "model": agent.name,
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "stress_type": scenario.stress_type,
                **metrics,
            }
        )
        for idx, row in enumerate(env.history):
            trajectory_rows.append(
                {
                    "seed": seed,
                    "model": agent.name,
                    "scenario_id": scenario.scenario_id,
                    "scenario_name": scenario.name,
                    "step": idx,
                    **row,
                }
            )
    return metrics_rows, trajectory_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    metric_cols = [
        "total_reward_yuan",
        "mean_reward_yuan",
        "cvar_5_yuan",
        "final_soc",
        "battery_throughput_mwh",
        "high_price_discharge_rate",
        "low_price_charge_rate",
    ]
    grouped = (
        df.groupby(["scenario_id", "scenario_name", "model"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([part for part in col if part]).rstrip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
    return grouped.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 6 long SAC/LE-DRL-SAC training and ablation.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="2026,2031,2042")
    parser.add_argument("--train-periods-per-scenario", type=int, default=288, help="288 equals 3 days at 15-min resolution.")
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--update-interval", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--reward-mode", choices=["advantage", "raw"], default="advantage")
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-ai-semantics", action="store_true", help="Use AI-enriched semantic columns if available.")
    parser.add_argument(
        "--semantic-guidance-weight",
        type=float,
        default=0.0,
        help="Optional blend weight for a deterministic semantic prior; keep 0 for thesis-grade SAC action output.",
    )
    parser.add_argument(
        "--semantic-guidance-power",
        type=float,
        default=1.6,
        help="Maximum MW of semantic prior action before blending.",
    )
    parser.add_argument(
        "--semantic-aux-reward-scale",
        type=float,
        default=0.35,
        help="Training-only semantic risk alignment reward for LE-DRL-SAC.",
    )
    parser.add_argument(
        "--semantic-actor-loss-weight",
        type=float,
        default=0.25,
        help="Actor regularization weight that makes LE-DRL-SAC learn risk-consistent actions from semantic features.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    train_data = training_sets(args.train_periods_per_scenario, args.use_ai_semantics)
    all_logs: list[dict] = []
    all_metrics: list[dict] = []
    all_trajectories: list[dict] = []
    checkpoints: list[dict] = []

    models = ["SAC-Numeric", "LE-DRL-SAC", "LE-DRL w/o Text"]
    for seed in seeds:
        for model in models:
            set_seed(seed)
            agent = build_agent(model, args)
            logs = train_one(agent, train_data, seed, args)
            all_logs.extend(logs)
            ckpt = OUT_DIR / "checkpoints" / f"{safe_name(model)}_seed{seed}.pt"
            agent.sac.save(ckpt)
            metrics, trajectories = evaluate_agent(agent, seed)
            all_metrics.extend(metrics)
            all_trajectories.extend(trajectories)
            checkpoints.append({"seed": seed, "model": model, "checkpoint": str(ckpt)})
            print(f"evaluated {model} seed={seed}: {[round(x['total_reward_yuan'], 1) for x in metrics]}")

    aggregate = aggregate_metrics(all_metrics)
    write_csv(OUT_DIR / "training_logs.csv", all_logs)
    write_csv(OUT_DIR / "evaluation_by_seed.csv", all_metrics)
    write_csv(OUT_DIR / "evaluation_aggregate.csv", aggregate)
    write_csv(OUT_DIR / "trajectories.csv", all_trajectories)
    payload = {
        "args": vars(args),
        "seeds": seeds,
        "models": models,
        "checkpoints": checkpoints,
        "training_rows_per_episode": args.train_periods_per_scenario,
        "training_scenarios": [item[0] for item in train_data],
    }
    (OUT_DIR / "experiment_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved long experiment outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
