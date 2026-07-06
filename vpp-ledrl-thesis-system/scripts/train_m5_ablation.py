"""M5 ablation training: retrain LE-DRL-SAC on S1-S5 under three semantic sources.

This script answers the M5 reviewer question: is the DeepSeek gain unique to LLM
understanding, or is it matched by a simpler keyword encoder / by arbitrary
structured 5-dim perturbation (noisy)? It reuses the S1-S5 training loop of
scripts/train_ledrl_with_s5.py but swaps the semantic-feature CSVs across three
sources produced by scripts/build_semantic_ablation_data.py:

  --semantic-source deepseek  : the original DeepSeek-cache CSVs (main-paper path)
  --semantic-source keyword   : LocalSemanticEncoder-derived scores (Chinese keywords)
  --semantic-source noisy     : DeepSeek scores + clipped Gaussian noise (sigma=0.20/0.30)

The three sources share the SAME column schema, so VPPEnv._semantic_signal consumes
them unchanged (environment.py:50-69); only the score values differ. SAC-Numeric is
also trained as the text-agnostic reference (it is identical across sources because
it ignores semantic features, but we train it once per source to keep each run
self-contained and to surface any seed-path coupling).

Checkpoints and logs are written to outputs/chapter6_long/m5_<source>/ so the
DeepSeek main-paper results in checkpoints_with_s5/ stay reproducible and untouched.

Minimal-verification usage (1 seed, ~minutes on CPU):
  python scripts/train_m5_ablation.py --semantic-source keyword \
      --seeds 2026 --episodes 20 --models LE-DRL-SAC,SAC-Numeric
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
    print(exc)
    raise SystemExit(1) from exc

from app.core.environment import VPPEnv  # noqa: E402
from app.core.experiment_design import SCENARIOS  # noqa: E402
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig  # noqa: E402
from app.core.simulation import calculate_metrics  # noqa: E402
from scripts.train_chapter6_long_sac import (  # noqa: E402
    ProgressBar,
    learning_reward,
    semantic_auxiliary_reward,
    dispatch_alignment_reward,
    set_seed,
)

# --- Semantic-source -> file mapping --------------------------------------
# Each source points at the S1-S4 scenario file and the S5 file that carry the
# corresponding ai_* columns. keyword/noisy files are produced by
# scripts/build_semantic_ablation_data.py.
SOURCE_FILES = {
    "deepseek": {
        "scenarios": "chapter6_ai_semantic_scenarios.csv",
        "s5": "s5_negative_price_surplus_ai_semantic.csv",
        "s7": "s7_export_curtailed_ai_semantic.csv",
    },
    "keyword": {
        "scenarios": "chapter6_ai_semantic_scenarios_keyword.csv",
        "s5": "s5_negative_price_surplus_ai_semantic_keyword.csv",
        "s7": "s7_export_curtailed_ai_semantic_keyword.csv",
    },
    "noisy": {
        "scenarios": "chapter6_ai_semantic_scenarios_noisy.csv",
        "s5": "s5_negative_price_surplus_ai_semantic_noisy.csv",
        "s7": "s7_export_curtailed_ai_semantic_noisy.csv",
    },
}

PROCESSED = ROOT / "data" / "processed"
OUT_BASE = ROOT / "outputs" / "chapter6_long"


def load_csv(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(
            f"Semantic-source CSV not found: {path}. "
            "For keyword/noisy, run scripts/build_semantic_ablation_data.py first."
        )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def training_sets_with_s5(source: str, periods_per_scenario: int, include_s5: bool = True, include_s7: bool = False) -> list[tuple[str, pd.DataFrame]]:
    files = SOURCE_FILES[source]
    ai = load_csv(files["scenarios"])
    rows = []
    for sc in SCENARIOS:
        sub = ai[ai["scenario_id"] == sc.scenario_id].head(periods_per_scenario).reset_index(drop=True)
        rows.append((sc.scenario_id, sub))
    if include_s5:
        s5 = load_csv(files["s5"]).head(periods_per_scenario).reset_index(drop=True)
        rows.append(("S5", s5))
    if include_s7:
        s7 = load_csv(files["s7"]).head(periods_per_scenario).reset_index(drop=True)
        rows.append(("S7", s7))
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


def eval_on_scenarios(agent, train_data) -> list[dict]:
    """Evaluate a trained agent on every training scenario (deterministic).

    Returns one row per (scenario_id) with total_reward, CVaR, throughput, and
    -- for negative-price scenarios -- the neg-price charge rate. The neg-price
    rate is 0 by definition on S1-S4 (no negative prices there).
    """
    rows = []
    for scenario_id, data in train_data:
        env = VPPEnv(data)
        state = env.reset(initial_soc=0.5)
        while not env.done():
            action = agent.act(state, deterministic=True)
            state, _, _, _ = env.step(action)
        history = pd.DataFrame(env.history)
        metrics = calculate_metrics(history)
        neg = history["price_yuan_mwh"] < 0
        metrics["neg_price_charge_rate"] = float(
            ((history["actual_action_mw"] < -0.1) & neg).sum() / max(1, int(neg.sum()))
        )
        rows.append({
            "scenario_id": scenario_id,
            "total_reward_yuan": metrics["total_reward_yuan"],
            "cvar_5_yuan": metrics["cvar_5_yuan"],
            "throughput_mwh": metrics["battery_throughput_mwh"],
            "neg_price_charge_rate": metrics["neg_price_charge_rate"],
        })
    return rows


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def main() -> None:
    p = argparse.ArgumentParser(description="M5 ablation: train LE-DRL-SAC on S1-S5 under deepseek/keyword/noisy semantic sources.")
    p.add_argument("--semantic-source", choices=list(SOURCE_FILES.keys()), default="keyword")
    p.add_argument("--include-s5", action="store_true", default=True,
                   help="Include S5 in training (new-event adaptation mode). Pass --no-include-s5 for the known-event main config.")
    p.add_argument("--no-include-s5", dest="include_s5", action="store_false",
                   help="Train on S1-S4 only (known-event main config: w=0.9, actor_loss=3.0).")
    p.add_argument("--include-s7", action="store_true", default=False,
                   help="Include S7 (export-curtailed local absorption) in training. "
                        "S7 is the keyword-blind event that tests DeepSeek distinctness.")
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
    # Relaxed regularizer (w_actor=0) is the new-event adaptation config; the M5
    # ablation uses the same relaxed config as Table s5_unseen so the comparison
    # is apples-to-apples with the main-paper S5 numbers.
    p.add_argument("--semantic-actor-loss-weight", type=float, default=0.0)
    p.add_argument("--models", default="LE-DRL-SAC,SAC-Numeric")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    # Output dir reflects training-set composition AND config (actor_loss, w) to
    # avoid checkpoint collisions when the same source is trained under different
    # configs (e.g. main actor_loss=3.0/w=0.9 vs relaxed actor_loss=0.0/w=0.0).
    tag = "_s1to4"
    if args.include_s5 and args.include_s7:
        tag = "_s5s7"
    elif args.include_s5:
        tag = "_s5"
    elif args.include_s7:
        tag = "_s7"
    # Config tag: short suffix encoding actor_loss and guidance_weight so main vs
    # relaxed runs of the same source land in different dirs.
    cfg_tag = f"_al{args.semantic_actor_loss_weight}_w{args.semantic_guidance_weight}"
    out_dir = OUT_BASE / f"m5_{args.semantic_source}{tag}{cfg_tag}"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    train_data = training_sets_with_s5(
        args.semantic_source, args.train_periods_per_scenario,
        include_s5=args.include_s5, include_s7=args.include_s7,
    )
    s5_data = next((d for sid, d in train_data if sid == "S5"), None)
    print(f"semantic_source={args.semantic_source}  scenarios={[sid for sid,_ in train_data]}  seeds={seeds}")

    all_logs = []
    eval_rows = []
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
            elif model == "LE-DRL w/o Text":
                # Zeroed semantic vector: same input dim as LE-DRL-SAC but the 5
                # semantic dims are always 0. Isolates whether the actor actually
                # uses the semantic channel when it is present-but-zero.
                agent = LEDRLAgent(LEDRLConfig(
                    include_semantic=True, semantic_mode="zero", name="LE-DRL w/o Text",
                    **common,
                ))
            else:
                raise ValueError(model)
            logs = train_one(agent, train_data, seed, args)
            all_logs.extend(logs)
            ckpt = ckpt_dir / f"{safe(model)}_seed{seed}.pt"
            agent.sac.save(ckpt)
            checkpoints.append({"seed": seed, "model": model, "checkpoint": str(ckpt)})
            scenario_metrics = eval_on_scenarios(agent, train_data)
            for sm in scenario_metrics:
                eval_rows.append({
                    "semantic_source": args.semantic_source, "seed": seed, "model": model,
                    "scenario_id": sm["scenario_id"],
                    "total_reward_yuan": sm["total_reward_yuan"],
                    "cvar_5_yuan": sm["cvar_5_yuan"],
                    "throughput_mwh": sm["throughput_mwh"],
                    "neg_price_charge_rate": sm["neg_price_charge_rate"],
                })
            s5_metric = next((sm for sm in scenario_metrics if sm["scenario_id"] == "S5"), None)
            if s5_metric:
                print(f"  EVAL {args.semantic_source} {model} seed={seed}: "
                      f"S5 reward={s5_metric['total_reward_yuan']:.1f}  neg_chg={s5_metric['neg_price_charge_rate']:.2f}  "
                      f"all={[ (sm['scenario_id'], round(sm['total_reward_yuan'],0)) for sm in scenario_metrics]}")
            else:
                print(f"  EVAL {args.semantic_source} {model} seed={seed}: "
                      f"all={[ (sm['scenario_id'], round(sm['total_reward_yuan'],0)) for sm in scenario_metrics]}")

    # Persist logs + eval rows + config.
    import csv as _csv
    with (out_dir / "training_logs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=list(all_logs[0].keys()))
        w.writeheader(); w.writerows(all_logs)
    with (out_dir / "s5_eval.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
        w.writeheader(); w.writerows(eval_rows)

    payload = {
        "args": vars(args), "seeds": seeds,
        "semantic_source": args.semantic_source,
        "models": args.models.split(","),
        "training_scenarios": [sid for sid, _ in train_data],
        "checkpoints": checkpoints,
        "source_files": SOURCE_FILES[args.semantic_source],
    }
    (out_dir / "m5_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print a tiny summary so the minimal-verification run is readable at a glance.
    print(f"\n=== M5 minimal-verification summary  source={args.semantic_source} ===")
    df = pd.DataFrame(eval_rows)
    print(df.to_string(index=False))
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
