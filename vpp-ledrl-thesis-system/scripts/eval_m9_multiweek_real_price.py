"""Evaluate LE-DRL-SAC vs SAC-Numeric on the multi-week real DE-LU price scenario.

N4 fix: the previous s5_real_price_week used a single real day tiled to seven
days (inter-day corr 0.997). This script evaluates on the genuine three-week
DE-LU day-ahead series (s5_real_price_multiweek, 21 independent daily patterns,
inter-day corr 0.875), so the bootstrap CI reflects real multi-day heterogeneity.

Three semantic sources (M5 ablation):
  - deepseek : the real DeepSeek scores (renewable=0.90, bias=0.80 on neg-price)
  - keyword  : LocalSemanticEncoder (MISSES negative-price -> bias=-0.75, wrong direction)
  - noisy    : DeepSeek + clipped Gaussian noise

Two models per source:
  - LE-DRL-SAC (native, relaxed regularizer w=0, actor_loss=0) -- matches the
    S5 new-event adaptation config in the main paper.
  - SAC-Numeric (text-agnostic reference)

Bootstrap CI: paired over 5 seeds. Also report a per-day breakdown so the
21 independent days are visible. Reward is in EUR (price column is EUR/MWh).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import VPPConfig
from app.core.environment import VPPEnv
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig
from app.core.rl.sac import SACAgent
from app.core.simulation import calculate_metrics

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "chapter6_long"
SEEDS = [2026, 2031, 2042, 2047, 2053]

SOURCE_FILES = {
    "deepseek": "s5_real_price_multiweek_ai_semantic.csv",
    "keyword": "s5_real_price_multiweek_ai_semantic_keyword.csv",
    "noisy": "s5_real_price_multiweek_ai_semantic_noisy.csv",
}


def load_data(source: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / SOURCE_FILES[source])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def eval_ckpt(ckpt: Path, model: str, data: pd.DataFrame) -> dict:
    """Evaluate one checkpoint on the full 21-day real-price series."""
    cfg = VPPConfig()
    sac = SACAgent.load(ckpt)
    if model == "LE-DRL-SAC":
        ag = LEDRLAgent(LEDRLConfig(include_semantic=True, semantic_mode="native",
                                    name="x", semantic_guidance_weight=0.0,
                                    semantic_guidance_power=2.0, use_ai_semantics=True))
    else:
        ag = LEDRLAgent(LEDRLConfig(include_semantic=False, name="x", use_ai_semantics=True))
    ag.sac = sac
    env = VPPEnv(data, config=cfg)
    state = env.reset(initial_soc=0.5)
    while not env.done():
        state, _, _, _ = env.step(ag.act(state, deterministic=True))
    history = pd.DataFrame(env.history)
    m = calculate_metrics(history)
    # negative-price charge rate: did the policy charge during real neg-price steps?
    history["price"] = pd.to_numeric(history["price_yuan_mwh"], errors="coerce")
    neg = history["price"] < 0
    neg_charge_rate = float(((history["actual_action_mw"] < -0.1) & neg).sum() / max(1, int(neg.sum())))
    # per-day reward breakdown (21 independent days)
    history["date"] = pd.to_datetime(history["timestamp"]).dt.date
    daily_reward = history.groupby("date")["reward_yuan"].sum().to_dict() if "reward_yuan" in history.columns else {}
    return {"total_reward_eur": m["total_reward_yuan"],
            "throughput_mwh": m["battery_throughput_mwh"],
            "cvar_5_eur": m["cvar_5_yuan"],
            "neg_price_charge_rate": neg_charge_rate,
            "daily_rewards": daily_reward}


def boot_ci(diffs: np.ndarray, n_boot: int = 20000, seed: int = 20260705) -> tuple[float, float, float]:
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=list(SOURCE_FILES.keys()), default="deepseek")
    p.add_argument("--ckpt-dir", type=str, required=True,
                   help="Directory with LE-DRL-SAC_seedN.pt and SAC-Numeric_seedN.pt checkpoints "
                        "(trained on the multiweek scenario, relaxed config).")
    p.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ckpt_dir = Path(args.ckpt_dir)
    data = load_data(args.source)

    rows = []
    for seed in seeds:
        for model in ["LE-DRL-SAC", "SAC-Numeric"]:
            ck = ckpt_dir / f"{model}_seed{seed}.pt"
            if not ck.exists():
                print(f"  SKIP {model} seed={seed}: {ck} not found")
                continue
            m = eval_ckpt(ck, model, data)
            rows.append({"source": args.source, "seed": seed, "model": model, **{k: v for k, v in m.items() if k != "daily_rewards"}})
            print(f"  {args.source} {model} seed={seed}: reward={m['total_reward_eur']:.1f} EUR  neg_chg={m['neg_price_charge_rate']:.2f}")
            # save daily breakdown for the first seed
            if seed == seeds[0]:
                daily = m["daily_rewards"]
                print(f"    daily rewards (EUR): " + " ".join(f"{d.strftime('%m-%d')}={v:.0f}" for d, v in sorted(daily.items())[:7]) + " ...")

    df = pd.DataFrame(rows)
    out_csv = OUT / f"m9_multiweek_eval_{args.source}_5seed.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_csv}")

    if len(df) >= 2 and "LE-DRL-SAC" in df["model"].values and "SAC-Numeric" in df["model"].values:
        print("\n" + "=" * 80)
        print(f"Gap: LE-DRL-SAC minus SAC-Numeric on multi-week real DE-LU ({args.source})")
        print("=" * 80)
        ledrl = df[df.model == "LE-DRL-SAC"].sort_values("seed")["total_reward_eur"].values
        numeric = df[df.model == "SAC-Numeric"].sort_values("seed")["total_reward_eur"].values
        n = min(len(ledrl), len(numeric))
        diffs = ledrl[:n] - numeric[:n]
        mean, lo, hi = boot_ci(diffs)
        fav = int((diffs > 0).sum())
        print(f"  gap={mean:+.1f} EUR  95% CI [{lo:+.1f}, {hi:+.1f}]  favorable={fav}/{n}")
        print(f"  LE-DRL-SAC mean neg-charge-rate: {df[df.model=='LE-DRL-SAC']['neg_price_charge_rate'].mean():.2f}")
        print(f"  SAC-Numeric mean neg-charge-rate: {df[df.model=='SAC-Numeric']['neg_price_charge_rate'].mean():.2f}")


if __name__ == "__main__":
    main()
