"""Evaluate deepseek vs keyword native LE-DRL-SAC on the FULL 672-step S5 (5 seeds).

The train_m5_ablation.py eval truncates S5 to 288 steps; this re-evaluates the
trained checkpoints on the complete 7-day S5 to get paper-grade numbers. Reports
the deepseek-minus-keyword native gap with 95% paired-bootstrap CI, plus each
source's native-minus-SAC-Numeric gap (the within-source adaptation gain).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.environment import VPPEnv  # noqa: E402
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig  # noqa: E402
from app.core.rl.sac import SACAgent  # noqa: E402
from app.core.simulation import calculate_metrics  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "chapter6_long"
SEEDS = [2026, 2031, 2042, 2047, 2053]


def load_s5(source: str) -> pd.DataFrame:
    name = {"deepseek": "s5_negative_price_surplus_ai_semantic.csv",
            "keyword": "s5_negative_price_surplus_ai_semantic_keyword.csv"}[source]
    df = pd.read_csv(PROCESSED / name)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def eval_ckpt(ckpt: Path, model: str, data: pd.DataFrame) -> dict:
    sac = SACAgent.load(ckpt)
    if model == "LE-DRL-SAC":
        ag = LEDRLAgent(LEDRLConfig(include_semantic=True, semantic_mode="native",
                                    name="x", semantic_guidance_weight=0.0, use_ai_semantics=True))
    else:  # SAC-Numeric
        ag = LEDRLAgent(LEDRLConfig(include_semantic=False, name="x", use_ai_semantics=True))
    ag.sac = sac
    env = VPPEnv(data)
    state = env.reset(initial_soc=0.5)
    while not env.done():
        state, _, _, _ = env.step(ag.act(state, deterministic=True))
    history = pd.DataFrame(env.history)
    m = calculate_metrics(history)
    neg = history["price_yuan_mwh"] < 0
    m["neg_price_charge_rate"] = float(((history["actual_action_mw"] < -0.1) & neg).sum() / max(1, int(neg.sum())))
    return {"total_reward_yuan": m["total_reward_yuan"],
            "throughput_mwh": m["battery_throughput_mwh"],
            "neg_price_charge_rate": m["neg_price_charge_rate"]}


def boot_ci(diffs: np.ndarray, n_boot: int = 20000, seed: int = 20260705) -> tuple[float, float, float]:
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def main():
    rows = []
    for source in ["deepseek", "keyword"]:
        s5 = load_s5(source)
        ckpt_dir = OUT / f"m5_{source}_s5" / "checkpoints"
        for seed in SEEDS:
            for model in ["LE-DRL-SAC", "SAC-Numeric"]:
                ck = ckpt_dir / f"{model.replace(' ', '_')}_seed{seed}.pt"
                # filename uses safe() which replaces spaces and '/' -> '_'; LE-DRL-SAC stays, w/o Text -> LE-DRL_w_o_Text
                if model == "LE-DRL-SAC":
                    ck = ckpt_dir / f"LE-DRL-SAC_seed{seed}.pt"
                if not ck.exists():
                    print(f"missing {ck}"); continue
                m = eval_ckpt(ck, model, s5)
                rows.append({"source": source, "seed": seed, "model": model, **m})
                print(f"  {source} {model} seed={seed}: reward={m['total_reward_yuan']:.1f} neg_chg={m['neg_price_charge_rate']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "m5_s5_full_eval_5seed.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("S5 (full 672-step) 5-seed mean reward")
    print("=" * 70)
    mean_tab = df.groupby(["source", "model"])["total_reward_yuan"].mean().unstack()
    print(mean_tab.round(1).to_string())

    print("\n" + "=" * 70)
    print("Gap 1: native-minus-SACNumeric per source (within-source adaptation gain)")
    print("=" * 70)
    for source in ["deepseek", "keyword"]:
        sub = df[df.source == source]
        nat = sub[sub.model == "LE-DRL-SAC"].sort_values("seed")["total_reward_yuan"].values
        num = sub[sub.model == "SAC-Numeric"].sort_values("seed")["total_reward_yuan"].values
        diffs = nat - num
        mean, lo, hi = boot_ci(diffs)
        fav = int((diffs > 0).sum())
        print(f"  {source:9s}: gap={mean:+.1f}  CI=[{lo:+.1f}, {hi:+.1f}]  favorable={fav}/{len(diffs)}")

    print("\n" + "=" * 70)
    print("Gap 2: deepseek-minus-keyword NATIVE on S5 (LLM-distinctness on new event)")
    print("=" * 70)
    ds = df[(df.source == "deepseek") & (df.model == "LE-DRL-SAC")].sort_values("seed")["total_reward_yuan"].values
    kw = df[(df.source == "keyword") & (df.model == "LE-DRL-SAC")].sort_values("seed")["total_reward_yuan"].values
    diffs = ds - kw
    mean, lo, hi = boot_ci(diffs)
    fav = int((diffs > 0).sum())
    print(f"  deepseek-keyword (native): gap={mean:+.1f}  CI=[{lo:+.1f}, {hi:+.1f}]  favorable={fav}/{len(diffs)}")
    print(f"  (DeepSeek S5 renewable=0.90 bias=+0.80; keyword S5 renewable=0.00 bias=-0.75)")
    print(f"  >>> 若 gap>0 且 CI 全正，说明 DeepSeek 在 S5 新事件上显著优于 keyword (LLM 独特性闭环)")


if __name__ == "__main__":
    main()
