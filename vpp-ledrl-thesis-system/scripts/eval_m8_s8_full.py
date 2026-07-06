"""Evaluate DeepSeek vs keyword native LE-DRL-SAC on FULL 672-step S8 (5 seeds, 2 configs).

S8 is the frequency-reserve scenario where the correct response (keep SOC mid-band
during reserve-request intervals) is OPPOSITE to what the physical signal suggests
(PV surplus -> charge -> fill up), and reserve-request intervals are scattered
across all hours so the hour encoding alone cannot predict them. The reserve
penalty (reserve_penalty_yuan_per_dev=600) makes "keep mid-band" reward-contingent
on the textual event.

This script reads checkpoints from m5_<source>_s8_al*_w*_rp600.0/ and evaluates on
full S8. Key metrics: total_reward, reserve_cost (the penalty paid), and the
mid-band-keeping rate during reserve intervals. Reports deepseek-minus-keyword
native gap with 95% CI.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import VPPConfig  # noqa: E402
from app.core.environment import VPPEnv  # noqa: E402
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig  # noqa: E402
from app.core.rl.sac import SACAgent  # noqa: E402
from app.core.simulation import calculate_metrics  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "chapter6_long"
SEEDS = [2026, 2031, 2042, 2047, 2053]
RESERVE_PENALTY = 600.0


def load_s8(source: str) -> pd.DataFrame:
    name = {"deepseek": "s8_freq_reserve_ai_semantic.csv",
            "keyword": "s8_freq_reserve_ai_semantic_keyword.csv"}[source]
    df = pd.read_csv(PROCESSED / name)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def eval_ckpt(ckpt: Path, model: str, data: pd.DataFrame, w: float) -> dict:
    cfg = VPPConfig(reserve_penalty_yuan_per_dev=RESERVE_PENALTY)
    sac = SACAgent.load(ckpt)
    if model == "LE-DRL-SAC":
        ag = LEDRLAgent(LEDRLConfig(include_semantic=True, semantic_mode="native",
                                    name="x", semantic_guidance_weight=w,
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
    # S8-specific: reserve cost (penalty paid) and mid-band-keeping during reserve.
    total_reserve_cost = float(history["reserve_cost_yuan"].sum())
    is_reserve = history["event_type"].str.contains("调频")
    # SOC deviation during reserve steps (lower = better mid-band keeping)
    soc_reserve = history.loc[is_reserve, "soc"]
    reserve_soc_dev = float(((soc_reserve - 0.5) ** 2).mean()) if len(soc_reserve) else float("nan")
    # Fraction of reserve steps where SOC stayed in [0.35, 0.65]
    midband = ((soc_reserve >= 0.35) & (soc_reserve <= 0.65)).mean() if len(soc_reserve) else float("nan")
    return {"total_reward_yuan": m["total_reward_yuan"],
            "throughput_mwh": m["battery_throughput_mwh"],
            "cvar_5_yuan": m["cvar_5_yuan"],
            "reserve_cost_yuan": total_reserve_cost,
            "reserve_soc_dev": reserve_soc_dev,
            "reserve_midband_rate": float(midband)}


def boot_ci(diffs: np.ndarray, n_boot: int = 20000, seed: int = 20260705) -> tuple[float, float, float]:
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def find_ckpt_dirs(source: str) -> list[Path]:
    dirs = sorted(OUT.glob(f"m5_{source}_s8_*"))
    return [d / "checkpoints" for d in dirs if (d / "checkpoints").exists()]


def parse_config(ckpt_dir: Path) -> tuple[float, float]:
    name = ckpt_dir.parent.name
    m = re.search(r"_al([\d.]+)_w([\d.]+)", name)
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


def main():
    rows = []
    for source in ["deepseek", "keyword"]:
        s8 = load_s8(source)
        for ckpt_dir in find_ckpt_dirs(source):
            al, w_train = parse_config(ckpt_dir)
            for seed in SEEDS:
                for model in ["LE-DRL-SAC", "SAC-Numeric"]:
                    ck = ckpt_dir / f"LE-DRL-SAC_seed{seed}.pt" if model == "LE-DRL-SAC" else ckpt_dir / f"SAC-Numeric_seed{seed}.pt"
                    if not ck.exists():
                        continue
                    m = eval_ckpt(ck, model, s8, w=w_train)
                    rows.append({"source": source, "seed": seed, "model": model,
                                 "actor_loss": al, "w": w_train, **m})
            print(f"{source} config(al={al},w={w_train}) done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "m8_s8_full_eval_5seed.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("S8 (full 672-step) 5-seed mean by source x model x config")
    print("=" * 90)
    for w in sorted(df["w"].unique()):
        sub = df[df.w == w]
        print(f"\n--- w={w} (safety {'ON' if w>0 else 'OFF'}) ---")
        agg = sub.groupby(["source", "model"])[["total_reward_yuan", "reserve_cost_yuan", "reserve_midband_rate"]].mean()
        print(agg.round(3).to_string())

    print("\n" + "=" * 90)
    print("Gap: deepseek-minus-keyword NATIVE on S8 (DeepSeek distinctness)")
    print("=" * 90)
    for w in sorted(df["w"].unique()):
        for metric in ["total_reward_yuan", "reserve_cost_yuan"]:
            ds = df[(df.source == "deepseek") & (df.model == "LE-DRL-SAC") & (df.w == w)].sort_values("seed")[metric].values
            kw = df[(df.source == "keyword") & (df.model == "LE-DRL-SAC") & (df.w == w)].sort_values("seed")[metric].values
            if len(ds) != len(kw) or len(ds) == 0:
                continue
            diffs = ds - kw
            mean, lo, hi = boot_ci(diffs)
            fav = int((diffs > 0).sum())
            cfg = "main (w=0.9, al=3.0)" if w > 0 else "relaxed (w=0, al=0)"
            sign = "higher" if metric == "total_reward_yuan" else "lower-better"
            print(f"  {cfg} [{metric}]: deepseek-keyword gap={mean:+.1f}  CI=[{lo:+.1f}, {hi:+.1f}]  favorable={fav}/{len(diffs)}  ({sign})")
    print(f"\n  (DeepSeek S8: renewable=0.20 bias=0; keyword S8: renewable=0 bias=0)")
    print(f"  >>> 若 total_reward gap>0 且 CI 全正，DeepSeek 在 S8 上显著优于 keyword")


if __name__ == "__main__":
    main()
