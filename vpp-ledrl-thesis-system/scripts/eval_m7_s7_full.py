"""Evaluate DeepSeek vs keyword native LE-DRL-SAC on the FULL 672-step S7 (5 seeds, 2 configs).

This is the S7 distinctness test. S7 is the export-curtailed local-absorption event
whose text ("就地消纳") the keyword encoder cannot match (returns all-zero scores)
while DeepSeek recognizes the synonym (renewable=0.80, storage_bias=+0.60). The
question: does DeepSeek's correct semantic signal translate into higher reward than
keyword's all-zero signal, under both the main config (actor_loss=3.0, w=0.9) and
the relaxed config (actor_loss=0, w=0)?

Reads checkpoints from outputs/chapter6_long/m5_<source>_s7/ (main config) and
m5_<source>_s7/ (relaxed -- same dir name because both use --include-s7 --no-include-s5;
the config difference is in the actor_loss/guidance_weight args, NOT the dir). To
distinguish, the training run writes checkpoints to m5_<source>_s7/ for BOTH configs
and they overwrite each other -- so this script assumes the LAST training run for
each source is the one to evaluate. If you ran both configs, run this script after
each config's training completes (checkpoints reflect that config).

Actually: the training script names dirs by training-set composition only
(m5_<source>_s7), so main-config and relaxed-config runs of the same source
overwrite each other. To keep both, the user should run configs sequentially per
source and copy the checkpoint dir between runs, OR this script should be run
right after each config. The training commands above run all 4 in parallel, which
means same-source configs collide. FIX: the training dir naming must include the
config. This script handles whatever checkpoints are present.
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


def load_s7(source: str) -> pd.DataFrame:
    name = {"deepseek": "s7_export_curtailed_ai_semantic.csv",
            "keyword": "s7_export_curtailed_ai_semantic_keyword.csv"}[source]
    df = pd.read_csv(PROCESSED / name)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def eval_ckpt(ckpt: Path, model: str, data: pd.DataFrame, w: float) -> dict:
    """Evaluate a checkpoint on full S7. w=semantic_guidance_weight (0.9 main, 0.0 relaxed)."""
    sac = SACAgent.load(ckpt)
    if model == "LE-DRL-SAC":
        ag = LEDRLAgent(LEDRLConfig(include_semantic=True, semantic_mode="native",
                                    name="x", semantic_guidance_weight=w,
                                    semantic_guidance_power=2.0, use_ai_semantics=True))
    else:
        ag = LEDRLAgent(LEDRLConfig(include_semantic=False, name="x", use_ai_semantics=True))
    ag.sac = sac
    env = VPPEnv(data)
    state = env.reset(initial_soc=0.5)
    while not env.done():
        state, _, _, _ = env.step(ag.act(state, deterministic=True))
    history = pd.DataFrame(env.history)
    m = calculate_metrics(history)
    # S7-specific: midday surplus charge rate (how much of the midday PV-surplus
    # window did the policy charge through). Higher = better local absorption.
    ts = pd.to_datetime(history["timestamp"])
    h = ts.dt.hour + ts.dt.minute / 60
    noon = h.between(10, 15)
    surplus = (history["pv_mw"] - history["load_mw"]) > 0.2
    noon_surplus = noon & surplus
    m["noon_surplus_charge_rate"] = float(
        ((history["actual_action_mw"] < -0.1) & noon_surplus).sum() / max(1, int(noon_surplus.sum()))
    )
    return {"total_reward_yuan": m["total_reward_yuan"],
            "throughput_mwh": m["battery_throughput_mwh"],
            "cvar_5_yuan": m["cvar_5_yuan"],
            "noon_surplus_charge_rate": m["noon_surplus_charge_rate"]}


def boot_ci(diffs: np.ndarray, n_boot: int = 20000, seed: int = 20260705) -> tuple[float, float, float]:
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def find_ckpt_dirs(source: str) -> list[Path]:
    """Find all m5_<source>_s7_* checkpoint dirs (one per config)."""
    pattern = f"m5_{source}_s7_*"
    dirs = sorted((OUT).glob(pattern))
    dirs = [d / "checkpoints" for d in dirs if (d / "checkpoints").exists()]
    return dirs


def parse_config(ckpt_dir: Path) -> tuple[float, float]:
    """Extract actor_loss (al) and guidance_weight (w) from dir name like
    m5_deepseek_s7_al3.0_w0.9/checkpoints."""
    name = ckpt_dir.parent.name  # m5_deepseek_s7_al3.0_w0.9
    import re
    m = re.search(r"_al([\d.]+)_w([\d.]+)", name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


def main():
    rows = []
    for source in ["deepseek", "keyword"]:
        s7 = load_s7(source)
        ckpt_dirs = find_ckpt_dirs(source)
        if not ckpt_dirs:
            print(f"skip {source}: no checkpoint dirs matching m5_{source}_s7_*")
            continue
        for ckpt_dir in ckpt_dirs:
            al, w_train = parse_config(ckpt_dir)
            # The training w is the test-time safety weight used during training-time eval.
            # For evaluation we use w_train (the config the actor was trained under).
            for seed in SEEDS:
                for model in ["LE-DRL-SAC", "SAC-Numeric"]:
                    ck = ckpt_dir / f"{model.replace(' ', '_')}_seed{seed}.pt"
                    if model == "LE-DRL-SAC":
                        ck = ckpt_dir / f"LE-DRL-SAC_seed{seed}.pt"
                    if not ck.exists():
                        continue
                    m = eval_ckpt(ck, model, s7, w=w_train)
                    rows.append({"source": source, "seed": seed, "model": model,
                                 "actor_loss": al, "w": w_train, "ckpt_dir": str(ckpt_dir), **m})
            print(f"{source} config(al={al},w={w_train}) done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "m7_s7_full_eval_5seed.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("S7 (full 672-step) 5-seed mean reward by source x model x config")
    print("=" * 80)
    mean_tab = df.groupby(["source", "model", "w"])["total_reward_yuan"].mean().reset_index()
    piv = mean_tab.pivot_table(index=["model", "w"], columns="source", values="total_reward_yuan")
    print(piv.round(1).to_string())

    print("\n" + "=" * 80)
    print("Gap 1: native-minus-SACNumeric per source x config (within-source gain)")
    print("=" * 80)
    for source in ["deepseek", "keyword"]:
        for w in sorted(df["w"].unique()):
            sub = df[(df.source == source) & (df.w == w)]
            nat = sub[sub.model == "LE-DRL-SAC"].sort_values("seed")["total_reward_yuan"].values
            num = sub[sub.model == "SAC-Numeric"].sort_values("seed")["total_reward_yuan"].values
            if len(nat) != len(num) or len(nat) == 0:
                continue
            diffs = nat - num
            mean, lo, hi = boot_ci(diffs)
            fav = int((diffs > 0).sum())
            print(f"  {source:9s} w={w}: gap={mean:+.1f}  CI=[{lo:+.1f}, {hi:+.1f}]  favorable={fav}/{len(diffs)}")

    print("\n" + "=" * 80)
    print("Gap 2: deepseek-minus-keyword NATIVE on S7 (DeepSeek distinctness)")
    print("=" * 80)
    for w in sorted(df["w"].unique()):
        ds = df[(df.source == "deepseek") & (df.model == "LE-DRL-SAC") & (df.w == w)].sort_values("seed")["total_reward_yuan"].values
        kw = df[(df.source == "keyword") & (df.model == "LE-DRL-SAC") & (df.w == w)].sort_values("seed")["total_reward_yuan"].values
        if len(ds) != len(kw) or len(ds) == 0:
            continue
        diffs = ds - kw
        mean, lo, hi = boot_ci(diffs)
        fav = int((diffs > 0).sum())
        cfg = "main (w=0.9, actor_loss=3.0)" if w > 0 else "relaxed (w=0, actor_loss=0)"
        print(f"  {cfg}: deepseek-keyword gap={mean:+.1f}  CI=[{lo:+.1f}, {hi:+.1f}]  favorable={fav}/{len(diffs)}")
    print(f"\n  (DeepSeek S7: renewable=0.80 bias=+0.60; keyword S7: renewable=0 bias=0)")
    print(f"  >>> 若 gap>0 且 CI 全正，DeepSeek 在 S7 上显著优于 keyword (LLM 独特性成立)")


if __name__ == "__main__":
    main()
