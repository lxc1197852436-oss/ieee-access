"""Evaluate already-trained M5 checkpoints on S1-S5 to see where semantic sources diverge.

Reads checkpoints from outputs/chapter6_long/m5_<source>/checkpoints/ for each source
(deepseek/keyword/noisy) and evaluates LE-DRL-SAC + SAC-Numeric on every scenario.
The point is to find whether DeepSeek diverges from keyword/noisy on the event-driven
scenarios (S2 high-temp, S4 curtailment) where textual semantics should matter more
than on S5 where the negative-price signal alone is strong enough to drive charging.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.environment import VPPEnv  # noqa: E402
from app.core.experiment_design import SCENARIOS  # noqa: E402
from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig  # noqa: E402
from app.core.rl.sac import SACAgent  # noqa: E402
from app.core.simulation import calculate_metrics  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT_BASE = ROOT / "outputs" / "chapter6_long"

SOURCE_FILES = {
    "deepseek": {
        "scenarios": "chapter6_ai_semantic_scenarios.csv",
        "s5": "s5_negative_price_surplus_ai_semantic.csv",
    },
    "keyword": {
        "scenarios": "chapter6_ai_semantic_scenarios_keyword.csv",
        "s5": "s5_negative_price_surplus_ai_semantic_keyword.csv",
    },
    "noisy": {
        "scenarios": "chapter6_ai_semantic_scenarios_noisy.csv",
        "s5": "s5_negative_price_surplus_ai_semantic_noisy.csv",
    },
}


def load_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / name)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def scenario_frames(source: str) -> list[tuple[str, pd.DataFrame]]:
    files = SOURCE_FILES[source]
    ai = load_csv(files["scenarios"])
    s5 = load_csv(files["s5"])
    frames = []
    for sc in SCENARIOS:
        frames.append((sc.scenario_id, ai[ai["scenario_id"] == sc.scenario_id].reset_index(drop=True)))
    frames.append(("S5", s5))
    return frames


def eval_ckpt(ckpt_path: Path, model: str, data: pd.DataFrame) -> dict:
    sac = SACAgent.load(ckpt_path)
    if model == "LE-DRL-SAC":
        ag = LEDRLAgent(LEDRLConfig(include_semantic=True, semantic_mode="native",
                                    name="x", semantic_guidance_weight=0.0, use_ai_semantics=True))
    elif model == "SAC-Numeric":
        ag = LEDRLAgent(LEDRLConfig(include_semantic=False, name="x", use_ai_semantics=True))
    else:
        raise ValueError(model)
    ag.sac = sac
    env = VPPEnv(data)
    state = env.reset(initial_soc=0.5)
    while not env.done():
        state, _, _, _ = env.step(ag.act(state, deterministic=True))
    history = pd.DataFrame(env.history)
    m = calculate_metrics(history)
    neg = history["price_yuan_mwh"] < 0
    m["neg_price_charge_rate"] = float(((history["actual_action_mw"] < -0.1) & neg).sum() / max(1, int(neg.sum())))
    return m


def main():
    rows = []
    for source in ["deepseek", "keyword", "noisy"]:
        ckpt_dir = OUT_BASE / f"m5_{source}" / "checkpoints"
        if not ckpt_dir.exists():
            print(f"skip {source}: no checkpoints at {ckpt_dir}")
            continue
        frames = scenario_frames(source)
        for model in ["LE-DRL-SAC", "SAC-Numeric"]:
            ck = ckpt_dir / f"{model}_seed2026.pt"
            if not ck.exists():
                print(f"  missing {ck}")
                continue
            for sid, data in frames:
                m = eval_ckpt(ck, model, data)
                rows.append({
                    "source": source, "model": model, "scenario_id": sid,
                    "total_reward_yuan": m["total_reward_yuan"],
                    "throughput_mwh": m["battery_throughput_mwh"],
                    "neg_price_charge_rate": m["neg_price_charge_rate"],
                })
        print(f"{source} done")

    df = pd.DataFrame(rows)
    out = OUT_BASE / "m5_cross_source_eval.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out}")

    # Pivot: per scenario, LE-DRL-SAC reward by source.
    print("\n=== LE-DRL-SAC total_reward by source x scenario (seed 2026) ===")
    led = df[df.model == "LE-DRL-SAC"]
    piv = led.pivot(index="scenario_id", columns="source", values="total_reward_yuan")
    piv = piv[["deepseek", "keyword", "noisy"]]
    print(piv.round(1).to_string())

    print("\n=== SAC-Numeric total_reward by source x scenario (should be identical across sources) ===")
    num = df[df.model == "SAC-Numeric"]
    pivn = num.pivot(index="scenario_id", columns="source", values="total_reward_yuan")
    pivn = pivn[["deepseek", "keyword", "noisy"]]
    print(pivn.round(1).to_string())

    print("\n=== gap (LE-DRL - SAC-Numeric) by source x scenario ===")
    gaps = []
    for sid in piv.index:
        for src in ["deepseek", "keyword", "noisy"]:
            led_v = piv.loc[sid, src]
            num_v = pivn.loc[sid, src]
            gaps.append({"scenario_id": sid, "source": src, "gap": led_v - num_v})
    gdf = pd.DataFrame(gaps).pivot(index="scenario_id", columns="source", values="gap")
    gdf = gdf[["deepseek", "keyword", "noisy"]]
    print(gdf.round(1).to_string())


if __name__ == "__main__":
    main()
