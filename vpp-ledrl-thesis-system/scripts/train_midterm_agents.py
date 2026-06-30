from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.data import load_vpp_dataset
from app.core.environment import VPPEnv
from app.core.policies import RuleBasedPolicy, SemanticEnhancedPolicy, RandomPolicy
from app.core.rl_agents import DiscreteSoftQAgent, SoftQPolicy
from app.core.simulation import run_policy


DATASET = ROOT / "data" / "processed" / "china_vpp_priority1_guangdong_sample.csv"
OUT_DIR = ROOT / "outputs" / "midterm"


def train_agent(data: pd.DataFrame, use_semantic: bool, episodes: int = 45) -> DiscreteSoftQAgent:
    agent = DiscreteSoftQAgent(use_semantic=use_semantic)
    for ep in range(episodes):
        env = VPPEnv(data)
        state = env.reset(initial_soc=0.45 + 0.1 * ((ep % 5) / 4))
        while not env.done():
            action = agent.select_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.epsilon = max(0.03, agent.epsilon * 0.96)
    return agent


def summarize_results(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        m = result["metrics"]
        rows.append(
            {
                "policy": result["policy"],
                "total_reward_yuan": round(m["total_reward_yuan"], 4),
                "mean_reward_yuan": round(m["mean_reward_yuan"], 4),
                "cvar_5_yuan": round(m["cvar_5_yuan"], 4),
                "battery_throughput_mwh": round(m["battery_throughput_mwh"], 4),
                "high_price_discharge_rate": round(m["high_price_discharge_rate"], 4),
                "low_price_charge_rate": round(m["low_price_charge_rate"], 4),
                "final_soc": round(m["final_soc"], 4),
                "event_count": m["event_count"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Run scripts/build_priority1_dataset.py first: {DATASET}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_vpp_dataset(DATASET)
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    numeric_agent = train_agent(train_df, use_semantic=False)
    semantic_agent = train_agent(train_df, use_semantic=True)
    numeric_agent.save(OUT_DIR / "softq_numeric_agent.json")
    semantic_agent.save(OUT_DIR / "softq_semantic_agent.json")

    policies = [
        RuleBasedPolicy(),
        SemanticEnhancedPolicy(),
        SoftQPolicy(numeric_agent, name="Soft-Q-Numeric"),
        SoftQPolicy(semantic_agent, name="Soft-Q-Semantic"),
        RandomPolicy(seed=2026),
    ]
    results = [run_policy(policy, test_df) for policy in policies]
    rows = summarize_results(results)
    write_csv(OUT_DIR / "midterm_model_comparison.csv", rows)

    compact = {
        "dataset": str(DATASET),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "results": rows,
        "note": "Soft-Q is a lightweight maximum-entropy RL baseline for midterm progress. Final thesis should upgrade it to continuous SAC/LE-DRL.",
    }
    (OUT_DIR / "midterm_model_comparison.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in rows:
        print(row)
    print(f"Saved: {OUT_DIR / 'midterm_model_comparison.csv'}")


if __name__ == "__main__":
    main()

