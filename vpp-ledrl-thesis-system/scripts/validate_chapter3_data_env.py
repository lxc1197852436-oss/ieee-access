from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.data import load_vpp_dataset
from app.core.environment import VPPEnv
from app.core.policies import RuleBasedPolicy


DATASET = ROOT / "data" / "processed" / "china_vpp_priority1_guangdong_sample.csv"
REPORT = ROOT / "reports" / "chapter3_data_env_validation.json"


def validate_dataset(df: pd.DataFrame) -> dict:
    required = [
        "timestamp",
        "region",
        "load_mw",
        "pv_mw",
        "price_yuan_mwh",
        "temperature_c",
        "event_type",
        "event_text",
    ]
    missing = [c for c in required if c not in df.columns]
    duplicated_ts = int(df["timestamp"].duplicated().sum())
    diffs = df["timestamp"].diff().dropna()
    expected_step = pd.Timedelta(minutes=15)
    irregular_steps = int((diffs != expected_step).sum())
    numeric_cols = ["load_mw", "pv_mw", "price_yuan_mwh", "temperature_c"]
    null_counts = {c: int(df[c].isna().sum()) for c in required if c in df.columns}
    ranges = {
        c: {"min": float(df[c].min()), "max": float(df[c].max()), "mean": float(df[c].mean())}
        for c in numeric_cols
    }
    event_counts = df["event_type"].value_counts().to_dict()
    return {
        "rows": int(len(df)),
        "missing_required_columns": missing,
        "duplicated_timestamps": duplicated_ts,
        "irregular_15min_steps": irregular_steps,
        "null_counts": null_counts,
        "numeric_ranges": ranges,
        "event_counts": event_counts,
        "time_range": [df["timestamp"].min().isoformat(), df["timestamp"].max().isoformat()],
    }


def validate_environment(df: pd.DataFrame) -> dict:
    env = VPPEnv(df)
    policy = RuleBasedPolicy()
    state = env.reset(initial_soc=0.5)
    rewards = []
    soc_values = []
    clipped_actions = 0
    for _ in range(min(96, len(df))):
        action = policy.act(state)
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        soc_values.append(info["soc"])
        if abs(info["requested_action_mw"] - info["actual_action_mw"]) > 1e-6:
            clipped_actions += 1
        state = next_state
        if done:
            break
    return {
        "checked_steps": len(rewards),
        "reward_sum_yuan": float(sum(rewards)),
        "reward_mean_yuan": float(sum(rewards) / max(1, len(rewards))),
        "soc_min": float(min(soc_values)),
        "soc_max": float(max(soc_values)),
        "action_clipped_count": clipped_actions,
        "history_fields": sorted(env.history[0].keys()) if env.history else [],
    }


def main() -> None:
    df = load_vpp_dataset(DATASET)
    report = {
        "dataset": str(DATASET),
        "dataset_validation": validate_dataset(df),
        "environment_validation": validate_environment(df),
        "conclusion": "pass",
    }
    if report["dataset_validation"]["missing_required_columns"] or report["dataset_validation"]["irregular_15min_steps"]:
        report["conclusion"] = "check_required"
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

