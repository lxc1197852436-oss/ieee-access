from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import ScenarioConfig
from app.core.data import generate_china_vpp_scenario
from app.core.experiment_design import SCENARIOS
from app.core.mpc_optimizer import LinearMPCOptimizerPolicy
from app.core.policies import RuleBasedPolicy, SemanticEnhancedPolicy, RandomPolicy
from app.core.rl_agents import DiscreteSoftQAgent, SoftQPolicy
from app.core.rolling_optimizer import EnhancedRollingHorizonPolicy, RollingHorizonOptimizerPolicy
from app.core.simulation import run_policy

OUT_DIR = ROOT / "outputs" / "chapter6"
REPORTS = ROOT / "reports"


def apply_stress(df: pd.DataFrame, stress_type: str) -> pd.DataFrame:
    df = df.copy()
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    if stress_type == "heat_load":
        mask = (hour >= 11) & (hour <= 22)
        df.loc[mask, "temperature_c"] += 2.5
        df.loc[mask, "load_mw"] *= 1.12
        event_mask = (hour >= 14) & (hour <= 20) & (df.index % 10 == 0)
        df.loc[event_mask, "event_type"] = "高温预警"
        df.loc[event_mask, "event_text"] = "广东气象台发布高温橙色预警，预计晚高峰空调负荷显著上升。"
    elif stress_type == "price_spike":
        mask = (hour >= 18) & (hour <= 22)
        df.loc[mask, "price_yuan_mwh"] *= 1.35
        df["price_yuan_mwh"] = df["price_yuan_mwh"].clip(upper=650)
        event_mask = mask & (df.index % 8 == 0)
        df.loc[event_mask, "event_type"] = "价格尖峰"
        df.loc[event_mask, "event_text"] = "现货市场公告提示日前价格异常波动，晚高峰可能出现尖峰电价。"
    elif stress_type == "renewable_curtailment":
        mask = (hour >= 10) & (hour <= 14)
        df.loc[mask, "pv_mw"] *= 1.18
        df["pv_mw"] = df["pv_mw"].clip(upper=5.4)
        event_mask = mask & (df.index % 12 == 0)
        df.loc[event_mask, "event_type"] = "新能源消纳"
        df.loc[event_mask, "event_text"] = "调度公告提示午间新能源消纳压力增大，建议提升储能充电能力。"
    return df


def scenario_data(scenario) -> pd.DataFrame:
    cfg = ScenarioConfig(
        start=scenario.start,
        periods=scenario.periods,
        freq="15min",
        seed=scenario.seed,
        region=scenario.region,
    )
    return apply_stress(generate_china_vpp_scenario(cfg), scenario.stress_type)


def load_softq_policies() -> list:
    policies = []
    num = ROOT / "outputs" / "midterm" / "softq_numeric_agent.json"
    sem = ROOT / "outputs" / "midterm" / "softq_semantic_agent.json"
    if num.exists():
        policies.append(SoftQPolicy(DiscreteSoftQAgent.load(num), name="Soft-Q-Numeric"))
    if sem.exists():
        policies.append(SoftQPolicy(DiscreteSoftQAgent.load(sem), name="Soft-Q-Semantic"))
    return policies


def evaluate_scenario(scenario) -> list[dict]:
    data = scenario_data(scenario)
    policies = [
        RuleBasedPolicy(),
        SemanticEnhancedPolicy(),
        RollingHorizonOptimizerPolicy(data=data),
        EnhancedRollingHorizonPolicy(data=data),
        LinearMPCOptimizerPolicy(data=data),
        RandomPolicy(seed=scenario.seed),
    ] + load_softq_policies()
    rows = []
    for policy in policies:
        result = run_policy(policy, data)
        metrics = result["metrics"]
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "stress_type": scenario.stress_type,
                "policy": result["policy"],
                **{k: metrics[k] for k in metrics},
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for scenario in SCENARIOS:
        rows = evaluate_scenario(scenario)
        all_rows.extend(rows)
        print(scenario.scenario_id, [(r["policy"], round(r["total_reward_yuan"], 1)) for r in rows])
    write_csv(OUT_DIR / "chapter6_scenario_results.csv", all_rows)
    (OUT_DIR / "chapter6_scenario_results.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(REPORTS / "chapter6_scenario_results.csv", all_rows)
    print(f"Saved: {OUT_DIR / 'chapter6_scenario_results.csv'}")


if __name__ == "__main__":
    main()

