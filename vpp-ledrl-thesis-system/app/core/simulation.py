from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from app.core.config import ScenarioConfig, VPPConfig
from app.core.data import generate_china_vpp_scenario
from app.core.environment import VPPEnv
from app.core.llm_provider import LLMProvider
from app.core.policies import POLICIES, Policy


def run_policy(policy: Policy, data: pd.DataFrame, config: VPPConfig | None = None, explain_every: int = 32) -> dict:
    env = VPPEnv(data, config)
    llm = LLMProvider()
    state = env.reset()
    explanations = []

    while not env.done():
        action = policy.act(state)
        current_state = state
        state, reward, done, info = env.step(action)
        if len(env.history) == 1 or info["event_type"] != "正常运行" or len(env.history) % explain_every == 0:
            pred = llm.predict(current_state, info["actual_action_mw"])
            explanations.append({**pred.__dict__, "timestamp": info["timestamp"], "event_type": info["event_type"]})

    history = pd.DataFrame(env.history)
    metrics = calculate_metrics(history)
    return {
        "policy": policy.name,
        "metrics": metrics,
        "history": history.to_dict(orient="records"),
        "explanations": explanations[:80],
    }


def calculate_metrics(history: pd.DataFrame) -> dict:
    rewards = history["reward_yuan"].to_numpy()
    cvar_cut = max(1, int(len(rewards) * 0.05))
    cvar_5 = float(np.sort(rewards)[:cvar_cut].mean())
    high_price = history["price_yuan_mwh"] >= history["price_yuan_mwh"].quantile(0.75)
    low_price = history["price_yuan_mwh"] <= history["price_yuan_mwh"].quantile(0.25)
    discharge = history["actual_action_mw"] > 0.1
    charge = history["actual_action_mw"] < -0.1
    return {
        "total_reward_yuan": float(rewards.sum()),
        "mean_reward_yuan": float(rewards.mean()),
        "cvar_5_yuan": cvar_5,
        "final_soc": float(history["soc"].iloc[-1]),
        "battery_throughput_mwh": float(history["actual_action_mw"].abs().sum() * 0.25),
        "high_price_discharge_rate": float((discharge & high_price).sum() / max(1, high_price.sum())),
        "low_price_charge_rate": float((charge & low_price).sum() / max(1, low_price.sum())),
        "event_count": int((history["event_type"] != "正常运行").sum()),
    }


def run_experiment(policy_names: list[str] | None = None, scenario: ScenarioConfig | None = None) -> dict:
    scenario_cfg = scenario or ScenarioConfig()
    data = generate_china_vpp_scenario(scenario_cfg)
    names = policy_names or ["rule", "ledrl", "random"]
    results = []
    for name in names:
        if name not in POLICIES:
            raise ValueError(f"Unknown policy: {name}")
        results.append(run_policy(POLICIES[name](), data))
    return {
        "scenario": asdict(scenario_cfg),
        "data_preview": _records_for_json(data.head(24)),
        "results": results,
    }


def _records_for_json(df: pd.DataFrame) -> list[dict]:
    records = df.to_dict(orient="records")
    for item in records:
        if "timestamp" in item:
            item["timestamp"] = item["timestamp"].isoformat()
    return records
