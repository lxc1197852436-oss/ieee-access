from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from app.core.config import VPPConfig
from app.core.semantic import LocalSemanticEncoder, SemanticSignal, semantic_vector


class VPPEnv:
    """Lightweight VPP environment for battery dispatch simulation."""

    def __init__(self, data: pd.DataFrame, config: VPPConfig | None = None):
        self.data = data.reset_index(drop=True).copy()
        self.config = config or VPPConfig()
        self.encoder = LocalSemanticEncoder()
        self.reset()

    def reset(self, initial_soc: float | None = None) -> dict:
        self.step_idx = 0
        self.soc = self.config.initial_soc if initial_soc is None else initial_soc
        self.history: list[dict] = []
        return self.state()

    def done(self) -> bool:
        return self.step_idx >= len(self.data)

    def state(self) -> dict:
        idx = min(self.step_idx, len(self.data) - 1)
        row = self.data.iloc[idx]
        signal = self._semantic_signal(row)
        hour = row["timestamp"].hour + row["timestamp"].minute / 60.0
        return {
            "idx": idx,
            "timestamp": row["timestamp"],
            "hour": hour,
            "load_mw": float(row["load_mw"]),
            "pv_mw": float(row["pv_mw"]),
            "price_yuan_mwh": float(row["price_yuan_mwh"]),
            "temperature_c": float(row["temperature_c"]),
            "event_type": str(row["event_type"]),
            "event_text": str(row["event_text"]),
            "soc": float(self.soc),
            "semantic": signal,
            "semantic_vector": semantic_vector(signal),
        }

    def _semantic_signal(self, row: pd.Series) -> SemanticSignal:
        ai_cols = [
            "ai_risk_score",
            "ai_price_spike_score",
            "ai_load_pressure_score",
            "ai_renewable_curtailment_score",
        ]
        if all(col in row.index for col in ai_cols) and not any(pd.isna(row[col]) for col in ai_cols):
            storage_bias = 0.0
            if "ai_recommended_storage_bias" in row.index and not pd.isna(row["ai_recommended_storage_bias"]):
                storage_bias = float(np.clip(row["ai_recommended_storage_bias"], -1.0, 1.0))
            return SemanticSignal(
                risk_score=float(np.clip(row["ai_risk_score"], 0.0, 1.0)),
                price_spike_score=float(np.clip(row["ai_price_spike_score"], 0.0, 1.0)),
                load_pressure_score=float(np.clip(row["ai_load_pressure_score"], 0.0, 1.0)),
                renewable_curtailment_score=float(np.clip(row["ai_renewable_curtailment_score"], 0.0, 1.0)),
                explanation_hint=str(row.get("ai_explanation", "AI语义评分")),
                storage_bias=storage_bias,
            )
        return self.encoder.encode(str(row["event_text"]))

    def step(self, action_mw: float) -> tuple[dict, float, bool, dict]:
        if self.done():
            raise RuntimeError("Environment is already done.")

        cfg = self.config
        row = self.data.iloc[self.step_idx]
        requested = float(np.clip(action_mw, -cfg.battery_power_mw, cfg.battery_power_mw))

        # Convention: action > 0 discharges battery; action < 0 charges battery.
        if requested >= 0:
            max_discharge_mwh = max(0.0, (self.soc - cfg.min_soc) * cfg.battery_capacity_mwh)
            actual_mw = min(requested, max_discharge_mwh * cfg.discharge_efficiency / cfg.dt_hours)
            delta_soc = -actual_mw * cfg.dt_hours / cfg.discharge_efficiency / cfg.battery_capacity_mwh
        else:
            max_charge_mwh = max(0.0, (cfg.max_soc - self.soc) * cfg.battery_capacity_mwh)
            actual_mw = -min(abs(requested), max_charge_mwh / cfg.charge_efficiency / cfg.dt_hours)
            delta_soc = -actual_mw * cfg.dt_hours * cfg.charge_efficiency / cfg.battery_capacity_mwh

        self.soc = float(np.clip(self.soc + delta_soc, cfg.min_soc, cfg.max_soc))
        pv = float(row["pv_mw"])
        load = float(row["load_mw"])
        price = float(row["price_yuan_mwh"])
        net_export_mw = pv - load + actual_mw

        revenue = net_export_mw * cfg.dt_hours * price
        degradation = abs(actual_mw) * cfg.dt_hours * cfg.degradation_cost_yuan_per_mwh
        curtailment_mwh = max(0.0, pv - load - max(0.0, -actual_mw)) * cfg.dt_hours
        curtailment_cost = curtailment_mwh * cfg.curtailment_penalty_yuan_per_mwh
        violation = cfg.violation_penalty_yuan if abs(actual_mw - requested) > 1e-6 else 0.0
        reward = revenue - degradation - curtailment_cost - violation

        record = {
            "timestamp": row["timestamp"].isoformat(),
            "load_mw": load,
            "pv_mw": pv,
            "price_yuan_mwh": price,
            "temperature_c": float(row["temperature_c"]),
            "event_type": str(row["event_type"]),
            "event_text": str(row["event_text"]),
            "requested_action_mw": requested,
            "actual_action_mw": float(actual_mw),
            "soc": self.soc,
            "net_export_mw": float(net_export_mw),
            "revenue_yuan": float(revenue),
            "degradation_yuan": float(degradation),
            "curtailment_cost_yuan": float(curtailment_cost),
            "violation_yuan": float(violation),
            "reward_yuan": float(reward),
        }
        self.history.append(record)
        self.step_idx += 1
        return self.state(), float(reward), self.done(), record

    def config_dict(self) -> dict:
        return asdict(self.config)
