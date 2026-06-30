from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from app.core.config import VPPConfig
from app.core.policies import Policy


def transition_reward(soc: float, action: float, row: pd.Series, cfg: VPPConfig) -> tuple[float, float]:
    requested = float(np.clip(action, -cfg.battery_power_mw, cfg.battery_power_mw))
    if requested >= 0:
        max_discharge_mwh = max(0.0, (soc - cfg.min_soc) * cfg.battery_capacity_mwh)
        actual = min(requested, max_discharge_mwh * cfg.discharge_efficiency / cfg.dt_hours)
        delta_soc = -actual * cfg.dt_hours / cfg.discharge_efficiency / cfg.battery_capacity_mwh
    else:
        max_charge_mwh = max(0.0, (cfg.max_soc - soc) * cfg.battery_capacity_mwh)
        actual = -min(abs(requested), max_charge_mwh / cfg.charge_efficiency / cfg.dt_hours)
        delta_soc = -actual * cfg.dt_hours * cfg.charge_efficiency / cfg.battery_capacity_mwh

    next_soc = float(np.clip(soc + delta_soc, cfg.min_soc, cfg.max_soc))
    pv = float(row["pv_mw"])
    load = float(row["load_mw"])
    price = float(row["price_yuan_mwh"])
    net_export = pv - load + actual
    revenue = net_export * cfg.dt_hours * price
    degradation = abs(actual) * cfg.dt_hours * cfg.degradation_cost_yuan_per_mwh
    curtailment_mwh = max(0.0, pv - load - max(0.0, -actual)) * cfg.dt_hours
    curtailment_cost = curtailment_mwh * cfg.curtailment_penalty_yuan_per_mwh
    violation = cfg.violation_penalty_yuan if abs(actual - requested) > 1e-6 else 0.0
    return next_soc, float(revenue - degradation - curtailment_cost - violation)


@dataclass
class RollingHorizonOptimizerPolicy(Policy):
    """Discrete rolling-horizon optimization baseline.

    This is a solver-free strong baseline. It approximates a rolling MILP by
    discretizing battery actions and SOC states over a short forecast horizon.
    """

    data: pd.DataFrame
    config: VPPConfig = VPPConfig()
    horizon: int = 16
    soc_bins: int = 33
    name: str = "Rolling-Horizon"

    def __post_init__(self):
        self.actions = np.asarray([-2.0, -1.2, -0.6, 0.0, 0.6, 1.2, 2.0], dtype=float)
        self.grid = np.linspace(self.config.min_soc, self.config.max_soc, self.soc_bins)

    def _soc_to_idx(self, soc: float) -> int:
        return int(np.argmin(np.abs(self.grid - soc)))

    def act(self, state: dict) -> float:
        start = int(state["idx"])
        end = min(len(self.data), start + self.horizon)
        horizon_df = self.data.iloc[start:end].reset_index(drop=True)
        if len(horizon_df) == 0:
            return 0.0

        # value[t, i] = best value from local step t with SOC grid i.
        value = np.zeros((len(horizon_df) + 1, self.soc_bins), dtype=float)
        policy = np.zeros((len(horizon_df), self.soc_bins), dtype=float)
        for t in range(len(horizon_df) - 1, -1, -1):
            row = horizon_df.iloc[t]
            for i, soc in enumerate(self.grid):
                best_v = -1e18
                best_a = 0.0
                for action in self.actions:
                    next_soc, reward = transition_reward(float(soc), float(action), row, self.config)
                    j = self._soc_to_idx(next_soc)
                    candidate = reward + value[t + 1, j]
                    if candidate > best_v:
                        best_v = candidate
                        best_a = float(action)
                value[t, i] = best_v
                policy[t, i] = best_a
        return float(policy[0, self._soc_to_idx(float(state["soc"]))])


@dataclass
class EnhancedRollingHorizonPolicy(Policy):
    """MPC-like rolling optimization baseline with engineering penalties.

    Compared with ``RollingHorizonOptimizerPolicy``, this baseline adds a
    terminal SOC value, action-smoothing penalty, and extra cycling penalty. It
    is still solver-free and deterministic, but it is closer to an operational
    MPC baseline than a pure one-step reward look-ahead.
    """

    data: pd.DataFrame
    config: VPPConfig = VPPConfig()
    horizon: int = 8
    soc_bins: int = 17
    terminal_soc_target: float = 0.50
    terminal_soc_penalty_yuan: float = 5000.0
    smoothing_penalty_yuan_per_mw: float = 10.0
    extra_cycling_cost_yuan_per_mwh: float = 40.0
    name: str = "Enhanced Rolling-Horizon"

    def __post_init__(self):
        self.actions = np.asarray([-1.6, -0.8, 0.0, 0.8, 1.6], dtype=float)
        self.grid = np.linspace(self.config.min_soc, self.config.max_soc, self.soc_bins)
        self.previous_action = 0.0

    def _soc_to_idx(self, soc: float) -> int:
        return int(np.argmin(np.abs(self.grid - soc)))

    def _terminal_value(self) -> np.ndarray:
        target = float(np.clip(self.terminal_soc_target, self.config.min_soc, self.config.max_soc))
        deviation = self.grid - target
        return -self.terminal_soc_penalty_yuan * deviation * deviation

    def _regularized_reward(self, soc: float, action: float, previous_action: float, row: pd.Series) -> tuple[float, float]:
        next_soc, reward = transition_reward(soc, action, row, self.config)
        throughput_mwh = abs(action) * self.config.dt_hours
        cycling_penalty = throughput_mwh * self.extra_cycling_cost_yuan_per_mwh
        smoothing_penalty = abs(action - previous_action) * self.smoothing_penalty_yuan_per_mw
        return next_soc, float(reward - cycling_penalty - smoothing_penalty)

    def act(self, state: dict) -> float:
        start = int(state["idx"])
        end = min(len(self.data), start + self.horizon)
        horizon_df = self.data.iloc[start:end].reset_index(drop=True)
        if len(horizon_df) == 0:
            return 0.0

        n_actions = len(self.actions)
        terminal = self._terminal_value()
        # value[t, soc_idx, prev_action_idx] = best regularized value from t.
        value = np.repeat(terminal[:, None], n_actions, axis=1)
        policy = np.zeros((len(horizon_df), self.soc_bins, n_actions), dtype=float)

        for t in range(len(horizon_df) - 1, -1, -1):
            row = horizon_df.iloc[t]
            next_value = np.empty_like(value)
            for i, soc in enumerate(self.grid):
                for prev_idx, previous_action in enumerate(self.actions):
                    best_v = -1e18
                    best_a = 0.0
                    for action_idx, action in enumerate(self.actions):
                        next_soc, reward = self._regularized_reward(
                            float(soc), float(action), float(previous_action), row
                        )
                        j = self._soc_to_idx(next_soc)
                        candidate = reward + value[j, action_idx]
                        if candidate > best_v:
                            best_v = candidate
                            best_a = float(action)
                    next_value[i, prev_idx] = best_v
                    policy[t, i, prev_idx] = best_a
            value = next_value

        current_soc_idx = self._soc_to_idx(float(state["soc"]))
        previous_action_idx = int(np.argmin(np.abs(self.actions - self.previous_action)))
        selected = float(policy[0, current_soc_idx, previous_action_idx])
        self.previous_action = selected
        return selected

