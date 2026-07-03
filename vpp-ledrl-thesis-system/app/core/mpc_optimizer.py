from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.config import VPPConfig
from app.core.policies import Policy

try:
    from scipy.optimize import linprog
except ModuleNotFoundError:  # pragma: no cover - optional baseline dependency
    linprog = None


@dataclass
class LinearMPCOptimizerPolicy(Policy):
    """Linear MPC baseline for battery dispatch.

    The policy solves a deterministic linear program over a receding forecast
    horizon. It uses the same sign convention as VPPEnv: positive actions
    discharge the battery and negative actions charge it. The optimization
    includes SOC dynamics, power bounds, a curtailment slack, degradation cost,
    and a terminal SOC absolute-deviation penalty.
    """

    data: pd.DataFrame
    config: VPPConfig = VPPConfig()
    horizon: int = 24
    terminal_soc_target: float = 0.50
    terminal_soc_penalty_yuan: float = 20000.0
    extra_cycling_cost_yuan_per_mwh: float = 120.0
    name: str = "Linear-MPC"

    def _idx(self, block: str, t: int, h: int) -> int:
        if block == "charge":
            return t
        if block == "discharge":
            return h + t
        if block == "soc":
            return 2 * h + t
        if block == "curtail":
            return 2 * h + (h + 1) + t
        raise ValueError(block)

    def act(self, state: dict) -> float:
        if linprog is None:
            raise RuntimeError("scipy is required for LinearMPCOptimizerPolicy")

        start = int(state["idx"])
        end = min(len(self.data), start + self.horizon)
        horizon_df = self.data.iloc[start:end].reset_index(drop=True)
        h = len(horizon_df)
        if h == 0:
            return 0.0

        cfg = self.config
        n_vars = 2 * h + (h + 1) + h + 2  # charge, discharge, soc, curtail, dev_pos, dev_neg
        dev_pos = n_vars - 2
        dev_neg = n_vars - 1

        c = np.zeros(n_vars, dtype=float)
        for t, row in horizon_df.iterrows():
            price = float(row["price_yuan_mwh"])
            # Minimize negative profit. Base pv-load revenue is action-independent.
            cycling_cost = (cfg.degradation_cost_yuan_per_mwh + self.extra_cycling_cost_yuan_per_mwh) * cfg.dt_hours
            c[self._idx("charge", t, h)] = price * cfg.dt_hours + cycling_cost
            c[self._idx("discharge", t, h)] = -price * cfg.dt_hours + cycling_cost
            c[self._idx("curtail", t, h)] = cfg.curtailment_penalty_yuan_per_mwh * cfg.dt_hours
        c[dev_pos] = self.terminal_soc_penalty_yuan
        c[dev_neg] = self.terminal_soc_penalty_yuan

        bounds: list[tuple[float | None, float | None]] = []
        bounds.extend([(0.0, cfg.battery_power_mw)] * h)  # charge MW
        bounds.extend([(0.0, cfg.battery_power_mw)] * h)  # discharge MW
        bounds.extend([(cfg.min_soc, cfg.max_soc)] * (h + 1))
        bounds.extend([(0.0, None)] * h)  # curtailment MW equivalent before dt
        bounds.extend([(0.0, None), (0.0, None)])

        a_eq = []
        b_eq = []

        # Initial SOC.
        row0 = np.zeros(n_vars, dtype=float)
        row0[self._idx("soc", 0, h)] = 1.0
        a_eq.append(row0)
        b_eq.append(float(state["soc"]))

        # SOC dynamics.
        for t in range(h):
            row = np.zeros(n_vars, dtype=float)
            row[self._idx("soc", t + 1, h)] = 1.0
            row[self._idx("soc", t, h)] = -1.0
            row[self._idx("charge", t, h)] = -cfg.charge_efficiency * cfg.dt_hours / cfg.battery_capacity_mwh
            row[self._idx("discharge", t, h)] = cfg.dt_hours / (cfg.discharge_efficiency * cfg.battery_capacity_mwh)
            a_eq.append(row)
            b_eq.append(0.0)

        # Terminal absolute deviation: soc_H - target = dev_pos - dev_neg.
        target = float(np.clip(self.terminal_soc_target, cfg.min_soc, cfg.max_soc))
        row = np.zeros(n_vars, dtype=float)
        row[self._idx("soc", h, h)] = 1.0
        row[dev_pos] = -1.0
        row[dev_neg] = 1.0
        a_eq.append(row)
        b_eq.append(target)

        a_ub = []
        b_ub = []
        for t, row_data in horizon_df.iterrows():
            pv_surplus = float(row_data["pv_mw"]) - float(row_data["load_mw"])
            # curtail >= pv - load - charge  => -curtail - charge <= -(pv-load)
            row = np.zeros(n_vars, dtype=float)
            row[self._idx("curtail", t, h)] = -1.0
            row[self._idx("charge", t, h)] = -1.0
            a_ub.append(row)
            b_ub.append(-pv_surplus)
            # Avoid simultaneous full charge and discharge.
            row2 = np.zeros(n_vars, dtype=float)
            row2[self._idx("charge", t, h)] = 1.0
            row2[self._idx("discharge", t, h)] = 1.0
            a_ub.append(row2)
            b_ub.append(cfg.battery_power_mw)

        result = linprog(
            c,
            A_ub=np.asarray(a_ub),
            b_ub=np.asarray(b_ub),
            A_eq=np.asarray(a_eq),
            b_eq=np.asarray(b_eq),
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            return 0.0
        charge0 = float(result.x[self._idx("charge", 0, h)])
        discharge0 = float(result.x[self._idx("discharge", 0, h)])
        return discharge0 - charge0
