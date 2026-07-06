from dataclasses import dataclass


@dataclass(frozen=True)
class VPPConfig:
    pv_capacity_mw: float = 5.0
    battery_power_mw: float = 2.0
    battery_capacity_mwh: float = 8.0
    initial_soc: float = 0.5
    min_soc: float = 0.1
    max_soc: float = 0.9
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    degradation_cost_yuan_per_mwh: float = 18.0
    curtailment_penalty_yuan_per_mwh: float = 80.0
    violation_penalty_yuan: float = 300.0
    dt_hours: float = 0.25
    # Reserve (frequency-regulation) reward: penalizes SOC deviation from a target
    # mid-band so that "keep SOC in the middle for bidirectional reserve" has a
    # reward payoff. Default 0.0 => disabled, preserving the original reward for
    # all existing scenarios (S1-S5, S7). S8 enables it to make the
    # "reserve-capacity" event semantics load-bearing in the reward.
    reserve_soc_target: float = 0.5
    reserve_penalty_yuan_per_dev: float = 0.0


@dataclass(frozen=True)
class ScenarioConfig:
    start: str = "2025-07-01 00:00:00"
    periods: int = 96 * 14
    freq: str = "15min"
    seed: int = 42
    region: str = "广东省"

