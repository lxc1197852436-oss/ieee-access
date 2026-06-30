from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Policy:
    name = "policy"

    def act(self, state: dict) -> float:
        raise NotImplementedError


@dataclass
class RuleBasedPolicy(Policy):
    name: str = "Rule-Based"
    low_price: float = 260.0
    high_price: float = 520.0
    power_mw: float = 1.6

    def act(self, state: dict) -> float:
        price = state["price_yuan_mwh"]
        hour = state["hour"]
        pv_surplus = state["pv_mw"] - state["load_mw"]
        if pv_surplus > 0.4 and state["soc"] < 0.86:
            return -self.power_mw
        if price < self.low_price and state["soc"] < 0.82:
            return -self.power_mw
        if price > self.high_price and state["soc"] > 0.18:
            return self.power_mw
        if 18 <= hour <= 21 and state["soc"] > 0.35:
            return self.power_mw * 0.7
        return 0.0


@dataclass
class SemanticEnhancedPolicy(Policy):
    """LE-DRL placeholder policy using semantic risk scores.

    This is intentionally deterministic and light for the first package. It is
    shaped like the future LE-DRL policy: numerical state + semantic signal
    decide the dispatch action. The training implementation can later replace
    this class without changing backend/frontend contracts.
    """

    name: str = "LE-DRL-Semantic"
    max_power_mw: float = 2.0

    def act(self, state: dict) -> float:
        price = state["price_yuan_mwh"]
        soc = state["soc"]
        hour = state["hour"]
        sem = state["semantic"]
        pv_surplus = state["pv_mw"] - state["load_mw"]

        # Noon renewable absorption: charge more aggressively when text says
        # renewable curtailment pressure is high.
        if pv_surplus > 0.2 and soc < 0.88:
            charge = 0.9 + sem.renewable_curtailment_score * 0.9
            return -min(self.max_power_mw, charge)

        # If high-price or demand-response text appears before evening peak,
        # preserve SOC instead of discharging too early.
        if 14 <= hour < 18 and sem.load_pressure_score > 0.35 and soc < 0.65:
            return -0.8

        if price > 420 or (18 <= hour <= 22 and (price > 360 or sem.price_spike_score > 0.3)):
            if soc > 0.22:
                risk_boost = 0.4 * sem.price_spike_score + 0.25 * sem.load_pressure_score
                return min(self.max_power_mw, 1.2 + risk_boost)

        if price < 190 and soc < 0.84:
            return -1.2

        return 0.0


@dataclass
class RandomPolicy(Policy):
    name: str = "Random"
    seed: int = 7
    max_power_mw: float = 2.0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def act(self, state: dict) -> float:
        return float(self.rng.uniform(-self.max_power_mw, self.max_power_mw))


POLICIES = {
    "rule": RuleBasedPolicy,
    "ledrl": SemanticEnhancedPolicy,
    "random": RandomPolicy,
}
