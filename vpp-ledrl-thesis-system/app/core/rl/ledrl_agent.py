from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.config import VPPConfig
from app.core.rl.sac import SACAgent, SACConfig
from app.core.rl.state_encoder import StateEncoder, StateEncoderConfig


@dataclass(frozen=True)
class LEDRLConfig:
    include_semantic: bool = True
    semantic_mode: str = "native"
    action_limit: float = 2.0
    hidden_dim: int = 128
    batch_size: int = 128
    warmup_steps: int = 500
    gamma: float = 0.97
    tau: float = 0.01
    alpha: float = 0.2
    lr: float = 3e-4
    device: str = "cpu"
    name: str | None = None
    use_ai_semantics: bool = False
    semantic_guidance_weight: float = 0.0
    semantic_guidance_power: float = 1.6
    semantic_actor_loss_weight: float = 0.0


class LEDRLAgent:
    """Language-enhanced SAC wrapper.

    When include_semantic=True, text-derived semantic features are concatenated
    with numeric VPP features. When False, the same SAC implementation becomes
    the numeric-only baseline.
    """

    def __init__(self, config: LEDRLConfig):
        self.config = config
        self.vpp_config = VPPConfig()
        self.encoder = StateEncoder(
            StateEncoderConfig(include_semantic=config.include_semantic, semantic_mode=config.semantic_mode)
        )
        self.sac = SACAgent(
            SACConfig(
                state_dim=self.encoder.feature_dim,
                action_dim=1,
                action_limit=config.action_limit,
                hidden_dim=config.hidden_dim,
                gamma=config.gamma,
                tau=config.tau,
                alpha=config.alpha,
                lr=config.lr,
                batch_size=config.batch_size,
                warmup_steps=config.warmup_steps,
                device=config.device,
                semantic_actor_loss_weight=config.semantic_actor_loss_weight,
            )
        )

    @property
    def name(self) -> str:
        if self.config.name:
            return self.config.name
        return "LE-DRL-SAC" if self.config.include_semantic else "SAC-Numeric"

    def encode(self, state: dict):
        return self.encoder.encode(state)

    def act(self, state: dict, deterministic: bool = False) -> float:
        base_action = self.sac.act(self.encode(state), deterministic=deterministic)
        return self._apply_semantic_guidance(state, base_action)

    def _apply_semantic_guidance(self, state: dict, base_action: float) -> float:
        cfg = self.config
        weight = float(np.clip(cfg.semantic_guidance_weight, 0.0, 1.0))
        if not cfg.include_semantic or cfg.semantic_mode != "native" or weight <= 0.0:
            return self._clip_to_feasible(state, base_action)

        prior = self._semantic_prior_action(state)
        action = (1.0 - weight) * float(base_action) + weight * prior
        return self._clip_to_feasible(state, action)

    def _clip_to_feasible(self, state: dict, action: float) -> float:
        cfg = self.vpp_config
        soc = float(state["soc"])
        max_discharge = max(0.0, (soc - cfg.min_soc) * cfg.battery_capacity_mwh)
        max_discharge_mw = max_discharge * cfg.discharge_efficiency / cfg.dt_hours
        max_charge = max(0.0, (cfg.max_soc - soc) * cfg.battery_capacity_mwh)
        max_charge_mw = max_charge / cfg.charge_efficiency / cfg.dt_hours
        high = min(self.config.action_limit, max_discharge_mw)
        low = -min(self.config.action_limit, max_charge_mw)
        return float(np.clip(action, low, high))

    def _semantic_prior_action(self, state: dict) -> float:
        """Policy prior derived from text semantics and current observable state.

        Sign convention follows the environment: positive discharges the battery,
        negative charges it. The prior is deliberately limited to current state
        variables and semantic event scores, so it does not use future prices or
        loads during evaluation.
        """
        cfg = self.config
        sem = state["semantic"]
        price = float(state["price_yuan_mwh"])
        soc = float(state["soc"])
        hour = float(state["hour"])
        pv_surplus = float(state["pv_mw"]) - float(state["load_mw"])
        max_power = min(cfg.action_limit, cfg.semantic_guidance_power)

        charge_score = 0.0
        discharge_score = 0.0

        if pv_surplus > 0.2 and soc < 0.88:
            charge_score = max(charge_score, 0.55 + 0.45 * sem.renewable_curtailment_score)
        if price < 260.0 and soc < 0.82:
            charge_score = max(charge_score, 0.75)
        if sem.storage_bias > 0.1 and soc < 0.86:
            charge_score = max(charge_score, 0.45 + 0.55 * sem.storage_bias)

        if price > 520.0 and soc > 0.18:
            discharge_score = max(discharge_score, 0.85)
        if 18.0 <= hour <= 22.0 and soc > 0.30:
            discharge_score = max(
                discharge_score,
                0.45 + 0.35 * max(sem.price_spike_score, sem.load_pressure_score),
            )
        if sem.storage_bias < -0.1 and soc > 0.24 and (price > 360.0 or 18.0 <= hour <= 22.0):
            discharge_score = max(discharge_score, 0.45 + 0.45 * abs(sem.storage_bias))

        # Before the evening peak, high load-pressure text should reserve SOC
        # instead of forcing early discharge unless the observed price is already high.
        if 14.0 <= hour < 18.0 and sem.load_pressure_score > 0.45 and price < 520.0:
            discharge_score *= 0.35
            if soc < 0.68:
                charge_score = max(charge_score, 0.45)

        if charge_score <= 0.0 and discharge_score <= 0.0:
            return 0.0
        if charge_score >= discharge_score:
            return -max_power * min(1.0, charge_score)
        return max_power * min(1.0, discharge_score)
