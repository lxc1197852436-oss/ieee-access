from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateEncoderConfig:
    include_semantic: bool = False
    semantic_mode: str = "native"
    price_center: float = 300.0
    price_scale: float = 220.0
    load_scale: float = 6.0
    pv_scale: float = 5.0
    temp_center: float = 30.0
    temp_scale: float = 8.0


class StateEncoder:
    """Convert VPP state dictionaries into normalized vectors."""

    def __init__(self, config: StateEncoderConfig | None = None):
        self.config = config or StateEncoderConfig()

    @property
    def dim(self) -> int:
        return self.feature_dim

    def encode(self, state: dict) -> np.ndarray:
        cfg = self.config
        hour = float(state["hour"])
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        features = [
            float(state["load_mw"]) / cfg.load_scale,
            float(state["pv_mw"]) / cfg.pv_scale,
            (float(state["price_yuan_mwh"]) - cfg.price_center) / cfg.price_scale,
            (float(state["temperature_c"]) - cfg.temp_center) / cfg.temp_scale,
            float(state["soc"]),
            hour_sin,
            hour_cos,
        ]
        # Keep the numeric dimension fixed at 7 but expose dim=5 would be
        # misleading; this module is new and uses 7 numeric features.
        if cfg.include_semantic:
            if cfg.semantic_mode == "native":
                semantic_features = [float(x) for x in state["semantic_vector"]]
            elif cfg.semantic_mode == "zero":
                semantic_features = [0.0] * 5
            else:
                raise ValueError(f"Unknown semantic_mode: {cfg.semantic_mode}")
            features.extend(semantic_features)
        return np.asarray(features, dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return 12 if self.config.include_semantic else 7
