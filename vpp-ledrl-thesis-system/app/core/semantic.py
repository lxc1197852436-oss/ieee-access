from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticSignal:
    risk_score: float
    price_spike_score: float
    load_pressure_score: float
    renewable_curtailment_score: float
    explanation_hint: str
    storage_bias: float = 0.0


class LocalSemanticEncoder:
    """Deterministic local semantic scorer for demos and no-key operation."""

    def encode(self, text: str) -> SemanticSignal:
        t = (text or "").lower()
        risk = 0.05
        price = 0.0
        load = 0.0
        renewable = 0.0
        storage_bias = 0.0
        hints: list[str] = []

        if any(k in text for k in ["高温", "寒潮", "橙色预警", "红色预警"]) or "warning" in t:
            risk += 0.45
            load += 0.55
            hints.append("气象预警提高负荷和运行风险权重")
        if any(k in text for k in ["需求响应", "削峰", "晚高峰"]):
            risk += 0.2
            load += 0.35
            storage_bias -= 0.25
            hints.append("需求响应事件提示晚高峰应保留储能")
        if any(k in text for k in ["尖峰电价", "价格异常", "现货市场", "price"]):
            risk += 0.25
            price += 0.75
            storage_bias -= 0.55
            hints.append("市场公告提高价格尖峰概率")
        if any(k in text for k in ["新能源消纳", "弃光", "光伏", "curtailment"]):
            risk += 0.15
            renewable += 0.75
            storage_bias += 0.65
            hints.append("新能源消纳事件提示午间优先充电")

        risk = min(1.0, risk)
        storage_bias = max(-1.0, min(1.0, storage_bias))
        return SemanticSignal(
            risk_score=risk,
            price_spike_score=min(1.0, price),
            load_pressure_score=min(1.0, load),
            renewable_curtailment_score=min(1.0, renewable),
            storage_bias=storage_bias,
            explanation_hint="；".join(hints) if hints else "未检测到显著文本风险事件",
        )


def semantic_vector(signal: SemanticSignal) -> list[float]:
    return [
        signal.risk_score,
        signal.price_spike_score,
        signal.load_pressure_score,
        signal.renewable_curtailment_score,
        math.sqrt(max(signal.risk_score, 0.0)),
    ]
