from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.policies import Policy


def _bucket(value: float, bins: list[float]) -> int:
    for i, edge in enumerate(bins):
        if value < edge:
            return i
    return len(bins)


def _logsumexp(values: np.ndarray) -> float:
    max_v = float(np.max(values))
    return max_v + math.log(float(np.sum(np.exp(values - max_v))))


@dataclass
class DiscreteSoftQAgent:
    """Small maximum-entropy Q-learning agent for midterm experiments.

    This agent discretizes the continuous VPP dispatch action into a small set
    of MW values. It is intentionally lightweight: it proves the training and
    evaluation loop without introducing PyTorch as a hard dependency. The final
    thesis can replace it with continuous SAC while keeping the same data and
    evaluation scripts.
    """

    use_semantic: bool = False
    gamma: float = 0.96
    alpha: float = 0.35
    lr: float = 0.12
    epsilon: float = 0.12
    actions: list[float] = field(default_factory=lambda: [-2.0, -1.2, -0.6, 0.0, 0.6, 1.2, 2.0])
    q_table: dict[str, list[float]] = field(default_factory=dict)

    def state_key(self, state: dict) -> str:
        hour = int(state["hour"] // 4)
        price = _bucket(float(state["price_yuan_mwh"]), [150, 230, 320, 410, 500])
        soc = _bucket(float(state["soc"]), [0.18, 0.3, 0.45, 0.6, 0.75, 0.86])
        pv_surplus = _bucket(float(state["pv_mw"]) - float(state["load_mw"]), [-1.5, -0.3, 0.3, 1.2])
        temp = _bucket(float(state["temperature_c"]), [28, 31, 34, 37])
        parts = [hour, price, soc, pv_surplus, temp]
        if self.use_semantic:
            sem = state["semantic"]
            parts.extend(
                [
                    _bucket(float(sem.risk_score), [0.25, 0.5, 0.75]),
                    _bucket(float(sem.price_spike_score), [0.3, 0.6]),
                    _bucket(float(sem.load_pressure_score), [0.3, 0.6]),
                    _bucket(float(sem.renewable_curtailment_score), [0.3, 0.6]),
                ]
            )
        return "|".join(map(str, parts))

    def q_values(self, key: str) -> np.ndarray:
        if key not in self.q_table:
            self.q_table[key] = [0.0 for _ in self.actions]
        return np.array(self.q_table[key], dtype=float)

    def select_action(self, state: dict, explore: bool = True) -> float:
        key = self.state_key(state)
        q = self.q_values(key)
        if explore and random.random() < self.epsilon:
            return random.choice(self.actions)
        probs = self.action_probabilities(q)
        idx = int(np.random.choice(len(self.actions), p=probs)) if explore else int(np.argmax(q))
        return float(self.actions[idx])

    def action_probabilities(self, q: np.ndarray) -> np.ndarray:
        logits = q / max(self.alpha, 1e-6)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def update(self, state: dict, action: float, reward: float, next_state: dict, done: bool) -> None:
        key = self.state_key(state)
        next_key = self.state_key(next_state)
        q = self.q_values(key)
        action_idx = min(range(len(self.actions)), key=lambda i: abs(self.actions[i] - action))
        if done:
            target = reward
        else:
            next_q = self.q_values(next_key)
            soft_value = self.alpha * _logsumexp(next_q / max(self.alpha, 1e-6))
            target = reward + self.gamma * soft_value
        q[action_idx] = (1.0 - self.lr) * q[action_idx] + self.lr * target
        self.q_table[key] = q.tolist()

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "use_semantic": self.use_semantic,
                    "gamma": self.gamma,
                    "alpha": self.alpha,
                    "lr": self.lr,
                    "epsilon": self.epsilon,
                    "actions": self.actions,
                    "q_table": self.q_table,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "DiscreteSoftQAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            use_semantic=bool(data["use_semantic"]),
            gamma=float(data["gamma"]),
            alpha=float(data["alpha"]),
            lr=float(data["lr"]),
            epsilon=float(data["epsilon"]),
            actions=[float(x) for x in data["actions"]],
        )
        agent.q_table = {str(k): [float(x) for x in v] for k, v in data["q_table"].items()}
        return agent


@dataclass
class SoftQPolicy(Policy):
    agent: DiscreteSoftQAgent
    name: str = "Soft-Q"

    def act(self, state: dict) -> float:
        return self.agent.select_action(state, explore=False)

