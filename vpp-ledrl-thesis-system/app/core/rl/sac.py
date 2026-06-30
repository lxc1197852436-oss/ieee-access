from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise ModuleNotFoundError(
        "PyTorch is required for continuous SAC. Install with: pip install -r requirements-ml.txt"
    ) from exc

from app.core.rl.replay_buffer import ReplayBuffer


LOG_STD_MIN = -20
LOG_STD_MAX = 2


@dataclass
class SACConfig:
    state_dim: int
    action_dim: int = 1
    action_limit: float = 2.0
    hidden_dim: int = 128
    gamma: float = 0.97
    tau: float = 0.01
    alpha: float = 0.2
    lr: float = 3e-4
    batch_size: int = 128
    replay_capacity: int = 80_000
    warmup_steps: int = 500
    device: str = "cpu"
    semantic_actor_loss_weight: float = 0.0


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_limit: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.action_limit = float(action_limit)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        tanh_action = torch.tanh(raw)
        action = tanh_action * self.action_limit
        log_prob = dist.log_prob(raw) - torch.log(1 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        mean, _ = self(state)
        return torch.tanh(mean) * self.action_limit


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden_dim)
        self.q2 = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent:
    def __init__(self, config: SACConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.actor = GaussianActor(config.state_dim, config.action_dim, config.hidden_dim, config.action_limit).to(self.device)
        self.critic = TwinCritic(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target = TwinCritic(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        self.replay = ReplayBuffer(config.state_dim, config.action_dim, config.replay_capacity)
        self.total_steps = 0

    def act(self, state_vec: np.ndarray, deterministic: bool = False) -> float:
        state = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(state)
            else:
                action, _ = self.actor.sample(state)
        return float(action.cpu().numpy().reshape(-1)[0])

    def add_transition(self, state, action, reward, next_state, done) -> None:
        self.replay.add(state, np.asarray([action], dtype=np.float32), reward, next_state, done)

    def update(self) -> dict:
        cfg = self.config
        if self.replay.size < max(cfg.batch_size, cfg.warmup_steps):
            return {}
        batch = self.replay.sample(cfg.batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions, next_logp = self.actor.sample(next_states)
            tq1, tq2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(tq1, tq2) - cfg.alpha * next_logp
            target = rewards + cfg.gamma * (1 - dones) * target_q

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_actions, logp = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, new_actions)
        actor_loss = (cfg.alpha * logp - torch.min(q1_pi, q2_pi)).mean()
        semantic_actor_loss = torch.tensor(0.0, device=self.device)
        if cfg.semantic_actor_loss_weight > 0.0 and states.shape[1] >= 12:
            semantic_targets = semantic_target_actions(states, cfg.action_limit)
            semantic_actor_loss = F.mse_loss(new_actions, semantic_targets)
            actor_loss = actor_loss + cfg.semantic_actor_loss_weight * semantic_actor_loss
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.mul_(1 - cfg.tau).add_(cfg.tau * param.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "semantic_actor_loss": float(semantic_actor_loss.item()),
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            out,
        )
        return out

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "SACAgent":
        payload = torch.load(path, map_location=device or "cpu")
        cfg = SACConfig(**payload["config"])
        if device:
            cfg.device = device
        agent = cls(cfg)
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        agent.critic_target.load_state_dict(payload["critic_target"])
        return agent


def semantic_target_actions(states, action_limit: float):
    """Build differentiable action targets from encoded semantic risk features.

    Encoded state layout comes from StateEncoder:
    price_norm index 2, SOC index 4, semantic indices 7:12.
    The semantic slice is [risk, price_spike, load_pressure,
    renewable_curtailment, storage_bias]. Positive action discharges; negative
    action charges.
    """
    price_norm = states[:, 2:3]
    soc = states[:, 4:5]
    price_spike = torch.clamp(states[:, 8:9], 0.0, 1.0)
    load_pressure = torch.clamp(states[:, 9:10], 0.0, 1.0)
    renewable = torch.clamp(states[:, 10:11], 0.0, 1.0)
    storage_bias = torch.clamp(states[:, 11:12], -1.0, 1.0)

    low_price = torch.sigmoid((-0.18 - price_norm) * 8.0)
    high_price = torch.sigmoid((price_norm - 1.00) * 6.0)
    can_charge = torch.sigmoid((0.86 - soc) * 20.0)
    can_discharge = torch.sigmoid((soc - 0.20) * 20.0)

    storage_charge = torch.clamp(storage_bias, min=0.0) * 0.75
    storage_discharge = torch.clamp(-storage_bias, min=0.0) * 0.75
    charge_need = torch.maximum(torch.maximum(low_price * 0.65, renewable * 0.85), storage_charge) * can_charge
    discharge_need = torch.maximum(
        torch.maximum(high_price * 0.85, torch.maximum(price_spike, load_pressure) * 0.55),
        storage_discharge,
    ) * can_discharge
    target = (discharge_need - charge_need) * float(action_limit) * 0.8
    return torch.clamp(target, -float(action_limit), float(action_limit))
