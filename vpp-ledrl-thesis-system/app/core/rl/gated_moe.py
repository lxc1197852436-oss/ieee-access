"""B3 gated mixture-of-experts SAC for event-coverage-adaptive VPP dispatch.

Architecture:
  - SAC-prior: SAC with a strong semantic-consistency regularizer; aligns the
    actor with the hand-crafted semantic target. Strong on known events where
    the prior is well-designed.
  - SAC-free: SAC with no regularizer; free to learn from LLM features as
    state. Strong on unseen events where the prior has no matching branch.
  - Gate: a small MLP that maps the encoded state (numeric + semantic) to a
    scalar weight w in [0,1]. The final action blends the two actors:
        a = (1-w) * a_free + w * a_prior
    The gate is trained by an advantage-weighted rule: it should follow the
    actor whose critic assigns a higher Q value to the current state-action.
    This lets the gate discover, from data, that known events favor the prior
    actor (high w) and unseen events favor the free actor (low w), without
    being told which is which.

The V4 boundary (night wind surplus, no PV) motivated this design: a single
LE-DRL-SAC trained on midday-PV surplus locks onto "PV surplus + negative
price -> charge" and underperforms on V4 where the cached semantic scores are
no longer diagnostic. A gate that reads Q-differences can detect this and
defer to the free actor.

Honesty note: this module trains one gate per seed. Results are reported as
produced. The gate is NOT guaranteed to learn the intended coverage pattern;
if it collapses to a constant, that is reported as a negative result.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(1) from exc

from app.core.rl.sac import SACAgent, SACConfig
from app.core.rl.state_encoder import StateEncoder, StateEncoderConfig


@dataclass(frozen=True)
class GatedMoEConfig:
    state_dim: int
    hidden_dim: int = 64
    lr_gate: float = 3e-4
    device: str = "cpu"
    # regularization weights for the two SAC experts
    prior_actor_loss_weight: float = 3.0
    free_actor_loss_weight: float = 0.0
    name: str = "Gated-MoE"
    # Exposed so the shared semantic_auxiliary_reward helper (which reads
    # agent.config.include_semantic / semantic_mode) works for this agent.
    include_semantic: bool = True
    semantic_mode: str = "native"


class GateNet(nn.Module):
    """Maps encoded state to a scalar blend weight w in [0,1].

    w = sigmoid(g(state)); w=1 means defer to SAC-prior, w=0 to SAC-free.
    """

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(state))


class GatedMoEAgent:
    """Mixture-of-two-SAC-experts with a learned gate.

    Each expert is a full SACAgent (actor + twin critic + target). The two
    experts share the same StateEncoder but maintain separate replay buffers
    because their exploration regimes differ. The gate is trained to follow
    whichever expert's critic reports a higher Q for the current state.
    """

    def __init__(self, config: GatedMoEConfig, encoder: StateEncoder):
        self.config = config
        self.encoder = encoder
        self.device = torch.device(config.device)

        # SAC-prior: strong semantic regularizer (aligns with prior on known events)
        self.sac_prior = SACAgent(SACConfig(
            state_dim=config.state_dim, action_dim=1, action_limit=2.0,
            hidden_dim=128, gamma=0.97, tau=0.01, alpha=0.2, lr=3e-4,
            batch_size=64, warmup_steps=256, device=config.device,
            semantic_actor_loss_weight=config.prior_actor_loss_weight,
        ))
        # SAC-free: no regularizer (adapts to unseen events)
        self.sac_free = SACAgent(SACConfig(
            state_dim=config.state_dim, action_dim=1, action_limit=2.0,
            hidden_dim=128, gamma=0.97, tau=0.01, alpha=0.2, lr=3e-4,
            batch_size=64, warmup_steps=256, device=config.device,
            semantic_actor_loss_weight=config.free_actor_loss_weight,
        ))
        self.gate = GateNet(config.state_dim, config.hidden_dim).to(self.device)
        self.gate_opt = torch.optim.Adam(self.gate.parameters(), lr=config.lr_gate)

    @property
    def name(self) -> str:
        return self.config.name

    def encode(self, state: dict) -> np.ndarray:
        return self.encoder.encode(state)

    def act(self, state: dict, deterministic: bool = False, return_w: bool = False):
        sv = self.encoder.encode(state)
        sv_t = torch.as_tensor(sv, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a_prior = float(self.sac_prior.actor.deterministic(sv_t).cpu().numpy().reshape(-1)[0])
            a_free = float(self.sac_free.actor.deterministic(sv_t).cpu().numpy().reshape(-1)[0])
            w = float(self.gate(sv_t).cpu().numpy().reshape(-1)[0])
        # During training, sample stochastic actions; blend deterministically.
        if not deterministic:
            with torch.no_grad():
                ap, _ = self.sac_prior.actor.sample(sv_t)
                af, _ = self.sac_free.actor.sample(sv_t)
            a_prior = float(ap.cpu().numpy().reshape(-1)[0])
            a_free = float(af.cpu().numpy().reshape(-1)[0])
        action = (1.0 - w) * a_free + w * a_prior
        if return_w:
            return float(np.clip(action, -2.0, 2.0)), w
        return float(np.clip(action, -2.0, 2.0))

    def gate_weight(self, state_vec: np.ndarray) -> float:
        sv_t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.gate(sv_t).cpu().numpy().reshape(-1)[0])

    def update_gate(self, states_np: np.ndarray) -> float:
        """Train the gate to follow the expert with higher Q.

        For each state in the batch, compute Q_prior and Q_free under each
        expert's critic (using that expert's own current actor action). The
        gate target is 1 (favor prior) where Q_prior > Q_free, else 0. Train
        gate with BCE on this soft target.
        """
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a_prior = self.sac_prior.actor.deterministic(states)
            a_free = self.sac_free.actor.deterministic(states)
            q1p, q2p = self.sac_prior.critic(states, a_prior)
            q1f, q2f = self.sac_free.critic(states, a_free)
            q_prior = torch.min(q1p, q2p).squeeze(-1)
            q_free = torch.min(q1f, q2f).squeeze(-1)
            # soft target: sigmoid of Q-difference, smoothed
            diff = (q_prior - q_free)
            target = torch.sigmoid(diff * 0.05)  # gentle, avoid hard labels
            target = target.unsqueeze(-1)
        w_pred = self.gate(states)
        loss = F.binary_cross_entropy(w_pred, target)
        self.gate_opt.zero_grad()
        loss.backward()
        self.gate_opt.step()
        return float(loss.item())

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": self.config.__dict__,
            "gate": self.gate.state_dict(),
        }, out)
        self.sac_prior.save(str(out).replace(".pt", "_prior.pt"))
        self.sac_free.save(str(out).replace(".pt", "_free.pt"))
        return out

    @classmethod
    def load(cls, path: str | Path, encoder: StateEncoder, device: str = "cpu") -> "GatedMoEAgent":
        path = Path(path)
        payload = torch.load(path, map_location=device)
        cfg = GatedMoEConfig(**payload["config"], device=device)
        agent = cls(cfg, encoder)
        agent.gate.load_state_dict(payload["gate"])
        prior_path = str(path).replace(".pt", "_prior.pt")
        free_path = str(path).replace(".pt", "_free.pt")
        agent.sac_prior = SACAgent.load(prior_path, device=device)
        agent.sac_free = SACAgent.load(free_path, device=device)
        return agent
