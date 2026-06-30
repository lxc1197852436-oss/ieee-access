from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from app.core.rl.ledrl_agent import LEDRLAgent, LEDRLConfig
except ModuleNotFoundError as exc:
    print(exc)
    print("Install ML dependencies first: pip install -r requirements-ml.txt")
    raise SystemExit(1)

from app.core.data import load_vpp_dataset
from app.core.environment import VPPEnv
from app.core.simulation import calculate_metrics


DATASET = ROOT / "data" / "processed" / "china_vpp_priority1_guangdong_sample.csv"
OUT_DIR = ROOT / "outputs" / "chapter4"


def train(agent: LEDRLAgent, train_rows, episodes: int = 8, updates_per_step: int = 1) -> list[dict]:
    logs = []
    for ep in range(episodes):
        env = VPPEnv(train_rows)
        state = env.reset(initial_soc=0.45 + 0.1 * ((ep % 4) / 3))
        ep_reward = 0.0
        losses = []
        while not env.done():
            state_vec = agent.encode(state)
            action = agent.sac.act(state_vec, deterministic=False)
            next_state, reward, done, _ = env.step(action)
            agent.sac.add_transition(state_vec, action, reward, agent.encode(next_state), done)
            for _ in range(updates_per_step):
                info = agent.sac.update()
                if info:
                    losses.append(info)
            state = next_state
            ep_reward += reward
        logs.append(
            {
                "episode": ep + 1,
                "reward": ep_reward,
                "updates": len(losses),
                "last_loss": losses[-1] if losses else {},
            }
        )
        print(agent.name, logs[-1])
    return logs


def evaluate(agent: LEDRLAgent, test_rows) -> dict:
    env = VPPEnv(test_rows)
    state = env.reset(initial_soc=0.5)
    while not env.done():
        action = agent.act(state, deterministic=True)
        state, _, _, _ = env.step(action)
    return calculate_metrics(__import__("pandas").DataFrame(env.history))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Chapter 4 SAC and LE-DRL-SAC agents.")
    parser.add_argument("--episodes", type=int, default=4, help="Training episodes for each agent.")
    parser.add_argument("--updates-per-step", type=int, default=1, help="Gradient updates per environment step.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or mps.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_vpp_dataset(DATASET)
    split = int(len(data) * 0.7)
    train_rows = data.iloc[:split].reset_index(drop=True)
    test_rows = data.iloc[split:].reset_index(drop=True)

    agents = [
        LEDRLAgent(LEDRLConfig(include_semantic=False, device=args.device)),
        LEDRLAgent(LEDRLConfig(include_semantic=True, device=args.device)),
    ]
    result = {"dataset": str(DATASET), "train_rows": len(train_rows), "test_rows": len(test_rows), "agents": []}
    for agent in agents:
        logs = train(agent, train_rows, episodes=args.episodes, updates_per_step=args.updates_per_step)
        ckpt = agent.sac.save(OUT_DIR / f"{agent.name}.pt")
        metrics = evaluate(agent, test_rows)
        result["agents"].append({"name": agent.name, "checkpoint": str(ckpt), "logs": logs, "metrics": metrics})
    (OUT_DIR / "chapter4_sac_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["agents"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
