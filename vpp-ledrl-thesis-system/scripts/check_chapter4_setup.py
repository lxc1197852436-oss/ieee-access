from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def main() -> None:
    status = {
        "torch_available": importlib.util.find_spec("torch") is not None,
        "chapter4_files": [
            "app/core/rl/state_encoder.py",
            "app/core/rl/replay_buffer.py",
            "app/core/rl/sac.py",
            "app/core/rl/ledrl_agent.py",
            "scripts/train_chapter4_sac.py",
        ],
    }
    status["missing_files"] = [p for p in status["chapter4_files"] if not Path(p).exists()]
    status["ready_for_training"] = status["torch_available"] and not status["missing_files"]
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

