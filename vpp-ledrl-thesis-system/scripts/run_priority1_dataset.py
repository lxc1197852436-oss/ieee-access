from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.data import load_vpp_dataset
from app.core.policies import POLICIES
from app.core.simulation import run_policy


def main() -> None:
    dataset = ROOT / "data" / "processed" / "china_vpp_priority1_guangdong_sample.csv"
    if not dataset.exists():
        raise FileNotFoundError(f"Run scripts/build_priority1_dataset.py first: {dataset}")
    data = load_vpp_dataset(dataset)
    results = [run_policy(POLICIES[name](), data) for name in ["rule", "ledrl", "random"]]
    out = ROOT / "outputs" / "priority1_dataset_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": str(dataset), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in results:
        print(item["policy"], item["metrics"])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

