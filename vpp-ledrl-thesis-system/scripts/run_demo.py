from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.simulation import run_experiment


def main():
    result = run_experiment()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "demo_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Demo experiment finished.")
    for item in result["results"]:
        print(item["policy"], item["metrics"])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
