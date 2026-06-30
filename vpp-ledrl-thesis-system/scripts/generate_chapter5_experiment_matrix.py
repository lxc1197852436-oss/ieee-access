from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.experiment_design import BASELINE_MODELS, METRICS, SCENARIOS, scenario_records

OUT = ROOT / "reports"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    scenario_rows = scenario_records()
    model_rows = BASELINE_MODELS
    metric_rows = [
        {"metric_id": k, "name": name, "unit": unit, "description": desc}
        for k, name, unit, desc in METRICS
    ]
    matrix = []
    for scenario in SCENARIOS:
        for model in BASELINE_MODELS:
            matrix.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_name": scenario.name,
                    "model_id": model["model_id"],
                    "model_name": model["name"],
                    "purpose": model["chapter"],
                }
            )

    write_csv(OUT / "chapter5_scenarios.csv", scenario_rows)
    write_csv(OUT / "chapter5_models.csv", model_rows)
    write_csv(OUT / "chapter5_metrics.csv", metric_rows)
    write_csv(OUT / "chapter5_experiment_matrix.csv", matrix)
    (OUT / "chapter5_experiment_design.json").write_text(
        json.dumps(
            {
                "scenarios": scenario_rows,
                "models": model_rows,
                "metrics": metric_rows,
                "experiment_count": len(matrix),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved experiment matrix with {len(matrix)} rows.")


if __name__ == "__main__":
    main()

