from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "chapter5_current_experiment_index.csv"


def read_midterm() -> list[dict]:
    path = ROOT / "outputs" / "midterm" / "midterm_model_comparison.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["source"] = "midterm_softq"
        row["status"] = "completed"
    return rows


def read_chapter4() -> list[dict]:
    path = ROOT / "outputs" / "chapter4" / "chapter4_sac_results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data.get("agents", []):
        m = item["metrics"]
        rows.append(
            {
                "policy": item["name"],
                "total_reward_yuan": m["total_reward_yuan"],
                "mean_reward_yuan": m["mean_reward_yuan"],
                "cvar_5_yuan": m["cvar_5_yuan"],
                "battery_throughput_mwh": m["battery_throughput_mwh"],
                "high_price_discharge_rate": m["high_price_discharge_rate"],
                "low_price_charge_rate": m["low_price_charge_rate"],
                "final_soc": m["final_soc"],
                "event_count": m["event_count"],
                "source": "chapter4_short_sac",
                "status": "short_train_only",
            }
        )
    return rows


def main() -> None:
    rows = read_midterm() + read_chapter4()
    if not rows:
        raise FileNotFoundError("No experiment results found.")
    fieldnames = [
        "source",
        "status",
        "policy",
        "total_reward_yuan",
        "mean_reward_yuan",
        "cvar_5_yuan",
        "battery_throughput_mwh",
        "high_price_discharge_rate",
        "low_price_charge_rate",
        "final_soc",
        "event_count",
    ]
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

