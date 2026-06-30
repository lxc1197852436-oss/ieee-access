from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"


def read_renewable_summary() -> dict[str, float]:
    path = PROCESSED / "nea_2024_renewable_summary.csv"
    out = {}
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                out[row["indicator"]] = float(row["value"])
            except (ValueError, KeyError):
                pass
    return out


def main() -> None:
    metrics = read_renewable_summary()
    wind = metrics.get("wind_utilization_rate", 95.9)
    solar = metrics.get("solar_utilization_rate", 96.8)

    rows = [
        {
            "region": "全国",
            "wind_utilization_2024_pct": wind,
            "solar_utilization_2024_pct": solar,
            "source_id": "nea_2024_renewable_summary",
            "note": "国家能源局2024年可再生能源并网运行情况，全国平均利用率；用于替代省级消纳表作为权威背景校准。",
        },
        {
            "region": "广东省样例场景",
            "wind_utilization_2024_pct": "",
            "solar_utilization_2024_pct": solar,
            "source_id": "model_derived_proxy",
            "note": "省级逐时消纳压力不使用外部表，改由VPP环境中的光伏剩余功率与弃光惩罚项动态刻画。",
        },
    ]

    out = PROCESSED / "new_energy_utilization_substitute_2024.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

