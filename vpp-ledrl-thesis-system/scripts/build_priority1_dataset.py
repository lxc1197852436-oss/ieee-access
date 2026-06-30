from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import ScenarioConfig
from app.core.data import generate_china_vpp_scenario

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def read_metric_csv(path: Path) -> dict[str, float]:
    metrics = {}
    if not path.exists() or path.stat().st_size == 0:
        return metrics
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = row.get("indicator") or row.get("region")
            value = row.get("value") or row.get("value_100m_kwh")
            if key and value not in (None, ""):
                try:
                    metrics[key] = float(value)
                except ValueError:
                    pass
    return metrics


def main() -> None:
    market = read_metric_csv(PROCESSED / "guangdong_market_storage_2024.csv")
    power = read_metric_csv(PROCESSED / "nea_2024_power_consumption.csv")
    renewable = read_metric_csv(PROCESSED / "nea_2024_renewable_summary.csv")

    # Use public Guangdong spot-market disclosure as a calibration range.
    min_price = market.get("guangdong_spot_price_min_yuan_kwh", 0.068) * 1000
    max_price = market.get("guangdong_spot_price_max_yuan_kwh", 0.525) * 1000
    spread = market.get("storage_day_ahead_spread_yuan_kwh", 0.167) * 1000

    cfg = ScenarioConfig(start="2025-07-01 00:00:00", periods=96 * 30, freq="15min", seed=2026, region="广东省")
    df = generate_china_vpp_scenario(cfg)

    # Calibrate synthetic price range to crawled Guangdong disclosure while
    # preserving relative intraday volatility.
    p = df["price_yuan_mwh"]
    normalized = (p - p.min()) / max(1e-9, (p.max() - p.min()))
    df["price_yuan_mwh"] = (min_price + normalized * (max_price - min_price)).round(4)
    df["source_note"] = "公开披露数据校准 + 仿真构造；不可表述为真实15分钟VPP运行数据"

    out = PROCESSED / "china_vpp_priority1_guangdong_sample.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    meta = {
        "dataset": str(out),
        "rows": len(df),
        "region": cfg.region,
        "time_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "price_calibration_yuan_mwh": {
            "min": min_price,
            "max": max_price,
            "storage_day_ahead_spread_reference": spread,
        },
        "background_calibration": {
            "national_power_consumption_2024_100m_kwh": power.get("全社会用电量"),
            "renewable_total_capacity_2024_100m_kw": renewable.get("renewable_total_capacity"),
            "solar_total_capacity_2024_100m_kw": renewable.get("solar_total_capacity"),
            "wind_total_capacity_2024_100m_kw": renewable.get("wind_total_capacity"),
        },
        "limits": [
            "该文件是论文优先级1样例数据集，适合跑通仿真、前端展示和方法验证。",
            "其中价格范围由公开广东市场披露校准，负荷、光伏、温度和文本事件为仿真构造。",
            "正式论文中应明确其为公开数据校准的仿真场景，不能称为真实VPP运行数据。",
        ],
    }
    (REPORTS / "priority1_dataset_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

