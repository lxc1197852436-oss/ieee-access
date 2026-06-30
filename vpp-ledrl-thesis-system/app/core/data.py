from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import ScenarioConfig


EVENTS = [
    ("正常运行", "Status: Normal"),
    ("高温预警", "广东气象台发布高温橙色预警，预计晚高峰空调负荷显著上升。"),
    ("需求响应", "电力交易中心发布需求响应邀约，鼓励用户在晚高峰削减负荷。"),
    ("价格尖峰", "现货市场公告提示日前价格异常波动，晚高峰可能出现尖峰电价。"),
    ("新能源消纳", "调度公告提示午间新能源消纳压力增大，建议提升储能充电能力。"),
]


def generate_china_vpp_scenario(config: ScenarioConfig | None = None) -> pd.DataFrame:
    """Generate a reproducible China-style VPP dispatch scenario.

    The data are synthetic but shaped by Chinese VPP thesis needs: summer load
    peaks, photovoltaic midday output, electricity price spikes, and Chinese
    text events that can be consumed by semantic-enhanced policies.
    """
    cfg = config or ScenarioConfig()
    rng = np.random.default_rng(cfg.seed)
    ts = pd.date_range(cfg.start, periods=cfg.periods, freq=cfg.freq)
    n = len(ts)
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    day = np.arange(n) / 96.0

    summer_factor = 1.0 + 0.08 * np.sin(2 * np.pi * day / 14.0)
    evening_peak = np.exp(-0.5 * ((hour - 20.0) / 2.4) ** 2)
    noon_load = 0.35 * np.exp(-0.5 * ((hour - 14.0) / 3.2) ** 2)
    base_load = 3.4 + 1.2 * evening_peak + noon_load
    load_mw = np.maximum(1.2, base_load * summer_factor + rng.normal(0, 0.12, n))

    solar_shape = np.maximum(0.0, np.sin((hour - 6.2) / 12.0 * np.pi))
    cloud = np.clip(0.25 + 0.25 * np.sin(2 * np.pi * day / 5.0) + rng.normal(0, 0.08, n), 0, 0.8)
    pv_mw = np.maximum(0.0, 5.0 * solar_shape * (1.0 - cloud) + rng.normal(0, 0.08, n))

    temperature_c = 29 + 5.8 * np.exp(-0.5 * ((hour - 15.0) / 4.0) ** 2) + rng.normal(0, 0.5, n)
    price_base = 380 + 140 * evening_peak - 90 * solar_shape + 5 * (temperature_c - 30)
    price_yuan_mwh = np.maximum(80, price_base + rng.normal(0, 25, n))

    event_type = np.array(["正常运行"] * n, dtype=object)
    event_text = np.array([EVENTS[0][1]] * n, dtype=object)

    hot_slots = np.where((temperature_c > 34) & (hour >= 11) & (hour <= 18))[0]
    response_slots = np.where((hour >= 17.5) & (hour <= 21.5) & (temperature_c > 33))[0]
    pv_slots = np.where((pv_mw > 3.0) & (load_mw < 3.8) & (hour >= 10) & (hour <= 14))[0]
    spike_slots = np.where((price_yuan_mwh > np.quantile(price_yuan_mwh, 0.9)) & (hour >= 18))[0]

    for idxs, label_idx, stride in [
        (hot_slots, 1, 11),
        (response_slots, 2, 9),
        (spike_slots, 3, 7),
        (pv_slots, 4, 13),
    ]:
        for i in idxs[::stride]:
            event_type[i] = EVENTS[label_idx][0]
            event_text[i] = EVENTS[label_idx][1]

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "region": cfg.region,
            "load_mw": load_mw.round(4),
            "pv_mw": pv_mw.round(4),
            "price_yuan_mwh": price_yuan_mwh.round(4),
            "temperature_c": temperature_c.round(3),
            "event_type": event_type,
            "event_text": event_text,
        }
    )
    return df


def save_scenario_csv(path: str | Path, config: ScenarioConfig | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_china_vpp_scenario(config)
    df.to_csv(out, index=False)
    return out


def load_vpp_dataset(path: str | Path) -> pd.DataFrame:
    """Load a VPP dataset with the standard thesis-system schema."""
    df = pd.read_csv(path)
    required = {
        "timestamp",
        "region",
        "load_mw",
        "pv_mw",
        "price_yuan_mwh",
        "temperature_c",
        "event_type",
        "event_text",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def scenario_metadata(config: ScenarioConfig | None = None) -> dict:
    cfg = config or ScenarioConfig()
    return asdict(cfg)
