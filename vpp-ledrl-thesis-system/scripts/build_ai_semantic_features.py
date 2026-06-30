from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import ScenarioConfig
from app.core.data import generate_china_vpp_scenario, load_vpp_dataset
from app.core.experiment_design import SCENARIOS
from app.core.llm_provider import AISemanticAssessment, LLMProvider
from scripts.run_chapter6_experiments import apply_stress


DATASET = ROOT / "data" / "processed" / "china_vpp_priority1_guangdong_sample.csv"
OUT_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw_sources" / "ai_semantic_cache"


def scenario_frames() -> list[pd.DataFrame]:
    frames = []
    for scenario in SCENARIOS:
        cfg = ScenarioConfig(
            start=scenario.start,
            periods=scenario.periods,
            freq="15min",
            seed=scenario.seed,
            region=scenario.region,
        )
        data = apply_stress(generate_china_vpp_scenario(cfg), scenario.stress_type)
        data["scenario_id"] = scenario.scenario_id
        data["scenario_name"] = scenario.name
        frames.append(data)
    return frames


def collect_events(include_priority1: bool) -> pd.DataFrame:
    frames = scenario_frames()
    if include_priority1 and DATASET.exists():
        priority1 = load_vpp_dataset(DATASET)
        priority1["scenario_id"] = "P1"
        priority1["scenario_name"] = "广东优先级1样例数据"
        frames.append(priority1)
    data = pd.concat(frames, ignore_index=True)
    cols = ["scenario_id", "scenario_name", "event_type", "event_text", "temperature_c", "price_yuan_mwh"]
    events = data[cols].drop_duplicates(subset=["event_type", "event_text"]).reset_index(drop=True)
    return events


def assessment_record(row: pd.Series, assessment: AISemanticAssessment) -> dict:
    return {
        "scenario_id": row["scenario_id"],
        "scenario_name": row["scenario_name"],
        "event_type": row["event_type"],
        "event_text": row["event_text"],
        "ai_risk_score": assessment.risk_score,
        "ai_price_spike_score": assessment.price_spike_score,
        "ai_load_pressure_score": assessment.load_pressure_score,
        "ai_renewable_curtailment_score": assessment.renewable_curtailment_score,
        "ai_recommended_storage_bias": assessment.recommended_storage_bias,
        "ai_event_summary": assessment.event_summary,
        "ai_explanation": assessment.explanation,
        "ai_provider": assessment.provider,
        "ai_model": assessment.model,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def enrich_frame(data: pd.DataFrame, event_features: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "event_type",
        "event_text",
        "ai_risk_score",
        "ai_price_spike_score",
        "ai_load_pressure_score",
        "ai_renewable_curtailment_score",
        "ai_recommended_storage_bias",
        "ai_event_summary",
        "ai_explanation",
        "ai_provider",
        "ai_model",
    ]
    return data.merge(event_features[cols], on=["event_type", "event_text"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AI semantic features for VPP thesis experiments.")
    parser.add_argument("--include-priority1", action="store_true", help="Also enrich the priority-1 Guangdong sample.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between remote API calls.")
    args = parser.parse_args()

    provider = LLMProvider()
    events = collect_events(include_priority1=args.include_priority1)
    rows: list[dict] = []
    raw_rows: list[dict] = []
    for idx, row in events.iterrows():
        context = {
            "scenario_id": row["scenario_id"],
            "scenario_name": row["scenario_name"],
            "event_type": row["event_type"],
            "temperature_c": float(row["temperature_c"]),
            "price_yuan_mwh": float(row["price_yuan_mwh"]),
        }
        assessment = provider.assess_event(str(row["event_text"]), context=context)
        record = assessment_record(row, assessment)
        rows.append(record)
        raw_rows.append({"index": int(idx), "context": context, "assessment": record})
        print(
            f"{idx + 1}/{len(events)} {row['event_type']} "
            f"risk={assessment.risk_score:.2f} price={assessment.price_spike_score:.2f} "
            f"load={assessment.load_pressure_score:.2f} renewable={assessment.renewable_curtailment_score:.2f} "
            f"provider={assessment.provider}"
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    event_features = pd.DataFrame(rows)
    write_csv(OUT_DIR / "ai_event_semantic_features.csv", rows)
    write_jsonl(RAW_DIR / "ai_event_semantic_features.jsonl", raw_rows)

    enriched_frames = []
    for frame in scenario_frames():
        enriched_frames.append(enrich_frame(frame, event_features))
    scenario_data = pd.concat(enriched_frames, ignore_index=True)
    scenario_data.to_csv(OUT_DIR / "chapter6_ai_semantic_scenarios.csv", index=False, encoding="utf-8-sig")

    if args.include_priority1 and DATASET.exists():
        priority1 = load_vpp_dataset(DATASET)
        enriched = enrich_frame(priority1, event_features)
        enriched.to_csv(OUT_DIR / "china_vpp_priority1_guangdong_sample_ai_semantic.csv", index=False, encoding="utf-8-sig")

    summary = {
        "provider": provider.provider,
        "model": rows[0]["ai_model"] if rows else provider.model,
        "event_count": len(rows),
        "event_features": str(OUT_DIR / "ai_event_semantic_features.csv"),
        "chapter6_scenarios": str(OUT_DIR / "chapter6_ai_semantic_scenarios.csv"),
    }
    (ROOT / "reports" / "ai_semantic_feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
