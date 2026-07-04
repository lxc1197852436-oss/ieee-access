"""Score the S5 unseen event with DeepSeek and produce an AI-semantic S5 dataset.

This mirrors scripts/build_ai_semantic_features.py but only for the S5 negative-
price event template. The DeepSeek assessment is cached to a JSONL record for
reproducibility, and merged onto the S5 scenario CSV so VPPEnv reads the ai_*
columns directly (see environment._semantic_signal).

We use the same DeepSeek model (deepseek-chat) and temperature=0.1 as the
training-time semantic feature build, so the S5 scores are comparable with the
S1-S4 cached scores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.llm_provider import LLMProvider

S5_CSV = ROOT / "data" / "processed" / "s5_negative_price_surplus.csv"
OUT_CSV = ROOT / "data" / "processed" / "s5_negative_price_surplus_ai_semantic.csv"
RAW_DIR = ROOT / "data" / "raw_sources" / "ai_semantic_cache"


def main() -> None:
    df = pd.read_csv(S5_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Distinct event texts present in S5 (the unseen negative-price event plus
    # the inherited "Status: Normal" baseline text).
    event_cols = ["event_type", "event_text", "temperature_c", "price_yuan_mwh"]
    events = df[event_cols].drop_duplicates(subset=["event_type", "event_text"]).reset_index(drop=True)

    provider = LLMProvider()
    print(f"provider={provider.provider} model={provider.model}")

    assessments: list[dict] = []
    for _, row in events.iterrows():
        context = {
            "scenario_id": "S5",
            "scenario_name": "负电价深度过剩场景",
            "event_type": str(row["event_type"]),
            "temperature_c": float(row["temperature_c"]),
            "price_yuan_mwh": float(row["price_yuan_mwh"]),
        }
        a = provider.assess_event(str(row["event_text"]), context=context, allow_fallback=False)
        rec = {
            "event_type": row["event_type"],
            "event_text": row["event_text"],
            "ai_risk_score": a.risk_score,
            "ai_price_spike_score": a.price_spike_score,
            "ai_load_pressure_score": a.load_pressure_score,
            "ai_renewable_curtailment_score": a.renewable_curtailment_score,
            "ai_recommended_storage_bias": a.recommended_storage_bias,
            "ai_event_summary": a.event_summary,
            "ai_explanation": a.explanation,
            "ai_provider": a.provider,
            "ai_model": a.model,
        }
        assessments.append(rec)
        print(
            f"  {row['event_type']}: risk={a.risk_score:.2f} price={a.price_spike_score:.2f} "
            f"load={a.load_pressure_score:.2f} renewable={a.renewable_curtailment_score:.2f} "
            f"bias={a.recommended_storage_bias:.2f}  provider={a.provider}"
        )

    feat = pd.DataFrame(assessments)
    cols = [
        "event_type", "event_text", "ai_risk_score", "ai_price_spike_score",
        "ai_load_pressure_score", "ai_renewable_curtailment_score",
        "ai_recommended_storage_bias", "ai_event_summary", "ai_explanation",
        "ai_provider", "ai_model",
    ]
    enriched = df.merge(feat[cols], on=["event_type", "event_text"], how="left")
    enriched.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "s5_event_semantic_features.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for rec in assessments:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved enriched S5: {OUT_CSV}")
    print(f"Saved raw assessments: {raw_path}")
    # Show the unseen-event scores explicitly.
    neg = feat[feat["event_type"] == "负电价消纳"].iloc[0]
    print(f"\nUnseen event '负电价消纳' DeepSeek scores:")
    print(f"  risk={neg['ai_risk_score']:.2f} price_spike={neg['ai_price_spike_score']:.2f} "
          f"load={neg['ai_load_pressure_score']:.2f} renewable={neg['ai_renewable_curtailment_score']:.2f} "
          f"storage_bias={neg['ai_recommended_storage_bias']:.2f}")


if __name__ == "__main__":
    main()
