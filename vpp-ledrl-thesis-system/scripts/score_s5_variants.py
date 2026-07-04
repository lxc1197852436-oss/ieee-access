"""Score S5 variant events with DeepSeek and add AI semantic columns.

The event text is identical to the S5 template, but the context (temperature,
price) differs per variant, so DeepSeek is queried once per (variant, event)
to produce variant-specific scores. Output schema matches
s5_negative_price_surplus_ai_semantic.csv so VPPEnv reads the ai_* columns.
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

IN_DIR = ROOT / "data" / "processed" / "s5_variants"
OUT_DIR = ROOT / "data" / "processed" / "s5_variants"
RAW_DIR = ROOT / "data" / "raw_sources" / "ai_semantic_cache"


def main() -> None:
    provider = LLMProvider()
    print(f"provider={provider.provider} model={provider.model}")
    all_raw = []

    for vid in ["V1", "V2", "V3", "V4"]:
        csv = IN_DIR / f"{vid.lower()}.csv"
        df = pd.read_csv(csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        events = df[["event_type", "event_text", "temperature_c", "price_yuan_mwh"]].drop_duplicates(
            subset=["event_type", "event_text"]
        ).reset_index(drop=True)

        assessments = []
        for _, row in events.iterrows():
            ctx = {"scenario_id": vid, "scenario_name": str(row["event_type"]),
                   "event_type": str(row["event_type"]),
                   "temperature_c": float(row["temperature_c"]),
                   "price_yuan_mwh": float(row["price_yuan_mwh"])}
            a = provider.assess_event(str(row["event_text"]), context=ctx, allow_fallback=False)
            rec = {"event_type": row["event_type"], "event_text": row["event_text"],
                   "ai_risk_score": a.risk_score, "ai_price_spike_score": a.price_spike_score,
                   "ai_load_pressure_score": a.load_pressure_score,
                   "ai_renewable_curtailment_score": a.renewable_curtailment_score,
                   "ai_recommended_storage_bias": a.recommended_storage_bias,
                   "ai_event_summary": a.event_summary, "ai_explanation": a.explanation,
                   "ai_provider": a.provider, "ai_model": a.model}
            assessments.append(rec)
            all_raw.append({"variant": vid, "context": ctx, "assessment": rec})

        feat = pd.DataFrame(assessments)
        cols = ["event_type", "event_text", "ai_risk_score", "ai_price_spike_score",
                "ai_load_pressure_score", "ai_renewable_curtailment_score",
                "ai_recommended_storage_bias", "ai_event_summary", "ai_explanation",
                "ai_provider", "ai_model"]
        enriched = df.merge(feat[cols], on=["event_type", "event_text"], how="left")
        out = OUT_DIR / f"{vid.lower()}_ai_semantic.csv"
        enriched.to_csv(out, index=False, encoding="utf-8-sig")
        neg = feat[feat["event_type"] == "负电价消纳"].iloc[0]
        print(f"{vid}: risk={neg['ai_risk_score']:.2f} price={neg['ai_price_spike_score']:.2f} "
              f"load={neg['ai_load_pressure_score']:.2f} renewable={neg['ai_renewable_curtailment_score']:.2f} "
              f"bias={neg['ai_recommended_storage_bias']:.2f}  -> {out.name}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with (RAW_DIR / "s5_variants_semantic.jsonl").open("w", encoding="utf-8") as f:
        for r in all_raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
