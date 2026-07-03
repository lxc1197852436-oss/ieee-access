"""Enrich the OOD real-weather scenarios with cached DeepSeek semantic scores.

The OOD scenarios reuse the same event templates as the training set, so the
DeepSeek scores are already cached in ai_event_semantic_features.csv. This
script merges those scores onto the OOD data so that LE-DRL-SAC, trained with
DeepSeek semantics, can be evaluated on the OOD set without re-querying the API
and with a consistent semantic distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OOD_PATH = ROOT / "data" / "processed" / "ood_vpp_scenarios.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "ai_event_semantic_features.csv"
OUT_PATH = ROOT / "data" / "processed" / "ood_vpp_scenarios_ai_semantic.csv"


def main() -> None:
    ood = pd.read_csv(OOD_PATH)
    features = pd.read_csv(FEATURES_PATH)
    keep = [
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
    enriched = ood.merge(features[keep], on=["event_type", "event_text"], how="left")
    missing = enriched[enriched["ai_risk_score"].isna()]
    if not missing.empty:
        print(f"WARN: {len(missing)} rows have no cached AI scores; falling back to local encoder for them.")
    enriched.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Rows: {len(enriched)}  AI-scored rows: {enriched['ai_risk_score'].notna().sum()}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
