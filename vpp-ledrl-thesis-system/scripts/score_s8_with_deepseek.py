"""Score the S8 frequency-reserve event with DeepSeek and produce an AI-semantic S8 dataset.

Mirrors scripts/score_s7_with_deepseek.py. The DeepSeek assessment is cached to a
JSONL record for reproducibility, and merged onto the S8 scenario CSV so VPPEnv
reads the ai_* columns directly.

CRITICAL VALIDATION: DeepSeek must recognize the "调频预留" (frequency reserve)
event and return renewable=0, storage_bias=0 (no charge bias, keep mid-band) --
the OPPOSITE of the keyword encoder, which returns all-zero (text avoids the
"光伏/消纳" tokens) and thus fails to steer the actor away from filling up.
If DeepSeek also returns a charge bias, the S8 ablation is moot.
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

S8_CSV = ROOT / "data" / "processed" / "s8_freq_reserve.csv"
OUT_CSV = ROOT / "data" / "processed" / "s8_freq_reserve_ai_semantic.csv"
RAW_DIR = ROOT / "data" / "raw_sources" / "ai_semantic_cache"


def main() -> None:
    df = pd.read_csv(S8_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Pick representative rows per event (non-midday normal for Status: Normal).
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["_hour"] = hour
    rep_rows = []
    for (et, et_text), grp in df.groupby(["event_type", "event_text"]):
        if et == "调频预留需求":
            rep = grp.iloc[0]
        else:
            sub = grp[(grp["_hour"] < 10) | (grp["_hour"] > 15)]
            sub = sub[sub["price_yuan_mwh"] > 300] if (sub["price_yuan_mwh"] > 300).any() else sub
            rep = sub.iloc[0] if len(sub) else grp.iloc[0]
        rep_rows.append(rep)
    events = pd.DataFrame(rep_rows)[["event_type", "event_text", "temperature_c", "price_yuan_mwh"]].reset_index(drop=True)

    provider = LLMProvider()
    print(f"provider={provider.provider} model={provider.model}")
    if provider.provider == "local" or not provider.api_key:
        print("WARNING: DeepSeek API not configured."); return

    # Normal-event scores reused from S1 cache (see score_s7_with_deepseek.py for
    # rationale: prevents scenario_name leakage into Status: Normal scoring).
    NORMAL_SCORES = {
        "ai_risk_score": 0.2, "ai_price_spike_score": 0.1,
        "ai_load_pressure_score": 0.3, "ai_renewable_curtailment_score": 0.2,
        "ai_recommended_storage_bias": 0.0,
        "ai_event_summary": "正常运行，无显著事件。",
        "ai_explanation": "DeepSeek S1-cached score for 'Status: Normal' (reused to avoid scenario_name leakage).",
        "ai_provider": "openai_compatible", "ai_model": "deepseek-chat",
    }
    assessments: list[dict] = []
    for _, row in events.iterrows():
        context = {
            "scenario_id": "S8",
            "scenario_name": "调频预留需求场景",
            "event_type": str(row["event_type"]),
            "temperature_c": float(row["temperature_c"]),
            "price_yuan_mwh": float(row["price_yuan_mwh"]),
        }
        if str(row["event_text"]).strip() == "Status: Normal":
            rec = {"event_type": row["event_type"], "event_text": row["event_text"], **NORMAL_SCORES}
        else:
            a = provider.assess_event(str(row["event_text"]), context=context, allow_fallback=False)
            rec = {
                "event_type": row["event_type"], "event_text": row["event_text"],
                "ai_risk_score": a.risk_score, "ai_price_spike_score": a.price_spike_score,
                "ai_load_pressure_score": a.load_pressure_score,
                "ai_renewable_curtailment_score": a.renewable_curtailment_score,
                "ai_recommended_storage_bias": a.recommended_storage_bias,
                "ai_event_summary": a.event_summary, "ai_explanation": a.explanation,
                "ai_provider": a.provider, "ai_model": a.model,
            }
        assessments.append(rec)
        print(
            f"  {row['event_type']}: risk={rec['ai_risk_score']:.2f} price={rec['ai_price_spike_score']:.2f} "
            f"load={rec['ai_load_pressure_score']:.2f} renewable={rec['ai_renewable_curtailment_score']:.2f} "
            f"bias={rec['ai_recommended_storage_bias']:.2f}  provider={rec['ai_provider']}"
        )

    feat = pd.DataFrame(assessments)
    cols = ["event_type", "event_text", "ai_risk_score", "ai_price_spike_score",
            "ai_load_pressure_score", "ai_renewable_curtailment_score",
            "ai_recommended_storage_bias", "ai_event_summary", "ai_explanation",
            "ai_provider", "ai_model"]
    enriched = df.drop(columns=["_hour"]).merge(feat[cols], on=["event_type", "event_text"], how="left")
    enriched.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "s8_event_semantic_features.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for rec in assessments:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved enriched S8: {OUT_CSV}")
    print(f"Saved raw assessments: {raw_path}")

    s8_ev = feat[feat["event_type"] == "调频预留需求"]
    if len(s8_ev):
        r = s8_ev.iloc[0]
        print(f"\n=== VALIDATION: S8 '调频预留需求' DeepSeek scores ===")
        print(f"  renewable={r['ai_renewable_curtailment_score']:.2f}  (expect ~0, NOT a charge signal)")
        print(f"  storage_bias={r['ai_recommended_storage_bias']:.2f}  (expect ~0, keep mid-band)")
        ok = (r['ai_renewable_curtailment_score'] < 0.3 and
              abs(r['ai_recommended_storage_bias']) < 0.3)
        print(f"  >>> {'PASS' if ok else 'FAIL'}: DeepSeek recognized reserve request" +
              ("" if ok else " -- check event_text"))


if __name__ == "__main__":
    main()
