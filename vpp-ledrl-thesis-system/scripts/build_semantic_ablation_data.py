"""Build keyword-encoder and noisy-score ablation datasets for the M5 ablation.

This script mirrors the DeepSeek semantic-feature build (scripts/build_ai_semantic_features.py,
scripts/score_s5_with_deepseek.py, scripts/score_s5_variants.py) but replaces the LLM-derived
scores with two ablation sources, so the M5 ablation (DeepSeek vs. keyword vs. noisy) can be
run by pointing the training scripts at the ablation CSVs instead of the DeepSeek CSVs.

Two ablation sources are produced, both writing the SAME column schema as the DeepSeek files
so VPPEnv._semantic_signal consumes them unchanged (environment.py:50-69):

  1. keyword  -- LocalSemanticEncoder (app/core/semantic.py), a deterministic Chinese-keyword
                 scorer that already exists as the DeepSeek fallback. This isolates whether the
                 DeepSeek gain is unique to LLM understanding or is matched by structured text
                 features obtainable from a keyword encoder.

  2. noisy    -- DeepSeek scores perturbed by clipped Gaussian noise (sigma=0.20 on the four
                 risk dims, sigma=0.30 on storage_bias). This isolates whether arbitrary
                 structured 5-dim perturbation of the scores still helps, i.e. whether the
                 semantic *content* rather than the *dimensionality* carries the value.

Coverage (mirrors every DeepSeek-scored file the training/eval scripts read):
  - chapter6_ai_semantic_scenarios.csv  (S1-S4 training data)
  - s5_negative_price_surplus_ai_semantic.csv  (S5 training data)
  - s5_real_price_week_ai_semantic.csv  (DE-LU real-price validation)
  - s5_variants/v1..v4_ai_semantic.csv  (held-out negative-price variants)

The DeepSeek source files are never overwritten; ablation outputs are written to
sibling files with a `_keyword` / `_noisy` suffix so the originals stay reproducible.

Reproducibility: keyword is fully deterministic. noisy uses a fixed seed (--seed, default 2026)
so the perturbation is reproducible across runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.semantic import LocalSemanticEncoder  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RAW_CACHE = ROOT / "data" / "raw_sources" / "ai_semantic_cache"

# (deepseek input file, ablation output suffix, event-dedup key columns)
# Each entry is one "scenario file" that the training/eval scripts read.
TARGETS = [
    ("chapter6_ai_semantic_scenarios.csv", "chapter6_ai_semantic_scenarios"),
    ("s5_negative_price_surplus_ai_semantic.csv", "s5_negative_price_surplus_ai_semantic"),
    ("s5_real_price_week_ai_semantic.csv", "s5_real_price_week_ai_semantic"),
    ("s5_real_price_multiweek_ai_semantic.csv", "s5_real_price_multiweek_ai_semantic"),
    ("s5_variants/v1_ai_semantic.csv", "s5_variants/v1_ai_semantic"),
    ("s5_variants/v2_ai_semantic.csv", "s5_variants/v2_ai_semantic"),
    ("s5_variants/v3_ai_semantic.csv", "s5_variants/v3_ai_semantic"),
    ("s5_variants/v4_ai_semantic.csv", "s5_variants/v4_ai_semantic"),
    ("s7_export_curtailed_ai_semantic.csv", "s7_export_curtailed_ai_semantic"),
]

# DeepSeek score columns (must match environment._semantic_signal ai_cols + storage_bias).
SCORE_COLS = [
    "ai_risk_score",
    "ai_price_spike_score",
    "ai_load_pressure_score",
    "ai_renewable_curtailment_score",
    "ai_recommended_storage_bias",
]
META_COLS = ["ai_event_summary", "ai_explanation", "ai_provider", "ai_model"]

# Per-column noise sigma. Risk dims are in [0,1]; storage_bias in [-1,1].
# A sigma of 0.20 on a [0,1] score is "moderate" -- enough to scramble the
# ordering between similar events but not enough to make high-risk vs. no-risk
# indistinguishable. The point is perturbation, not destruction.
NOISE_SIGMA = {
    "ai_risk_score": 0.20,
    "ai_price_spike_score": 0.20,
    "ai_load_pressure_score": 0.20,
    "ai_renewable_curtailment_score": 0.20,
    "ai_recommended_storage_bias": 0.30,
}


def score_row_with_keyword(encoder: LocalSemanticEncoder, row: pd.Series) -> dict:
    """Score one deduplicated event with the local keyword encoder."""
    signal = encoder.encode(str(row["event_text"]))
    # Mirror LLMProvider._assess_event_local storage_bias derivation so the
    # keyword source uses the SAME mapping the fallback path already uses.
    bias = signal.renewable_curtailment_score - signal.price_spike_score
    if signal.load_pressure_score > 0.4:
        bias -= 0.2
    bias = float(max(-1.0, min(1.0, bias)))
    return {
        "ai_risk_score": signal.risk_score,
        "ai_price_spike_score": signal.price_spike_score,
        "ai_load_pressure_score": signal.load_pressure_score,
        "ai_renewable_curtailment_score": signal.renewable_curtailment_score,
        "ai_recommended_storage_bias": bias,
        "ai_event_summary": f"keyword: {signal.explanation_hint}",
        "ai_explanation": f"LocalSemanticEncoder keyword score: {signal.explanation_hint}",
        "ai_provider": "keyword",
        "ai_model": "local-keyword",
    }


def perturb_score(value: float, sigma: float, rng: np.random.Generator, low: float, high: float) -> float:
    noise = rng.normal(0.0, sigma)
    return float(max(low, min(high, value + noise)))


def score_events_keyword(encoder: LocalSemanticEncoder, events: pd.DataFrame) -> pd.DataFrame:
    """Apply keyword encoder to deduplicated event rows. Returns one row per event."""
    rows = []
    for _, row in events.iterrows():
        rows.append(score_row_with_keyword(encoder, row))
    return pd.DataFrame(rows)


def score_events_noisy(events: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Perturb DeepSeek scores with clipped Gaussian noise. Returns one row per event."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in events.iterrows():
        rec = {
            "ai_risk_score": perturb_score(float(row["ai_risk_score"]), NOISE_SIGMA["ai_risk_score"], rng, 0.0, 1.0),
            "ai_price_spike_score": perturb_score(
                float(row["ai_price_spike_score"]), NOISE_SIGMA["ai_price_spike_score"], rng, 0.0, 1.0
            ),
            "ai_load_pressure_score": perturb_score(
                float(row["ai_load_pressure_score"]), NOISE_SIGMA["ai_load_pressure_score"], rng, 0.0, 1.0
            ),
            "ai_renewable_curtailment_score": perturb_score(
                float(row["ai_renewable_curtailment_score"]),
                NOISE_SIGMA["ai_renewable_curtailment_score"],
                rng,
                0.0,
                1.0,
            ),
            "ai_recommended_storage_bias": perturb_score(
                float(row["ai_recommended_storage_bias"]),
                NOISE_SIGMA["ai_recommended_storage_bias"],
                rng,
                -1.0,
                1.0,
            ),
            "ai_event_summary": f"noisy(sigma={NOISE_SIGMA['ai_risk_score']}): {row.get('ai_event_summary', '')}",
            "ai_explanation": "DeepSeek score + clipped Gaussian noise (M5 noisy ablation)",
            "ai_provider": "noisy",
            "ai_model": "deepseek-chat+gauss",
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def build_one(src_path: Path, out_stem: str, encoder: LocalSemanticEncoder, seed: int) -> dict:
    """Read a DeepSeek-scored file, produce keyword + noisy ablation files, return a summary."""
    if not src_path.exists():
        return {"file": str(src_path), "status": "MISSING (skipped)"}

    df = pd.read_csv(src_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    has_deepseek = all(col in df.columns for col in SCORE_COLS)
    if not has_deepseek:
        return {"file": str(src_path), "status": "NO_DEEPSEEK_COLS (skipped)"}

    # Deduplicate by (event_type, event_text) -- the same key the DeepSeek build uses.
    event_key = ["event_type", "event_text"]
    events = df[event_key].drop_duplicates(subset=event_key).reset_index(drop=True)

    # Keyword scores are computed from event_text alone; attach DeepSeek context
    # columns only so the keyword scorer signature matches (it ignores them).
    kw_feat = score_events_keyword(encoder, events)
    kw_feat = pd.concat([events, kw_feat], axis=1)

    # Noisy scores are computed by perturbing the DeepSeek scores already in df.
    # Pull the DeepSeek scores for the deduplicated events.
    deepseek_scores = df[event_key + SCORE_COLS].drop_duplicates(subset=event_key).reset_index(drop=True)
    noisy_feat = score_events_noisy(deepseek_scores, seed=seed)
    noisy_feat = pd.concat([events, noisy_feat], axis=1)

    # Replace the score + meta columns in df with each ablation source, write out.
    drop_cols = [c for c in SCORE_COLS + META_COLS if c in df.columns]

    df_kw = df.drop(columns=drop_cols).merge(kw_feat, on=event_key, how="left")
    df_noisy = df.drop(columns=drop_cols).merge(noisy_feat, on=event_key, how="left")

    kw_path = PROCESSED / f"{out_stem}_keyword.csv"
    noisy_path = PROCESSED / f"{out_stem}_noisy.csv"
    df_kw.to_csv(kw_path, index=False, encoding="utf-8-sig")
    df_noisy.to_csv(noisy_path, index=False, encoding="utf-8-sig")

    return {
        "file": str(src_path),
        "keyword_out": str(kw_path),
        "noisy_out": str(noisy_path),
        "n_rows": len(df),
        "n_events": len(events),
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build keyword / noisy ablation datasets for the M5 ablation.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for the noisy-score perturbation.")
    args = parser.parse_args()

    encoder = LocalSemanticEncoder()
    summary = []
    for src_name, out_stem in TARGETS:
        src_path = PROCESSED / src_name
        info = build_one(src_path, out_stem, encoder, seed=args.seed)
        summary.append(info)
        status = info.get("status", "?")
        n = info.get("n_rows", "-")
        ne = info.get("n_events", "-")
        print(f"  [{status}] {src_name}  rows={n} events={ne}")

    # Persist a manifest so the training scripts and the paper can reference exact paths.
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "noise_sigma": NOISE_SIGMA,
        "targets": summary,
        "keyword_source": "app.core.semantic.LocalSemanticEncoder",
        "noisy_source": "deepseek-chat scores + clipped Gaussian noise",
    }
    manifest_path = RAW_CACHE / "m5_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print("Done. Ablation CSVs written next to the DeepSeek CSVs with _keyword / _noisy suffixes.")


if __name__ == "__main__":
    main()
