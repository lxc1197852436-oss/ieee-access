from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSICAL_PATH = ROOT / "outputs" / "chapter6" / "chapter6_scenario_results.csv"
CLASSICAL_FALLBACK_PATH = ROOT / "reports" / "chapter6_scenario_results.csv"
VALIDATION_PATH = ROOT / "outputs" / "chapter6" / "enhanced_rolling_validation.csv"
ENHANCED_SWEEP_PATH = ROOT / "outputs" / "chapter6" / "enhanced_rolling_sweep.csv"
ENHANCED_SWEEP_TAG = "light_40_10"
LONG_PATH = ROOT / "outputs" / "chapter6_long" / "evaluation_aggregate.csv"
IEEE_FIG_DIR = REPO_ROOT / "ieee_pkg" / "ieee_access_vpp_ledrl_20260630" / "figures"
LONG_FALLBACK_PATH = IEEE_FIG_DIR / "evaluation_aggregate.csv"
SWEEP_PATH = ROOT / "outputs" / "chapter6_long" / "prior_weight_sweep_summary.csv"
SWEEP_FALLBACK_PATH = IEEE_FIG_DIR / "prior_weight_sweep_summary.csv"
NUMERIC_GUIDANCE_SWEEP_PATH = ROOT / "outputs" / "chapter6_long" / "sac_numeric_guidance_sweep.csv"
NUMERIC_GUIDANCE_SWEEP_TAG = "numeric_guidance_100"

METRICS = [
    "total_reward_yuan_mean",
    "cvar_5_yuan_mean",
    "battery_throughput_mwh_mean",
    "high_price_discharge_rate_mean",
    "low_price_charge_rate_mean",
]

CORE_ORDER = [
    "LE-DRL-SAC + semantic safety layer (w=0.9)",
    "Rule-Based",
    "Rolling-Horizon",
    "Enhanced Rolling-Horizon",
    "Linear-MPC",
    "SAC-Numeric",
    "SAC-Numeric + numeric safety layer",
    "LE-DRL w/o Text",
    "LE-DRL-SAC",
]

SCENARIO_ORDER = ["S1", "S2", "S3", "S4"]


def normalize_classical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"policy": "model"}).copy()
    for col in [
        "total_reward_yuan",
        "cvar_5_yuan",
        "battery_throughput_mwh",
        "high_price_discharge_rate",
        "low_price_charge_rate",
    ]:
        out[f"{col}_mean"] = out[col]
    return out[
        [
            "scenario_id",
            "scenario_name",
            "model",
            "total_reward_yuan_mean",
            "cvar_5_yuan_mean",
            "battery_throughput_mwh_mean",
            "high_price_discharge_rate_mean",
            "low_price_charge_rate_mean",
        ]
    ]


def load_classical() -> pd.DataFrame:
    path = CLASSICAL_PATH if CLASSICAL_PATH.exists() else CLASSICAL_FALLBACK_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing classical baseline results: {CLASSICAL_PATH} or {CLASSICAL_FALLBACK_PATH}")
    classical = normalize_classical(pd.read_csv(path))
    if ENHANCED_SWEEP_PATH.exists():
        sweep = pd.read_csv(ENHANCED_SWEEP_PATH)
        sweep = sweep[sweep["tag"] == ENHANCED_SWEEP_TAG]
        enhanced = normalize_classical(sweep)
        classical = pd.concat([classical[classical["model"] != "Enhanced Rolling-Horizon"], enhanced], ignore_index=True)
    elif VALIDATION_PATH.exists():
        validation = normalize_classical(pd.read_csv(VALIDATION_PATH))
        enhanced = validation[validation["model"] == "Enhanced Rolling-Horizon"]
        classical = pd.concat([classical[classical["model"] != "Enhanced Rolling-Horizon"], enhanced], ignore_index=True)
    return classical


def load_long_rl() -> pd.DataFrame:
    # Prefer the freshly regenerated long-run aggregate (AI-semantic training)
    # over the stale fallback committed under ieee_pkg/figures.
    path = LONG_PATH if LONG_PATH.exists() else LONG_FALLBACK_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing long RL results: {LONG_PATH} or {LONG_FALLBACK_PATH}")
    long_df = pd.read_csv(path)
    long_df = long_df.rename(columns={"model_": "model"}) if "model_" in long_df.columns else long_df
    return long_df[
        [
            "scenario_id",
            "scenario_name",
            "model",
            "total_reward_yuan_mean",
            "cvar_5_yuan_mean",
            "battery_throughput_mwh_mean",
            "high_price_discharge_rate_mean",
            "low_price_charge_rate_mean",
        ]
    ]


def load_numeric_safety_layer() -> pd.DataFrame:
    if not NUMERIC_GUIDANCE_SWEEP_PATH.exists():
        return pd.DataFrame()
    sweep = pd.read_csv(NUMERIC_GUIDANCE_SWEEP_PATH)
    sweep = sweep[sweep["tag"] == NUMERIC_GUIDANCE_SWEEP_TAG].copy()
    if sweep.empty:
        return pd.DataFrame()
    return normalize_classical(sweep)


def load_safety_layer() -> pd.DataFrame:
    path = SWEEP_PATH if SWEEP_PATH.exists() else SWEEP_FALLBACK_PATH
    if not path.exists():
        return pd.DataFrame()
    sweep = pd.read_csv(path)
    row = sweep[sweep["weight"].round(2) == 0.9]
    if row.empty:
        return pd.DataFrame()
    r = row.iloc[0]
    rows = []
    # Only cross-scenario summary is available for the safety-layer sweep here.
    rows.append(
        {
            "model": "LE-DRL-SAC + semantic safety layer (w=0.9)",
            "total_reward_yuan_mean": float(r["total_reward_yuan_mean"]),
            "cvar_5_yuan_mean": float(r["cvar_5_yuan_mean"]),
            "battery_throughput_mwh_mean": float(r["battery_throughput_mwh_mean"]),
            "high_price_discharge_rate_mean": float(r["high_price_discharge_rate_mean"]),
            "low_price_charge_rate_mean": float(r["low_price_charge_rate_mean"]),
        }
    )
    return pd.DataFrame(rows)


def aggregate_by_model(scenario_rows: pd.DataFrame) -> pd.DataFrame:
    summary = scenario_rows.groupby("model", as_index=False)[METRICS].mean()
    safety = load_safety_layer()
    if not safety.empty:
        summary = pd.concat([summary, safety], ignore_index=True)
    summary["_order"] = summary["model"].map({name: i for i, name in enumerate(CORE_ORDER)}).fillna(999)
    return summary.sort_values(["_order", "total_reward_yuan_mean"], ascending=[True, False]).drop(columns="_order")


def scenario_core_rows(scenario_rows: pd.DataFrame) -> pd.DataFrame:
    rows = scenario_rows.copy()
    rows["rank_in_scenario"] = rows.groupby("scenario_id")["total_reward_yuan_mean"].rank(ascending=False, method="min")
    rows["_scenario_order"] = rows["scenario_id"].map({s: i for i, s in enumerate(SCENARIO_ORDER)}).fillna(999)
    rows["_model_order"] = rows["model"].map({name: i for i, name in enumerate(CORE_ORDER)}).fillna(999)
    return rows.sort_values(["_scenario_order", "_model_order"]).drop(columns=["_scenario_order", "_model_order"])


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def main() -> None:
    IEEE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    scenario_rows = pd.concat([load_classical(), load_long_rl(), load_numeric_safety_layer()], ignore_index=True)
    scenario_rows = scenario_rows[scenario_rows["model"].isin(CORE_ORDER)]

    scenario_table = scenario_core_rows(scenario_rows)
    summary = aggregate_by_model(scenario_rows)

    scenario_table.to_csv(IEEE_FIG_DIR / "combined_scenario_core_baselines.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(IEEE_FIG_DIR / "revised_core_baseline_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(IEEE_FIG_DIR / "combined_baseline_summary.csv", index=False, encoding="utf-8-sig")

    copy_if_exists(SWEEP_PATH, IEEE_FIG_DIR / "prior_weight_sweep_summary.csv")

    print("Saved IEEE baseline tables:")
    print(IEEE_FIG_DIR / "combined_scenario_core_baselines.csv")
    print(IEEE_FIG_DIR / "revised_core_baseline_summary.csv")
    print(IEEE_FIG_DIR / "combined_baseline_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
