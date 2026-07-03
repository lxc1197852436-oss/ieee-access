from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Install matplotlib first: pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
IEEE_DIR = REPO_ROOT / "ieee_pkg" / "ieee_access_vpp_ledrl_20260630"
FIG_DIR = IEEE_DIR / "figures"

SUMMARY_CSV = FIG_DIR / "combined_baseline_summary.csv"
SCENARIO_CSV = FIG_DIR / "combined_scenario_core_baselines.csv"
SAFETY_BY_SEED = ROOT / "outputs" / "chapter6_long" / "prior_weight_sweep_by_seed.csv"
OOD_CSV = ROOT / "outputs" / "chapter6_long" / "ood_evaluation.csv"

COLORS = {
    "proposed": "#0B4F6C",
    "rl": "#2A9D8F",
    "baseline": "#8D99AE",
    "mpc": "#E76F51",
    "text": "#1F2937",
    "grid": "#CBD5E1",
    "light_blue": "#E8F1F8",
    "light_green": "#EAF7F2",
    "light_orange": "#FFF2E5",
    "light_gray": "#F3F4F6",
    "light_red": "#FCEDE8",
}

MODEL_LABELS = {
    "LE-DRL-SAC + semantic safety layer (w=0.9)": "LE-DRL-SAC\n+ semantic layer",
    "Rule-Based": "Rule-Based",
    "SAC-Numeric + numeric safety layer": "SAC-Numeric\n+ numeric layer",
    "LE-DRL-SAC": "LE-DRL-SAC\nactor only",
    "Enhanced Rolling-Horizon": "Enhanced\nRolling-Horizon",
    "Linear-MPC": "Linear-MPC",
    "Rolling-Horizon": "Rolling\nHorizon",
    "SAC-Numeric": "SAC-Numeric",
    "LE-DRL w/o Text": "LE-DRL\nw/o Text",
}

PLOT_ORDER = [
    "LE-DRL-SAC + semantic safety layer (w=0.9)",
    "Rule-Based",
    "SAC-Numeric + numeric safety layer",
    "LE-DRL-SAC",
    "Enhanced Rolling-Horizon",
    "Linear-MPC",
    "Rolling-Horizon",
    "SAC-Numeric",
    "LE-DRL w/o Text",
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.35,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _panel(ax, label, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.012, 0.012),
            0.976,
            0.976,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0.9,
            edgecolor="#94A3B8",
            facecolor="white",
            linestyle="--",
        )
    )
    ax.text(0.04, 0.93, label, ha="left", va="center", fontsize=10.5, fontweight="bold", color=COLORS["proposed"])
    ax.text(0.13, 0.93, title, ha="left", va="center", fontsize=8.8, fontweight="bold", color=COLORS["text"])


def _box(ax, x, y, w, h, title, subtitle, face, edge, title_size=8.4, sub_size=7.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.016",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=face,
        )
    )
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=COLORS["text"])
    ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=sub_size, color="#475569", linespacing=1.25)


def _arrow(ax, x1, y1, x2, y2, color="#334155", rad=0.0, lw=0.9):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def draw_architecture() -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))

    # Panel a: inputs
    ax = axes[0, 0]
    _panel(ax, "a", "Event and numerical inputs")
    _box(ax, 0.10, 0.42, 0.36, 0.26, "Numerical state", "load, PV, price,\nT, SOC, time", COLORS["light_blue"], COLORS["proposed"])
    _box(ax, 0.54, 0.42, 0.36, 0.26, "Operational text", "weather warning,\nmarket notice,\ncurtailment notice", COLORS["light_blue"], COLORS["proposed"])
    ax.text(0.50, 0.16, "public-data-calibrated\n15-min dispatch trajectory", ha="center", va="center", fontsize=7.0, color="#64748B")

    # Panel b: DeepSeek semantic mapping
    ax = axes[0, 1]
    _panel(ax, "b", "DeepSeek semantic mapping")
    _box(ax, 0.10, 0.46, 0.34, 0.24, "DeepSeek encoder", "queried per event\n(OpenAI-compatible)", COLORS["light_orange"], "#B45309")
    _box(ax, 0.55, 0.46, 0.36, 0.24, "Cached risk vector", "[risk, price, load,\ncurtailment, storage bias]", COLORS["light_orange"], "#B45309")
    _arrow(ax, 0.44, 0.58, 0.55, 0.58, color="#B45309")
    ax.text(0.50, 0.18, "LLM maps text to scores only;\nit never emits the dispatch action", ha="center", va="center", fontsize=7.0, color="#B45309", style="italic")

    # Panel c: LE-DRL-SAC decision core
    ax = axes[1, 0]
    _panel(ax, "c", "LE-DRL-SAC decision core")
    _box(ax, 0.08, 0.42, 0.24, 0.24, "Augmented state", "numeric +\nsemantic", COLORS["light_green"], COLORS["rl"])
    _box(ax, 0.38, 0.42, 0.24, 0.24, "SAC actor", "a_SAC", COLORS["light_green"], COLORS["rl"])
    _box(ax, 0.68, 0.42, 0.24, 0.24, "Safety layer", "a=(1-w)a_SAC\n+ w a_sem", COLORS["light_green"], COLORS["rl"])
    _arrow(ax, 0.32, 0.54, 0.38, 0.54, color=COLORS["rl"])
    _arrow(ax, 0.62, 0.54, 0.68, 0.54, color=COLORS["rl"])
    _arrow(ax, 0.80, 0.42, 0.50, 0.16, color="#B45309", rad=-0.18)
    ax.text(0.50, 0.10, "w = 0.9, prior power = 2.0 MW", ha="center", va="center", fontsize=7.0, color=COLORS["rl"])

    # Panel d: feasible dispatch and evaluation
    ax = axes[1, 1]
    _panel(ax, "d", "Feasible dispatch and evaluation")
    _box(ax, 0.10, 0.46, 0.34, 0.24, "Feasibility clip", "SOC and power\nbounds", COLORS["light_gray"], "#64748B")
    _box(ax, 0.54, 0.46, 0.36, 0.24, "Battery dispatch", "charge / discharge\ninto the VPP", COLORS["light_gray"], "#64748B")
    _arrow(ax, 0.44, 0.58, 0.54, 0.58, color="#64748B")
    ax.text(0.50, 0.16, "reward, CVaR, throughput,\nOOD on real weather", ha="center", va="center", fontsize=7.0, color="#64748B")

    fig.suptitle("Language-enhanced VPP dispatch with an auditable semantic safety layer", fontsize=11.2, fontweight="bold", color=COLORS["text"], y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "ieee_vpp_ledrl_architecture")


def proposed_scenario_rows() -> pd.DataFrame:
    if not SAFETY_BY_SEED.exists():
        return pd.DataFrame()
    df = pd.read_csv(SAFETY_BY_SEED)
    df = df[df["weight"].round(2) == 0.9]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("scenario", as_index=False).agg(
        total_reward_yuan_mean=("total_reward_yuan", "mean"),
        total_reward_yuan_std=("total_reward_yuan", "std"),
        cvar_5_yuan_mean=("cvar_5_yuan", "mean"),
        battery_throughput_mwh_mean=("battery_throughput_mwh", "mean"),
        high_price_discharge_rate_mean=("high_price_discharge_rate", "mean"),
        low_price_charge_rate_mean=("low_price_charge_rate", "mean"),
    )
    g["scenario_id"] = g["scenario"]
    g["scenario_name"] = g["scenario"].map({"S1": "常规夏季运行场景", "S2": "高温负荷压力场景", "S3": "价格尖峰场景", "S4": "新能源消纳压力场景"})
    g["model"] = "LE-DRL-SAC + semantic safety layer (w=0.9)"
    return g.drop(columns=["scenario"])


def load_scenario_table() -> pd.DataFrame:
    base = pd.read_csv(SCENARIO_CSV)
    prop = proposed_scenario_rows()
    if not prop.empty:
        base = pd.concat([base, prop], ignore_index=True, sort=False)
    return base


def plot_baseline_reward_error() -> None:
    setup_style()
    scenario_df = load_scenario_table()
    rows = []
    for model in PLOT_ORDER:
        sub = scenario_df[scenario_df["model"] == model]
        if sub.empty:
            continue
        rows.append(
            {
                "model": model,
                "mean": float(sub["total_reward_yuan_mean"].mean()),
                "std": float(sub["total_reward_yuan_mean"].std(ddof=1)) if len(sub) > 1 else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    colors = [COLORS["proposed"] if "semantic safety" in m else COLORS["mpc"] if "MPC" in m else COLORS["baseline"] for m in df["model"]]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(df))
    ax.bar(x, df["mean"], yerr=df["std"], capsize=3, color=colors, edgecolor="#334155", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in df["model"]], rotation=30, ha="right")
    ax.set_ylabel("Average total reward (yuan)\nmean ± scenario std")
    ax.set_title("Cross-scenario reward comparison with variability")
    ax.axhline(df.loc[df["model"] == "Rule-Based", "mean"].iloc[0], color="#111827", linestyle="--", linewidth=0.9, label="Rule-Based")
    ax.legend(loc="lower right")
    save(fig, "baseline_reward_with_error")


def plot_reward_throughput_tradeoff() -> None:
    setup_style()
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["model"].isin(PLOT_ORDER)]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for _, r in df.iterrows():
        model = r["model"]
        color = COLORS["proposed"] if "semantic safety" in model else COLORS["mpc"] if "MPC" in model else COLORS["baseline"]
        ax.scatter(r["battery_throughput_mwh_mean"], r["total_reward_yuan_mean"], s=55, color=color, edgecolor="#334155", linewidth=0.6)
        ax.annotate(MODEL_LABELS.get(model, model).replace("\n", " "), (r["battery_throughput_mwh_mean"], r["total_reward_yuan_mean"]), xytext=(4, 3), textcoords="offset points", fontsize=7.0)
    ax.set_xlabel("Battery throughput (MWh)")
    ax.set_ylabel("Average total reward (yuan)")
    ax.set_title("Reward--throughput trade-off")
    save(fig, "reward_throughput_tradeoff")


def plot_ood_reward() -> None:
    if not OOD_CSV.exists():
        return
    setup_style()
    df = pd.read_csv(OOD_CSV)
    summary = df.groupby("policy", as_index=False).agg(reward=("total_reward_yuan", "mean"), std=("total_reward_yuan", "std"), throughput=("battery_throughput_mwh", "mean"))
    order = ["LE-DRL-SAC + semantic safety layer", "SAC-Numeric + numeric safety layer", "Rule-Based", "Enhanced Rolling-Horizon", "Rolling-Horizon"]
    summary = summary.set_index("policy").reindex([x for x in order if x in summary["policy"].values]).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(summary))
    colors = [COLORS["proposed"] if "LE-DRL" in p else COLORS["baseline"] for p in summary["policy"]]
    ax.bar(x, summary["reward"], yerr=summary["std"].fillna(0), capsize=3, color=colors, edgecolor="#334155", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace(" + ", "\n+ ") for p in summary["policy"]], rotation=20, ha="right")
    ax.set_ylabel("OOD average total reward (yuan)")
    ax.set_title("OOD evaluation on real Guangzhou weather")
    save(fig, "ood_real_weather_reward")


def main() -> None:
    draw_architecture()
    plot_baseline_reward_error()
    plot_reward_throughput_tradeoff()
    plot_ood_reward()
    print(f"Saved IEEE figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
