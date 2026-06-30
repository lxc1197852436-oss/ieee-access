from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"


PALETTE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "gray": "#5F6B7A",
    "text": "#1F2933",
    "line": "#2F3A45",
    "light_blue": "#EAF4FB",
    "light_green": "#EAF7F1",
    "light_orange": "#FFF4DD",
    "light_red": "#FCEDE8",
    "light_gray": "#F4F6F8",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def box(ax, xy, wh, text, face, edge, weight="normal", size=8.2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.05,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=PALETTE["text"],
        fontsize=size,
        fontweight=weight,
        linespacing=1.23,
    )
    return patch


def arrow(ax, start, end, color=None, rad=0.0, lw=1.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color or PALETTE["line"],
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def add_band_label(ax, x, y, text, color):
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=color,
    )


def draw_main_architecture() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.15, 4.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Language-enhanced VPP dispatch framework",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=PALETTE["text"],
    )

    add_band_label(ax, 0.035, 0.82, "Data and events", PALETTE["blue"])
    add_band_label(ax, 0.035, 0.57, "Semantic layer", PALETTE["red"])
    add_band_label(ax, 0.035, 0.415, "Decision layer", PALETTE["green"])
    add_band_label(ax, 0.035, 0.105, "Evaluation", PALETTE["orange"])

    # Data layer
    public_data = box(
        ax,
        (0.18, 0.75),
        (0.17, 0.115),
        "Public-data\ncalibration\n(load, PV, price)",
        PALETTE["light_blue"],
        PALETTE["blue"],
    )
    sim_env = box(
        ax,
        (0.42, 0.75),
        (0.18, 0.115),
        "15-min VPP\nsimulation\n(SOC constraints)",
        PALETTE["light_blue"],
        PALETTE["blue"],
    )
    text_events = box(
        ax,
        (0.67, 0.75),
        (0.18, 0.115),
        "Operational text\nevents\n(weather, market)",
        PALETTE["light_blue"],
        PALETTE["blue"],
    )

    # Semantic layer
    llm = box(
        ax,
        (0.27, 0.50),
        (0.19, 0.12),
        "LLM / local\nsemantic encoder",
        PALETTE["light_red"],
        PALETTE["red"],
    )
    semantic_features = box(
        ax,
        (0.55, 0.50),
        (0.23, 0.12),
        "Structured risk features\nrisk, price spike,\nload pressure, curtailment",
        PALETTE["light_red"],
        PALETTE["red"],
        size=7.7,
    )

    # Decision layer
    numeric_state = box(
        ax,
        (0.12, 0.26),
        (0.19, 0.12),
        "Numeric state\nload, PV, price,\ntemperature, SOC, time",
        PALETTE["light_green"],
        PALETTE["green"],
        size=7.7,
    )
    augmented_state = box(
        ax,
        (0.38, 0.26),
        (0.17, 0.12),
        "Augmented state\nnumeric + semantic",
        PALETTE["light_green"],
        PALETTE["green"],
    )
    sac = box(
        ax,
        (0.62, 0.26),
        (0.18, 0.12),
        "LE-DRL-SAC\nActor-Critic\nlearned action",
        PALETTE["light_green"],
        PALETTE["green"],
        size=7.7,
    )
    action = box(
        ax,
        (0.84, 0.26),
        (0.12, 0.12),
        "Semantic\nsafety layer\nw = 0.75",
        PALETTE["light_green"],
        PALETTE["green"],
        size=7.7,
    )

    # Evaluation layer
    baselines = box(
        ax,
        (0.17, 0.055),
        (0.19, 0.105),
        "Baselines\nSAC-Numeric\nLE-DRL w/o Text",
        PALETTE["light_orange"],
        PALETTE["orange"],
        size=7.6,
    )
    metrics = box(
        ax,
        (0.45, 0.055),
        (0.19, 0.105),
        "Multi-scenario tests\nS1-S4, 3 seeds\nreward and CVaR",
        PALETTE["light_orange"],
        PALETTE["orange"],
        size=7.6,
    )
    behavior = box(
        ax,
        (0.72, 0.055),
        (0.17, 0.105),
        "Behavior analysis\nhigh-price discharge\nlow-price charge",
        PALETTE["light_orange"],
        PALETTE["orange"],
        size=7.4,
    )
    battery = box(
        ax,
        (0.84, 0.405),
        (0.12, 0.095),
        "Battery\ncharge /\ndischarge",
        PALETTE["light_gray"],
        PALETTE["gray"],
        size=7.5,
    )

    # Arrows
    arrow(ax, (0.35, 0.807), (0.42, 0.807))
    arrow(ax, (0.60, 0.807), (0.67, 0.807))
    arrow(ax, (0.76, 0.75), (0.39, 0.62), rad=0.14, color=PALETTE["red"])
    arrow(ax, (0.46, 0.56), (0.55, 0.56), color=PALETTE["red"])
    arrow(ax, (0.51, 0.75), (0.22, 0.38), rad=0.08, color=PALETTE["blue"])
    arrow(ax, (0.665, 0.50), (0.465, 0.38), rad=0.04, color=PALETTE["red"])
    arrow(ax, (0.31, 0.32), (0.38, 0.32), color=PALETTE["green"])
    arrow(ax, (0.55, 0.32), (0.62, 0.32), color=PALETTE["green"])
    arrow(ax, (0.80, 0.32), (0.84, 0.32), color=PALETTE["green"])
    arrow(ax, (0.90, 0.38), (0.90, 0.405), color=PALETTE["gray"])
    arrow(ax, (0.90, 0.26), (0.54, 0.16), rad=-0.15, color=PALETTE["orange"])
    arrow(ax, (0.36, 0.107), (0.45, 0.107), color=PALETTE["orange"])
    arrow(ax, (0.64, 0.107), (0.72, 0.107), color=PALETTE["orange"])

    ax.text(
        0.5,
        0.005,
        "Boundary: the language model produces risk features only; final dispatch combines the SAC actor with a reproducible semantic safety layer.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["gray"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "ieee_vpp_ledrl_architecture.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "ieee_vpp_ledrl_architecture.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw_main_architecture()
    print(f"Saved architecture figure to {OUT_DIR}")
