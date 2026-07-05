"""Clean SCI-style architecture figure for LE-DRL-SAC with a learned event-coverage gate.

Design goals (vs. the previous version):
  * four explicit swim lanes (a) Inputs (b) LLM advisor (c) Gated MoE SAC (d) Output
  * horizontal main flow, orthogonal (Manhattan) connectors, no crossing diagonals
  * each SAC expert paired with its action label on the SAME row (action carried on the arrow)
  * a single reserved accent colour for the learned gate; encoder recoloured to avoid clash
  * an explicit colour legend (blue = prior/known lane, orange = free/unseen lane)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"

C = {
    "text": "#1F2933",
    "line": "#3A4551",
    # lane fills (very light)
    "lane_a": "#F5F7FA",
    "lane_b": "#F7F3FB",
    "lane_c": "#F3F8F4",
    "lane_d": "#FAF6F0",
    # box strokes / fills
    "in_edge": "#2F6F3E", "in_face": "#EAF4EE",      # inputs (green)
    "llm_edge": "#8A5A2B", "llm_face": "#F6EEDF",     # encoder (warm brown, distinct from gate)
    "prior_edge": "#0F6FB5", "prior_face": "#E7F1FA",  # prior lane (blue)
    "free_edge": "#D07A16", "free_face": "#FDF1DE",    # free lane (orange)
    "gate_edge": "#7A3EA6", "gate_face": "#F0E7F8",    # gate (reserved purple)
    "out_edge": "#B23A2E", "out_face": "#FBEDE9",      # output (red)
    "neutral_edge": "#5F6B7A", "neutral_face": "#F2F4F6",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def lane(ax, x0, x1, face):
    ax.add_patch(Rectangle((x0, 0.055), x1 - x0, 0.80, facecolor=face,
                           edgecolor="none", zorder=0))


def box(ax, cx, cy, w, h, text, face, edge, weight="normal", size=8.0):
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.014",
                                linewidth=1.15, edgecolor=edge, facecolor=face,
                                zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", color=C["text"],
            fontsize=size, fontweight=weight, linespacing=1.28, zorder=4)


def straight(ax, start, end, color=None, lw=1.15, label=None, loff=(0, 0.028)):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color or C["line"],
                                 shrinkA=1, shrinkB=1, zorder=2))
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx + loff[0], my + loff[1], label, ha="center", va="center",
                fontsize=7.6, fontstyle="italic", color=color or C["line"], zorder=4)


def ortho(ax, start, end, color=None, lw=1.15, angleA=0, angleB=90, rad=9,
          label=None, loff=(0, 0.028)):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11, linewidth=lw,
        color=color or C["line"], shrinkA=1, shrinkB=1, zorder=2,
        connectionstyle=f"angle,angleA={angleA},angleB={angleB},rad={rad}"))
    if label:
        ax.text(end[0] + loff[0], end[1] + loff[1], label, ha="center", va="center",
                fontsize=7.6, fontstyle="italic", color=color or C["line"], zorder=4)


def header(ax, cx, text, color):
    ax.text(cx, 0.860, text, ha="center", va="center", fontsize=8.2,
            fontweight="bold", color=color)


def draw() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.960, "Language-Enhanced VPP Dispatch with a Learned Event-Coverage Gate",
            ha="center", va="center", fontsize=10.0, fontweight="bold", color=C["text"])

    # swim lanes
    lane(ax, 0.015, 0.190, C["lane_a"])
    lane(ax, 0.200, 0.375, C["lane_b"])
    lane(ax, 0.385, 0.790, C["lane_c"])
    lane(ax, 0.800, 0.985, C["lane_d"])
    header(ax, 0.102, "(a) Inputs", C["in_edge"])
    header(ax, 0.288, "(b) LLM advisor", C["llm_edge"])
    header(ax, 0.588, "(c) Gated mixture-of-experts SAC", C["gate_edge"])
    header(ax, 0.892, "(d) Output & evaluation", C["out_edge"])

    # (a) inputs
    box(ax, 0.102, 0.640, 0.150, 0.150,
        "Operational\ntext events\n(weather, market,\ncurtailment)",
        C["in_face"], C["in_edge"], size=7.2)
    box(ax, 0.102, 0.320, 0.150, 0.150,
        "Numerical\nVPP state\n(load, PV, price,\nSOC, time)",
        C["in_face"], C["in_edge"], size=7.2)

    # (b) LLM advisor
    box(ax, 0.288, 0.640, 0.150, 0.150,
        "DeepSeek\nsemantic encoder\n(cached, emits\nno action)",
        C["llm_face"], C["llm_edge"], weight="bold", size=7.2)
    box(ax, 0.288, 0.360, 0.150, 0.120,
        "Semantic risk\nvector $s^{sem}$\n(5-dim, [0,1])",
        C["llm_face"], C["llm_edge"], size=7.2)

    # (c) gated MoE
    box(ax, 0.450, 0.480, 0.095, 0.170,
        "Augmented\nstate\n$s^{aug}\\!=$\n$[s^{num}\\!,s^{sem}]$",
        C["neutral_face"], C["neutral_edge"], size=7.0)
    box(ax, 0.600, 0.660, 0.150, 0.115,
        "SAC-prior expert\n(regularized;\nstrong on known)",
        C["prior_face"], C["prior_edge"], weight="bold", size=7.0)
    box(ax, 0.600, 0.300, 0.150, 0.115,
        "SAC-free expert\n(no regularizer;\nadaptive on unseen)",
        C["free_face"], C["free_edge"], weight="bold", size=7.0)
    box(ax, 0.588, 0.480, 0.150, 0.090,
        "Learned gate\n$g_\\psi(s)\\!\\to\\! w$",
        C["gate_face"], C["gate_edge"], weight="bold", size=7.0)
    box(ax, 0.740, 0.480, 0.078, 0.200,
        "Blend\n$a=(1\\!-\\!w)a^{free}$\n$+\\,w\\,a^{prior}$",
        C["neutral_face"], C["neutral_edge"], size=6.6)

    # (d) output
    box(ax, 0.892, 0.660, 0.160, 0.120,
        "Feasibility clip\n(battery power,\nSOC limits)",
        C["out_face"], C["out_edge"], size=7.2)
    box(ax, 0.892, 0.470, 0.160, 0.095,
        "Final dispatch\naction $a^{final}$",
        C["out_face"], C["out_edge"], weight="bold", size=7.2)
    box(ax, 0.892, 0.255, 0.160, 0.140,
        "Evaluation\nreward, CVaR,\nthroughput,\nOOD real weather,\nunseen-event",
        C["neutral_face"], C["neutral_edge"], size=7.0)

    # arrows -------------------------------------------------------
    # inputs -> advisor
    straight(ax, (0.177, 0.640), (0.213, 0.640), C["in_edge"])           # text -> encoder
    straight(ax, (0.288, 0.565), (0.288, 0.420), C["llm_edge"])          # encoder -> s^sem
    # into augmented state (orthogonal, no diagonals)
    ortho(ax, (0.363, 0.360), (0.4025, 0.450), C["llm_edge"],
          angleA=0, angleB=90, rad=6)                                    # s^sem -> aug
    ortho(ax, (0.177, 0.320), (0.4025, 0.450), C["in_edge"],
          angleA=0, angleB=-90, rad=6)                                   # num state -> aug
    # augmented -> experts and gate
    ortho(ax, (0.4975, 0.525), (0.525, 0.660), C["neutral_edge"],
          angleA=0, angleB=90, rad=6)                                    # aug -> prior
    ortho(ax, (0.4975, 0.435), (0.525, 0.300), C["neutral_edge"],
          angleA=0, angleB=-90, rad=6)                                   # aug -> free
    straight(ax, (0.4975, 0.480), (0.513, 0.480), C["neutral_edge"])     # aug -> gate
    # experts + gate -> blend (orthogonal); action labels placed clear of the box
    ortho(ax, (0.675, 0.660), (0.701, 0.570), C["prior_edge"],
          angleA=0, angleB=90, rad=5, label="$a^{prior}$", loff=(-0.030, 0.030))
    ortho(ax, (0.675, 0.300), (0.701, 0.390), C["free_edge"],
          angleA=0, angleB=-90, rad=5, label="$a^{free}$", loff=(-0.030, -0.030))
    straight(ax, (0.663, 0.480), (0.701, 0.480), C["gate_edge"],
             label="$w$", loff=(0.0, 0.030))
    # blend -> output
    ortho(ax, (0.779, 0.545), (0.812, 0.660), C["out_edge"],
          angleA=0, angleB=90, rad=5)                                    # blend -> clip
    straight(ax, (0.892, 0.600), (0.892, 0.518), C["out_edge"])          # clip -> final
    straight(ax, (0.892, 0.422), (0.892, 0.325), C["neutral_edge"])      # final -> eval

    # gate-semantics caption + legend
    ax.text(0.5, 0.028,
            "Gate semantics:  high $w$ on known events (defer to prior)   /   "
            "low $w$ on unseen events (defer to free actor)",
            ha="center", va="center", fontsize=7.6, fontstyle="italic",
            color=C["gate_edge"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "ieee_vpp_ledrl_architecture.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "ieee_vpp_ledrl_architecture.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw()
    print(f"Saved architecture figure to {OUT_DIR}")
