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
    ax.text(cx, 0.885, text, ha="center", va="center", fontsize=9.0,
            fontweight="bold", color=color)


def draw() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.16, 3.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Language-Enhanced VPP Dispatch with a Learned Event-Coverage Gate",
            ha="center", va="center", fontsize=11.0, fontweight="bold", color=C["text"])

    # swim lanes
    lane(ax, 0.015, 0.205, C["lane_a"])
    lane(ax, 0.215, 0.410, C["lane_b"])
    lane(ax, 0.420, 0.775, C["lane_c"])
    lane(ax, 0.785, 0.985, C["lane_d"])
    header(ax, 0.110, "(a) Inputs", C["in_edge"])
    header(ax, 0.312, "(b) LLM advisor", C["llm_edge"])
    header(ax, 0.597, "(c) Gated mixture-of-experts SAC", C["gate_edge"])
    header(ax, 0.885, "(d) Output & evaluation", C["out_edge"])

    # (a) inputs
    box(ax, 0.110, 0.660, 0.165, 0.135,
        "Operational\ntext events\n(weather, market,\ncurtailment)",
        C["in_face"], C["in_edge"], size=7.6)
    box(ax, 0.110, 0.340, 0.165, 0.135,
        "Numerical\nVPP state\n(load, PV, price,\nSOC, time)",
        C["in_face"], C["in_edge"], size=7.6)

    # (b) LLM advisor
    box(ax, 0.312, 0.660, 0.165, 0.135,
        "DeepSeek\nsemantic encoder\n(cached, emits\nno action)",
        C["llm_face"], C["llm_edge"], weight="bold", size=7.6)
    box(ax, 0.312, 0.400, 0.165, 0.115,
        "Semantic risk\nvector $s^{sem}$\n(5-dim, [0,1])",
        C["llm_face"], C["llm_edge"], size=7.6)

    # (c) gated MoE
    box(ax, 0.490, 0.500, 0.105, 0.150,
        "Augmented\nstate\n$s^{aug}=$\n$[s^{num},s^{sem}]$",
        C["neutral_face"], C["neutral_edge"], size=7.5)
    box(ax, 0.640, 0.660, 0.170, 0.120,
        "SAC-prior expert\n(regularized;\nstrong on known)",
        C["prior_face"], C["prior_edge"], weight="bold", size=7.5)
    box(ax, 0.640, 0.340, 0.170, 0.120,
        "SAC-free expert\n(no regularizer;\nadaptive on unseen)",
        C["free_face"], C["free_edge"], weight="bold", size=7.5)
    box(ax, 0.640, 0.500, 0.170, 0.095,
        "Learned gate\n$g_\\psi(s)\\!\\to\\! w\\in[0,1]$",
        C["gate_face"], C["gate_edge"], weight="bold", size=7.5)
    box(ax, 0.740, 0.500, 0.055, 0.300,
        "Blend\n$a=$\n$(1\\!-\\!w)a^{free}$\n$+\\,w\\,a^{prior}$",
        C["neutral_face"], C["neutral_edge"], size=6.8)

    # (d) output
    box(ax, 0.885, 0.660, 0.170, 0.120,
        "Feasibility clip\n(battery power,\nSOC limits)",
        C["out_face"], C["out_edge"], size=7.6)
    box(ax, 0.885, 0.470, 0.170, 0.095,
        "Final dispatch\naction $a^{final}$",
        C["out_face"], C["out_edge"], weight="bold", size=7.6)
    box(ax, 0.885, 0.260, 0.170, 0.140,
        "Evaluation\nreward, CVaR,\nthroughput,\nOOD real weather,\nunseen-event",
        C["neutral_face"], C["neutral_edge"], size=7.5)

    # arrows -------------------------------------------------------
    # inputs -> advisor
    straight(ax, (0.193, 0.660), (0.230, 0.660), C["in_edge"])           # text -> encoder
    straight(ax, (0.312, 0.593), (0.312, 0.458), C["llm_edge"])          # encoder -> s^sem
    # into augmented state (orthogonal, no diagonals)
    ortho(ax, (0.395, 0.400), (0.4375, 0.470), C["llm_edge"],
          angleA=0, angleB=90, rad=7)                                    # s^sem -> aug
    ortho(ax, (0.193, 0.340), (0.4375, 0.470), C["in_edge"],
          angleA=0, angleB=-90, rad=7)                                   # num state -> aug
    # augmented -> experts and gate
    ortho(ax, (0.5425, 0.545), (0.555, 0.660), C["neutral_edge"],
          angleA=0, angleB=90, rad=7)                                    # aug -> prior
    ortho(ax, (0.5425, 0.455), (0.555, 0.340), C["neutral_edge"],
          angleA=0, angleB=-90, rad=7)                                   # aug -> free
    straight(ax, (0.5425, 0.500), (0.555, 0.500), C["neutral_edge"])     # aug -> gate
    # experts + gate -> blend (orthogonal), actions carried as labels
    ortho(ax, (0.725, 0.660), (0.7125, 0.585), C["prior_edge"],
          angleA=0, angleB=90, rad=6, label="$a^{prior}$", loff=(0.028, 0.028))
    ortho(ax, (0.725, 0.340), (0.7125, 0.415), C["free_edge"],
          angleA=0, angleB=-90, rad=6, label="$a^{free}$", loff=(0.028, -0.028))
    straight(ax, (0.725, 0.500), (0.7125, 0.500), C["gate_edge"],
             label="$w$", loff=(0.0, 0.028))
    # blend -> output
    ortho(ax, (0.7675, 0.560), (0.800, 0.660), C["out_edge"],
          angleA=0, angleB=90, rad=6)                                    # blend -> clip
    straight(ax, (0.885, 0.600), (0.885, 0.518), C["out_edge"])          # clip -> final
    straight(ax, (0.885, 0.422), (0.885, 0.330), C["neutral_edge"])      # final -> eval

    # gate-semantics caption + legend
    ax.text(0.5, 0.028,
            "Gate semantics:  high $w$ on known events (defer to prior)   /   "
            "low $w$ on unseen events (defer to free actor)",
            ha="center", va="center", fontsize=7.8, fontstyle="italic",
            color=C["gate_edge"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "ieee_vpp_ledrl_architecture.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "ieee_vpp_ledrl_architecture.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw()
    print(f"Saved architecture figure to {OUT_DIR}")
