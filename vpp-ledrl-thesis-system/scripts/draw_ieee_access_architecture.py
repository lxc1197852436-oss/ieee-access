"""Draw the IEEE Access architecture figure: gated mixture-of-experts LE-DRL-SAC.

Layout (left to right, four columns):
  (a) Inputs: text events + numerical state
  (b) LLM advisor: DeepSeek -> cached semantic risk vector (no action)
  (c) Gated MoE: two SAC experts (prior/free) + learned gate g(s)->w
  (d) Output & eval: blend -> feasibility clip -> action -> metrics
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "thesis_figures"
IEEE_FIG_DIR = Path(__file__).resolve().parents[1].parent / "ieee_pkg" / "ieee_access_vpp_ledrl_20260630" / "figures"


def configure_font():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10.5


def box(ax, x, y, w, h, text, color="#E8F4FD", edge="#2C5F8D", fontsize=10, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                       linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, text=None, color="#555555", lw=1.4):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                        linewidth=lw, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.18, text, ha="center", va="bottom",
                fontsize=9, color=color, style="italic")


def main():
    configure_font()
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # Title
    ax.text(7.75, 7.85, "Language-Enhanced VPP Dispatch with a Learned Event-Coverage Gate",
            ha="center", fontsize=13.5, fontweight="bold")

    # Column headers
    headers = [(1.35, "(a) Inputs"), (4.55, "(b) LLM advisor"),
               (8.55, "(c) Gated mixture-of-experts SAC"), (13.4, "(d) Output & eval")]
    for x, t in headers:
        ax.text(x, 7.25, t, ha="center", fontsize=10.5, fontweight="bold", color="#666666")

    # ---- (a) Inputs ----
    box(ax, 0.4, 5.55, 2.0, 1.0, "Operational\ntext events", color="#FFF4E5", edge="#C77F2C", fontsize=10.5)
    box(ax, 0.4, 4.15, 2.0, 1.0, "Numerical\nVPP state", color="#E5F5E5", edge="#3A7D3A", fontsize=10.5)
    ax.text(0.4, 3.7, "(weather, price,\ncurtailment notices)", ha="left", fontsize=8, color="#888888", style="italic")
    ax.text(0.4, 3.85, "", ha="left", fontsize=8)

    # ---- (b) LLM advisor ----
    box(ax, 3.6, 5.55, 2.1, 1.0, "DeepSeek\nsemantic encoder\n(cached, no action)",
        color="#F0E5F5", edge="#6B3D7A", bold=True, fontsize=10)
    box(ax, 3.6, 4.15, 2.1, 1.0, "Semantic risk\nfeatures $s^{sem}$\n(5-dim, $[0,1]$)",
        color="#F0E5F5", edge="#6B3D7A", fontsize=10)
    # augmented state node
    box(ax, 3.6, 2.75, 2.1, 0.75, "Augmented state\n$s^{aug}=[s^{num}, s^{sem}]$",
        color="#FFFFFF", edge="#444444", fontsize=9.5)

    # ---- (c) Gated MoE ----
    box(ax, 6.7, 5.85, 2.7, 0.85, "SAC-prior\n(regularized; strong on known events)",
        color="#E8F4FD", edge="#2C5F8D", bold=True, fontsize=9.5)
    box(ax, 6.7, 4.75, 2.7, 0.85, "SAC-free\n(no regularizer; adaptive on unseen)",
        color="#FFF4E5", edge="#C77F2C", bold=True, fontsize=9.5)
    box(ax, 10.1, 5.15, 2.4, 1.0, "Learned gate\n$g_\\psi(s) \\to w \\in [0,1]$\n(Q-difference trained)",
        color="#F0E5F5", edge="#6B3D7A", bold=True, fontsize=9.5)
    # action nodes
    box(ax, 6.7, 3.45, 2.7, 0.55, "$a^{prior}$", color="#E8F4FD", edge="#2C5F8D", fontsize=9.5)
    box(ax, 6.7, 2.65, 2.7, 0.55, "$a^{free}$", color="#FFF4E5", edge="#C77F2C", fontsize=9.5)
    # blend
    box(ax, 10.1, 3.05, 2.4, 0.85, "Blend\n$a=(1-w)a^{free}+w a^{prior}$",
        color="#FFFFFF", edge="#444444", fontsize=10, bold=True)

    # ---- (d) Output & eval ----
    box(ax, 13.0, 5.4, 2.3, 0.9, "Feasibility clip\n(battery power,\nSOC limits)",
        color="#E5F5E5", edge="#3A7D3A", fontsize=9.5)
    box(ax, 13.0, 4.05, 2.3, 0.85, "Final dispatch\naction $a^{final}$",
        color="#E5F5E5", edge="#3A7D3A", bold=True, fontsize=10)
    box(ax, 13.0, 2.4, 2.3, 1.2, "Evaluation\nreward, CVaR,\nthroughput,\nOOD real weather,\nunseen-event variants",
        color="#F5F5F5", edge="#666666", fontsize=9)

    # ---- Arrows ----
    # inputs -> encoder/state
    arrow(ax, 2.4, 6.05, 3.6, 6.05)
    arrow(ax, 2.4, 4.65, 3.6, 4.65)
    # encoder -> sem features
    arrow(ax, 4.65, 5.55, 4.65, 5.15)
    # sem -> augmented
    arrow(ax, 4.65, 4.15, 4.65, 3.5)
    # numeric -> augmented
    arrow(ax, 2.4, 4.4, 3.6, 3.15)

    # augmented -> two experts + gate
    arrow(ax, 5.7, 3.15, 6.7, 6.05, color="#2C5F8D")
    arrow(ax, 5.7, 3.15, 6.7, 5.0, color="#C77F2C")
    arrow(ax, 5.7, 3.15, 10.1, 5.55, color="#6B3D7A")

    # experts -> actions
    arrow(ax, 8.05, 5.85, 8.05, 4.0)
    arrow(ax, 8.05, 4.75, 8.05, 3.2)

    # gate -> blend (w)
    arrow(ax, 11.3, 5.15, 11.3, 3.9, text="$w$", color="#6B3D7A", lw=1.6)
    # actions -> blend
    arrow(ax, 9.4, 3.7, 10.1, 3.5)
    arrow(ax, 9.4, 2.92, 10.1, 3.2)

    # blend -> feasibility -> action -> eval
    arrow(ax, 12.5, 3.5, 13.0, 5.55)
    arrow(ax, 14.15, 5.4, 14.15, 4.9)
    arrow(ax, 14.15, 4.05, 14.15, 3.6)

    # Gate semantics caption
    ax.text(8.55, 1.85, "Gate semantics:  high $w$ on known events (defer to prior)   /   low $w$ on unseen events (defer to free actor)",
            ha="center", fontsize=9.5, style="italic", color="#6B3D7A")

    fig.tight_layout(pad=0.3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IEEE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig3_ledrl_sac_model_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_ledrl_sac_model_architecture.svg", bbox_inches="tight")
    fig.savefig(IEEE_FIG_DIR / "ieee_vpp_ledrl_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(IEEE_FIG_DIR / "ieee_vpp_ledrl_architecture.svg", bbox_inches="tight")
    plt.close(fig)
    print("Saved architecture figure.")


if __name__ == "__main__":
    main()
