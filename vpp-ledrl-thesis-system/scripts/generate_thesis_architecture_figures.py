from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "thesis_figures"
REPORT_PATH = ROOT / "reports" / "thesis_architecture_diagrams.md"


def configure_font() -> None:
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"


def box(ax, xy, wh, text, fc="#F7FAFC", ec="#2B6CB0", fontsize=12, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.045",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#1A202C",
        linespacing=1.35,
    )
    return patch


def arrow(ax, start, end, color="#4A5568", rad=0.0, lw=1.5):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def lane_label(ax, y, text):
    ax.text(
        0.035,
        y,
        text,
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#2D3748",
    )


def finalize(fig, ax, filename: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout(pad=0.25)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename}.svg", bbox_inches="tight")
    plt.close(fig)


def draw_overall_architecture():
    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    ax.text(
        0.5,
        0.965,
        "论文总体技术架构：AI文本风险预测 + 强化学习动作决策 + AI决策解释",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#102A43",
    )

    lane_label(ax, 0.82, "数据与场景")
    lane_label(ax, 0.60, "语义理解")
    lane_label(ax, 0.38, "强化学习决策")
    lane_label(ax, 0.16, "评估与论文输出")

    data1 = box(ax, (0.12, 0.74), (0.18, 0.12), "国内公开数据\n国家能源局 / 广东市场\n标准与行业报告", "#EBF8FF", "#2B6CB0")
    data2 = box(ax, (0.34, 0.74), (0.18, 0.12), "VPP仿真数据\n负荷 / 光伏 / 电价\n温度 / SOC", "#EBF8FF", "#2B6CB0")
    data3 = box(ax, (0.56, 0.74), (0.18, 0.12), "中文文本事件\n气象预警 / 市场公告\n需求响应 / 消纳提示", "#EBF8FF", "#2B6CB0")
    env = box(ax, (0.78, 0.74), (0.16, 0.12), "统一VPP环境\n15分钟调度步长\n约束与奖励函数", "#E6FFFA", "#2C7A7B")

    sem1 = box(ax, (0.18, 0.52), (0.2, 0.12), "DeepSeek / 本地编码器\n文本事件结构化", "#FFF5F5", "#C53030")
    sem2 = box(ax, (0.44, 0.52), (0.2, 0.12), "语义风险特征\nrisk / price_spike\nload / curtailment", "#FFF5F5", "#C53030")
    sem3 = box(ax, (0.70, 0.52), (0.2, 0.12), "AI决策解释\n解释充电 / 放电 / 保持\n不直接输出动作", "#FFF5F5", "#C53030")

    rl1 = box(ax, (0.12, 0.30), (0.18, 0.12), "数值状态\n负荷 光伏 电价\n温度 SOC 时间", "#F0FFF4", "#2F855A")
    rl2 = box(ax, (0.34, 0.30), (0.18, 0.12), "增强状态\n数值状态 + 语义状态", "#F0FFF4", "#2F855A")
    rl3 = box(ax, (0.56, 0.30), (0.18, 0.12), "LE-DRL-SAC\nSAC Actor-Critic\n语义一致性训练", "#F0FFF4", "#2F855A")
    rl4 = box(ax, (0.78, 0.30), (0.16, 0.12), "储能动作\n连续充放电功率\n环境约束校验", "#F0FFF4", "#2F855A")

    out1 = box(ax, (0.16, 0.08), (0.18, 0.12), "对比模型\nSAC-Numeric\nLE-DRL w/o Text", "#FFFAF0", "#B7791F")
    out2 = box(ax, (0.41, 0.08), (0.18, 0.12), "多场景评估\nS1-S4 / 多随机种子\n收益与CVaR", "#FFFAF0", "#B7791F")
    out3 = box(ax, (0.66, 0.08), (0.2, 0.12), "论文与展示输出\n图表 / 轨迹 / Dashboard\n参考文献与结论", "#FFFAF0", "#B7791F")

    for a, b in [(data1, data2), (data2, data3), (data3, env)]:
        arrow(ax, (a.get_x() + a.get_width(), a.get_y() + a.get_height() / 2), (b.get_x(), b.get_y() + b.get_height() / 2))
    arrow(ax, (data3.get_x() + data3.get_width() / 2, data3.get_y()), (sem1.get_x() + sem1.get_width() / 2, sem1.get_y() + sem1.get_height()))
    arrow(ax, (sem1.get_x() + sem1.get_width(), sem1.get_y() + sem1.get_height() / 2), (sem2.get_x(), sem2.get_y() + sem2.get_height() / 2))
    arrow(ax, (sem2.get_x() + sem2.get_width(), sem2.get_y() + sem2.get_height() / 2), (sem3.get_x(), sem3.get_y() + sem3.get_height() / 2))
    arrow(ax, (env.get_x() + env.get_width() / 2, env.get_y()), (rl4.get_x() + rl4.get_width() / 2, rl4.get_y() + rl4.get_height()))
    arrow(ax, (rl1.get_x() + rl1.get_width(), rl1.get_y() + rl1.get_height() / 2), (rl2.get_x(), rl2.get_y() + rl2.get_height() / 2))
    arrow(ax, (sem2.get_x() + sem2.get_width() / 2, sem2.get_y()), (rl2.get_x() + rl2.get_width() / 2, rl2.get_y() + rl2.get_height()))
    arrow(ax, (rl2.get_x() + rl2.get_width(), rl2.get_y() + rl2.get_height() / 2), (rl3.get_x(), rl3.get_y() + rl3.get_height() / 2))
    arrow(ax, (rl3.get_x() + rl3.get_width(), rl3.get_y() + rl3.get_height() / 2), (rl4.get_x(), rl4.get_y() + rl4.get_height() / 2))
    arrow(ax, (rl4.get_x() + rl4.get_width() / 2, rl4.get_y()), (out2.get_x() + out2.get_width() / 2, out2.get_y() + out2.get_height()))
    arrow(ax, (out1.get_x() + out1.get_width(), out1.get_y() + out1.get_height() / 2), (out2.get_x(), out2.get_y() + out2.get_height() / 2))
    arrow(ax, (out2.get_x() + out2.get_width(), out2.get_y() + out2.get_height() / 2), (out3.get_x(), out3.get_y() + out3.get_height() / 2))
    arrow(ax, (rl4.get_x() + rl4.get_width(), rl4.get_y() + rl4.get_height() / 2), (sem3.get_x() + sem3.get_width() / 2, sem3.get_y()), rad=-0.25, color="#A0AEC0")

    finalize(fig, ax, "fig1_overall_research_architecture")


def draw_system_architecture():
    fig, ax = plt.subplots(figsize=(14.5, 7.8))
    ax.text(0.5, 0.95, "实验系统实现架构", ha="center", fontsize=18, fontweight="bold", color="#102A43")

    layers = [
        ("数据层", 0.76, "#EBF8FF", "#2B6CB0", ["公开数据抓取", "结构化数据表", "AI语义特征缓存", "实验结果CSV"]),
        ("环境层", 0.58, "#E6FFFA", "#2C7A7B", ["VPPEnv", "SOC状态转移", "收益/退化/弃光惩罚", "动作可行性校验"]),
        ("模型层", 0.40, "#F0FFF4", "#2F855A", ["Rule-Based", "SAC-Numeric", "LE-DRL-SAC", "LE-DRL w/o Text"]),
        ("服务与展示层", 0.22, "#FFFAF0", "#B7791F", ["FastAPI接口", "本地Dashboard", "训练/评估脚本", "论文图表导出"]),
    ]
    prev_centers = None
    for label, y, fc, ec, items in layers:
        ax.text(0.07, y + 0.045, label, ha="left", va="center", fontsize=14, fontweight="bold", color="#2D3748")
        centers = []
        for j, item in enumerate(items):
            x = 0.18 + j * 0.19
            b = box(ax, (x, y), (0.15, 0.09), item, fc, ec, fontsize=11)
            centers.append((x + 0.075, y))
        if prev_centers is not None:
            for c1, c2 in zip(prev_centers, centers):
                arrow(ax, (c1[0], c1[1]), (c2[0], y + 0.09), color="#718096")
        prev_centers = centers

    ax.text(
        0.5,
        0.08,
        "设计原则：数据可追溯、环境可复现、算法可替换、结果可展示；AI只负责语义理解与解释，最终动作由强化学习策略输出。",
        ha="center",
        fontsize=12,
        color="#2D3748",
    )
    finalize(fig, ax, "fig2_system_implementation_architecture")


def draw_model_architecture():
    fig, ax = plt.subplots(figsize=(14.5, 7.8))
    ax.text(0.5, 0.95, "LE-DRL-SAC模型架构", ha="center", fontsize=18, fontweight="bold", color="#102A43")

    n1 = box(ax, (0.08, 0.68), (0.18, 0.12), "数值输入\n负荷 / 光伏 / 电价\n温度 / SOC / 时间", "#EBF8FF", "#2B6CB0")
    t1 = box(ax, (0.08, 0.43), (0.18, 0.12), "文本输入\n气象预警 / 市场公告\n需求响应 / 消纳提示", "#FFF5F5", "#C53030")
    n2 = box(ax, (0.32, 0.68), (0.16, 0.12), "数值归一化\nStateEncoder", "#EBF8FF", "#2B6CB0")
    t2 = box(ax, (0.32, 0.43), (0.16, 0.12), "AI语义编码\nDeepSeek / 本地规则", "#FFF5F5", "#C53030")
    fuse = box(ax, (0.54, 0.56), (0.16, 0.12), "增强状态\ns_aug = [s_num, s_sem]", "#F0FFF4", "#2F855A")
    actor = box(ax, (0.76, 0.66), (0.16, 0.11), "Actor网络\n输出连续动作", "#F0FFF4", "#2F855A")
    critic = box(ax, (0.76, 0.44), (0.16, 0.11), "Twin Critic\nQ1 / Q2评估", "#F0FFF4", "#2F855A")
    env = box(ax, (0.54, 0.22), (0.16, 0.12), "VPP环境\n执行动作并返回奖励", "#E6FFFA", "#2C7A7B")
    loss = box(ax, (0.76, 0.22), (0.16, 0.12), "训练目标\nSAC损失 + 语义一致性正则", "#FFFAF0", "#B7791F")

    arrow(ax, (0.26, 0.74), (0.32, 0.74))
    arrow(ax, (0.26, 0.49), (0.32, 0.49))
    arrow(ax, (0.48, 0.74), (0.54, 0.62))
    arrow(ax, (0.48, 0.49), (0.54, 0.62))
    arrow(ax, (0.70, 0.62), (0.76, 0.715))
    arrow(ax, (0.70, 0.62), (0.76, 0.495))
    arrow(ax, (0.84, 0.66), (0.66, 0.34), rad=-0.15)
    arrow(ax, (0.70, 0.28), (0.76, 0.28))
    arrow(ax, (0.84, 0.44), (0.84, 0.34))
    arrow(ax, (0.76, 0.28), (0.48, 0.49), rad=0.22, color="#718096")
    arrow(ax, (0.62, 0.22), (0.62, 0.56), rad=-0.2, color="#718096")

    ax.text(
        0.5,
        0.08,
        "关键边界：大语言模型不直接输出储能动作；其输出作为语义状态和解释依据，最终动作由可复现的SAC Actor产生。",
        ha="center",
        fontsize=12,
        color="#2D3748",
    )
    finalize(fig, ax, "fig3_ledrl_sac_model_architecture")


def write_report():
    text = """# 论文架构图说明

已生成三张可直接用于论文或答辩 PPT 的架构图。

| 图号建议 | 文件 | 建议位置 | 用途 |
|---|---|---|---|
| 图1-1 或 图5-1 | `outputs/thesis_figures/fig1_overall_research_architecture.png` | 第1章技术路线或第5章系统总览 | 展示全文“大架构”：公开数据、VPP环境、AI语义、LE-DRL-SAC、评估输出 |
| 图5-1 | `outputs/thesis_figures/fig2_system_implementation_architecture.png` | 第5章系统实现 | 展示数据层、环境层、模型层、服务与展示层 |
| 图4-1 | `outputs/thesis_figures/fig3_ledrl_sac_model_architecture.png` | 第4章方法设计 | 展示数值状态、文本语义、增强状态、Actor-Critic训练关系 |

## 推荐正文引用写法

在第1章技术路线部分可写：

“本文总体技术架构如图1-1所示。系统首先基于国内公开数据构建虚拟电厂仿真环境，再利用 DeepSeek 或本地语义编码器将中文文本事件转化为结构化风险特征，随后将数值状态与语义状态共同输入 LE-DRL-SAC 模型进行训练。最终通过多场景、多随机种子和文本消融实验评价模型收益、风险和储能行为。”

在第4章方法部分可写：

“LE-DRL-SAC 模型结构如图4-1所示。大语言模型不直接输出调度动作，而是提供文本事件风险特征和决策解释；最终充放电动作由 SAC Actor 输出，并由 VPP 环境进行物理约束校验。”
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    configure_font()
    draw_overall_architecture()
    draw_system_architecture()
    draw_model_architecture()
    write_report()
    print(f"Saved figures to: {OUT_DIR}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
