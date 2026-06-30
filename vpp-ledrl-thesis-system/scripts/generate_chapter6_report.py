from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "chapter6" / "chapter6_scenario_results.csv"
REPORT = ROOT / "reports" / "chapter6_results_analysis.md"
SVG = ROOT / "outputs" / "chapter6" / "chapter6_total_reward.svg"


def read_rows() -> list[dict]:
    with RESULTS.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key])


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["policy"]].append(row)
    out = []
    for policy, items in groups.items():
        out.append(
            {
                "policy": policy,
                "avg_total_reward": sum(f(x, "total_reward_yuan") for x in items) / len(items),
                "avg_cvar": sum(f(x, "cvar_5_yuan") for x in items) / len(items),
                "avg_high_discharge": sum(f(x, "high_price_discharge_rate") for x in items) / len(items),
                "avg_low_charge": sum(f(x, "low_price_charge_rate") for x in items) / len(items),
            }
        )
    return sorted(out, key=lambda x: x["avg_total_reward"], reverse=True)


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| 场景 | 模型 | 总收益(元) | CVaR 5%(元) | 高价放电率 | 低价充电率 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['scenario_id']} {r['scenario_name']} | {r['policy']} | {f(r, 'total_reward_yuan'):.1f} | {f(r, 'cvar_5_yuan'):.1f} | {f(r, 'high_price_discharge_rate'):.3f} | {f(r, 'low_price_charge_rate'):.3f} |"
        )
    return "\n".join(lines)


def aggregate_table(rows: list[dict]) -> str:
    lines = [
        "| 模型 | 平均总收益(元) | 平均CVaR 5%(元) | 平均高价放电率 | 平均低价充电率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['policy']} | {r['avg_total_reward']:.1f} | {r['avg_cvar']:.1f} | {r['avg_high_discharge']:.3f} | {r['avg_low_charge']:.3f} |"
        )
    return "\n".join(lines)


def make_svg(rows: list[dict]) -> None:
    scenarios = sorted(set(r["scenario_id"] for r in rows))
    policies = ["Rule-Based", "LE-DRL-Semantic", "Rolling-Horizon", "Random", "Soft-Q-Numeric", "Soft-Q-Semantic"]
    lookup = {(r["scenario_id"], r["policy"]): f(r, "total_reward_yuan") for r in rows}
    values = [lookup[(s, p)] for s in scenarios for p in policies if (s, p) in lookup]
    min_v, max_v = min(values), max(values)
    width, height = 1120, 620
    left, top = 130, 70
    group_w = 230
    bar_w = 26
    scale_h = 360
    colors = {
        "Rule-Based": "#2155a3",
        "LE-DRL-Semantic": "#0f8b8d",
        "Rolling-Horizon": "#7c3aed",
        "Random": "#64748b",
        "Soft-Q-Numeric": "#b45309",
        "Soft-Q-Semantic": "#dc2626",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        '<text x="30" y="35" font-size="22" font-weight="700" fill="#172033">第6章多场景总收益对比</text>',
        '<text x="30" y="58" font-size="13" fill="#5c667a">值越接近0表示亏损越小；当前为阶段性基线实验，正式结论需加入长训练SAC/LE-DRL。</text>',
    ]
    for si, s in enumerate(scenarios):
        base_x = left + si * group_w
        parts.append(f'<text x="{base_x}" y="{top + scale_h + 38}" font-size="13" font-weight="600" fill="#172033">{s}</text>')
        for pi, p in enumerate(policies):
            if (s, p) not in lookup:
                continue
            v = lookup[(s, p)]
            h = 20 + (v - min_v) / max(1e-9, max_v - min_v) * (scale_h - 20)
            x = base_x + pi * (bar_w + 5)
            y = top + scale_h - h
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="3" fill="{colors[p]}"/>')
    legend_x, legend_y = 30, height - 125
    for i, p in enumerate(policies):
        y = legend_y + i * 18
        parts.append(f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" fill="{colors[p]}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{y}" font-size="12" fill="#172033">{p}</text>')
    parts.append("</svg>")
    SVG.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    rows = read_rows()
    agg = aggregate(rows)
    best = agg[0]
    make_svg(rows)
    text = f"""# 第6章 实验结果与分析

## 6.1 本章概述

本章基于第5章确定的实验设计，对当前已经实现的模型进行多场景评估。需要说明的是，本章当前结果属于阶段性实验：Rule-Based、LE-DRL-Semantic、Soft-Q 与 Rolling-Horizon 已完成跨场景评估；连续动作 SAC 与 LE-DRL-SAC 已在第4章完成短训练验证，但尚未进行 50-100 episode 长训练和多随机种子实验。因此，当前第6章重点是建立结果分析框架、识别模型问题和明确正式实验改进方向。

实验场景包括常规夏季运行、高温负荷压力、价格尖峰和新能源消纳压力四类。评价指标包括总收益、CVaR 5%、高价放电率和低价充电率。

## 6.2 多场景实验结果

{markdown_table(rows)}

## 6.3 平均性能对比

{aggregate_table(agg)}

从平均总收益看，当前表现最好的模型为 `{best['policy']}`，平均总收益为 {best['avg_total_reward']:.1f} 元。Rule-Based 在多数场景中仍具有较强竞争力，说明固定阈值策略在当前价格校准和储能参数下并不弱。LE-DRL-Semantic 在中期单测试集上表现较好，但在多场景实验中并未稳定超过 Rule-Based，说明启发式语义策略的泛化能力有限。

## 6.4 分场景分析

### 6.4.1 常规夏季运行场景

在 S1 场景中，各模型面对常规负荷、电价和光伏波动。Rule-Based 与 LE-DRL-Semantic 的收益接近，而 Soft-Q 系列明显较差。这说明当前离散 Soft-Q 的状态分箱和动作选择仍不够精细，尚不能替代连续动作强化学习。

### 6.4.2 高温负荷压力场景

S2 场景增强了高温预警和晚高峰负荷压力。该场景用于测试文本语义是否能够帮助模型提前保留储能。从当前结果看，语义启发式策略未明显超过规则策略，说明仅依靠规则语义分数还不足以形成稳定优势，后续需要训练版 LE-DRL-SAC 从交互中学习何时保留电量、何时高价放电。

### 6.4.3 价格尖峰场景

S3 场景中 Rolling-Horizon 的收益略优于 Rule-Based，说明在价格尖峰条件下，具备短期未来信息的滚动优化能够更好地安排储能动作。该结果可作为后续 SAC/LE-DRL 的强基准参照。

### 6.4.4 新能源消纳压力场景

S4 场景增强午间光伏出力与新能源消纳文本事件。当前 Rule-Based 和 LE-DRL-Semantic 接近，但 Rolling-Horizon 并未表现出明显优势，原因可能是当前弃光惩罚权重和价格收益之间的权衡仍需调参。后续实验应单独绘制午间光伏、储能充电和弃光惩罚曲线。

## 6.5 当前模型问题

1. Soft-Q 使用离散动作和状态分箱，表达能力不足，在多场景中显著弱于规则策略。
2. LE-DRL-Semantic 属于启发式语义策略，能体现语义输入的作用，但缺乏训练适应能力。
3. Rolling-Horizon 当前为离散动态规划近似强基准，缺少真实 MILP 求解器和终端 SOC 价值约束。
4. SAC-Numeric 与 LE-DRL-SAC 仍处于短训练验证阶段，尚不能作为最终第6章结论。

## 6.6 后续正式实验安排

正式论文第6章需要在当前框架上补充：

1. 将 SAC-Numeric 与 LE-DRL-SAC 训练轮数提升到 50-100 episode。
2. 增加至少 3 个随机种子，报告均值和标准差。
3. 增加 LE-DRL w/o Text 消融实验。
4. 将 Rolling-Horizon 升级为更严格的 MILP 或加入终端 SOC 约束。
5. 对高温、价格尖峰和新能源消纳场景分别绘制动作轨迹和 SOC 轨迹。

## 6.7 本章小结

本章完成了多场景实验框架和阶段性结果分析。结果表明，当前规则策略仍是较强基线，启发式语义策略和 Soft-Q 尚不足以形成最终论文结论；但实验也证明了系统已经能够在多场景下统一评估模型，并能够输出收益、风险和储能行为指标。下一步应重点完成连续动作 SAC 与 LE-DRL-SAC 的长训练和文本消融实验，从而形成最终毕业论文的核心实验结果。
"""
    REPORT.write_text(text, encoding="utf-8")
    print(f"Saved: {REPORT}")
    print(f"Saved: {SVG}")


if __name__ == "__main__":
    main()

