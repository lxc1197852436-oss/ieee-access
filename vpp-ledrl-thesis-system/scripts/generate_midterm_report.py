from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_CSV = ROOT / "outputs" / "midterm" / "midterm_model_comparison.csv"
REPORT = ROOT / "reports" / "midterm_progress_report.md"
OUTLINE = ROOT / "reports" / "midterm_defense_outline.md"


def read_rows() -> list[dict]:
    with RESULT_CSV.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def table(rows: list[dict]) -> str:
    headers = [
        "模型",
        "总收益(元)",
        "平均奖励",
        "CVaR 5%",
        "高价放电率",
        "低价充电率",
        "储能吞吐(MWh)",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append(
            "| {policy} | {total_reward_yuan} | {mean_reward_yuan} | {cvar_5_yuan} | {high_price_discharge_rate} | {low_price_charge_rate} | {battery_throughput_mwh} |".format(
                **r
            )
        )
    return "\n".join(lines)


def best(rows: list[dict], key: str, higher: bool = True) -> dict:
    return sorted(rows, key=lambda r: float(r[key]), reverse=higher)[0]


def main() -> None:
    rows = read_rows()
    best_reward = best(rows, "total_reward_yuan", higher=True)
    best_risk = best(rows, "cvar_5_yuan", higher=True)

    report = f"""# 研究生中期答辩进度报告

## 课题名称

大语言模型与深度强化学习融合的虚拟电厂动态优化与决策研究

## 一、当前阶段定位

当前工作已从开题阶段的理论设计推进到“数据管线、仿真环境、策略基线、语义增强接口和展示系统”闭环验证阶段。项目已经具备中期答辩所需的阶段性成果：可以说明研究问题、数据来源、方法框架、初步实验结果、当前不足和后续计划。

需要强调的是，目前使用的是“国内公开数据校准的虚拟电厂仿真场景”，不是完整真实的虚拟电厂 15 分钟运行数据。该表述更严谨，也更符合国内公开数据可获得性。

## 二、已完成工作

### 1. 国内公开数据源调研与抓取

已完成优先级1国内数据源整理，形成以下结构化数据：

- 国家能源局 2024 年全社会用电量与分产业用电量数据。
- 国家能源局 2024 年可再生能源装机与发电量数据。
- 广东电力现货市场与独立储能公开披露数据，包括价格区间、储能充放电电量、充放电均价和价差。
- 基于上述公开数据校准的广东虚拟电厂 15 分钟样例场景，共 2880 行，覆盖 30 天。

相关文件：

- `data/processed/nea_2024_power_consumption.csv`
- `data/processed/nea_2024_renewable_summary.csv`
- `data/processed/guangdong_market_storage_2024.csv`
- `data/processed/china_vpp_priority1_guangdong_sample.csv`

### 2. 虚拟电厂仿真环境

已实现轻量级 VPP 调度环境，包含以下状态与约束：

- 数值状态：负荷、光伏出力、电价、温度、SOC、时间特征。
- 文本状态：高温预警、需求响应、价格尖峰、新能源消纳等中文事件。
- 动作空间：储能充放电功率。
- 约束项：储能功率边界、SOC 上下限、充放电效率。
- 奖励函数：购售电收益、储能退化成本、弃光惩罚和动作不可执行惩罚。

### 3. 语义增强策略与AI接口

已实现本地中文语义风险识别器，可将文本事件转为风险分数、价格尖峰分数、负荷压力分数和新能源消纳分数。项目已预留 OpenAI-compatible API 接口，后续可接入 OpenAI、DeepSeek、通义千问、智谱等主流模型，用于文本事件风险预测和调度解释。

### 4. 可训练强化学习基线

已实现离散动作最大熵 Soft-Q 强化学习基线，包含：

- `Soft-Q-Numeric`：仅使用数值状态。
- `Soft-Q-Semantic`：使用数值状态与文本语义状态。

该基线用于中期阶段证明训练闭环可运行。后续正式论文阶段将升级为连续动作 SAC 与 LE-DRL。

### 5. 前后端展示系统

已实现 FastAPI 后端与本地 Dashboard，支持：

- 一键运行虚拟电厂调度实验。
- 展示电价、SOC、储能动作轨迹。
- 展示模型收益、CVaR、储能利用率等指标。
- 展示中文文本事件和 AI 解释结果。

运行方式：

```bash
cd "/Users/xingchengli/Documents/New project/vpp-ledrl-thesis-system"
source .venv/bin/activate
python -m app.backend.main
```

访问地址：`http://127.0.0.1:8000`

## 三、阶段性实验结果

实验设置：使用广东公开数据校准的 30 天 15 分钟虚拟电厂样例数据，前 70% 作为训练集，后 30% 作为测试集。对比模型包括 Rule-Based、LE-DRL-Semantic、Soft-Q-Numeric、Soft-Q-Semantic 和 Random。

{table(rows)}

从当前结果看，`{best_reward["policy"]}` 获得最高总收益，`{best_risk["policy"]}` 在 CVaR 5% 风险指标上表现最好。语义启发式策略在高电价放电率和低价充电率上明显优于规则策略，说明文本事件和语义风险信号对储能调度具有一定辅助价值。

Soft-Q 训练基线目前没有超过语义启发式策略，说明当前离散动作、状态分箱和训练轮数仍较粗糙。这个结果可以在中期答辩中作为后续研究重点：下一阶段需要将轻量 Soft-Q 替换为连续动作 SAC，并对语义融合方式、奖励函数和训练稳定性进行优化。

## 四、当前创新点雏形

1. 构建了面向中国虚拟电厂场景的“公开数据校准 + 仿真 VPP”实验数据管线。
2. 将中文市场公告、气象预警和调度事件纳入强化学习状态空间，形成语义增强状态。
3. 实现了可复现实验平台，能够统一比较规则策略、语义增强策略和强化学习策略。
4. 预留主流 AI 接口，使大模型用于风险预测和决策解释，而不是直接不可控地输出调度动作。

## 五、存在问题

1. 国内公开数据仍以汇总披露为主，缺少真实完整的 15 分钟 VPP 运行数据。
2. 当前 LE-DRL-Semantic 是启发式语义策略，还不是最终训练版 LE-DRL。
3. Soft-Q 基线为离散动作最大熵强化学习，只能作为中期过渡模型。
4. 文本事件目前以真实公告/预警表达方式构造，后续需要接入更多真实历史文本。
5. 评价指标还需要加入约束违约率、收益波动、极端事件响应收益和消融实验。

## 六、下一阶段计划

### 1. 模型升级

- 实现 PyTorch 版连续动作 SAC。
- 构建 LE-DRL：数值状态 + 中文文本语义向量 + SAC。
- 加入文本消融实验：LE-DRL w/o Text。

### 2. 数据扩展

- 继续补充广东、山东、山西等电力现货公开信息。
- 接入可用天气 API 或 Open-Meteo 中国坐标历史天气。
- 补充气象预警、需求响应和市场公告文本样本。

### 3. 实验扩展

- 常规日、高温预警、价格尖峰、新能源消纳受限四类场景。
- Rule、SAC、LE-DRL、LE-DRL w/o Text、MILP 上界对比。
- 收益、CVaR、储能吞吐、SOC 越限率、极端场景响应效果评价。

### 4. 论文写作

- 完成第1章绪论、第2章相关理论、第3章数据与环境建模初稿。
- 将当前系统架构和中期实验结果写入“阶段性成果”。
- 明确数据边界，避免夸大国内公开数据的真实性和完整性。

## 七、中期答辩建议表述

建议在答辩中表述为：

“本研究已完成面向中国虚拟电厂场景的数据源调研、公开数据抓取、仿真环境搭建、语义增强状态构建、初步强化学习基线和可视化系统。当前实验表明，语义事件对储能调度收益和风险控制具有辅助价值。下一阶段将重点实现连续动作 SAC 与 LE-DRL 训练，并扩展中国区域数据和极端事件实验。”
"""

    outline = """# 中期答辩PPT建议提纲

1. 课题名称与研究目标
2. 研究背景：双碳目标、新型电力系统、虚拟电厂、文本信息盲区
3. 国内外研究现状：VPP调度、DRL、电力大模型、多模态融合
4. 问题定义：数值状态 + 文本事件的异构增强决策
5. 已完成工作一：国内公开数据抓取与广东场景校准
6. 已完成工作二：VPP仿真环境与奖励函数
7. 已完成工作三：语义增强策略与AI解释接口
8. 已完成工作四：Soft-Q训练基线与模型对比
9. 初步实验结果：收益、CVaR、高价放电率、低价充电率
10. 当前不足：真实数据不足、Soft-Q仍是过渡模型、需升级SAC
11. 下一步计划：连续SAC、LE-DRL、文本消融、MILP基准、极端场景
12. 预期论文贡献与时间安排
"""

    REPORT.write_text(report, encoding="utf-8")
    OUTLINE.write_text(outline, encoding="utf-8")
    print(f"Saved: {REPORT}")
    print(f"Saved: {OUTLINE}")


if __name__ == "__main__":
    main()

