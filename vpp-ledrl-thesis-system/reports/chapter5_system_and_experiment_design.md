# 第5章 系统实现与实验设计

## 5.1 本章概述

第3章完成了数据体系与虚拟电厂环境建模，第4章完成了 SAC 与 LE-DRL-SAC 方法设计。本章进一步说明论文实验平台的系统实现方式，并给出正式实验设计。系统实现部分用于支撑算法可复现和中期/答辩展示；实验设计部分用于明确后续第6章需要完成的对比模型、场景设置和评价指标。

本文系统实现遵循“数据可追溯、环境可复现、算法可替换、结果可展示”的原则。后端负责数据读取、环境仿真、模型训练和指标计算；前端负责展示电价、SOC、储能动作、文本事件和模型指标；算法层支持规则策略、过渡强化学习基线、连续 SAC 和 LE-DRL-SAC[21, 22, 29, C1]。

## 5.2 系统总体架构

系统分为五层：

1. 数据层：存储公开抓取数据、结构化指标、广东 VPP 样例数据和实验结果。
2. 环境层：实现虚拟电厂功率平衡、储能 SOC 转移、奖励函数和约束检查。
3. 模型层：实现 Rule-Based、Soft-Q、SAC-Numeric 和 LE-DRL-SAC 等策略。
4. 服务层：通过 FastAPI 提供实验运行、场景预览和系统健康检查接口。
5. 展示层：通过本地 Dashboard 展示实验轨迹、指标和 AI 解释。

![图5-1 实验系统实现架构](../outputs/thesis_figures/fig2_system_implementation_architecture.png)

图5-1展示了本文实验系统实现架构。数据层负责公开数据、仿真场景和实验结果管理；环境层负责 VPP 状态转移、奖励函数和约束检查；模型层负责 Rule-Based、SAC-Numeric、LE-DRL-SAC 和文本消融模型；服务与展示层负责 API、Dashboard、训练评估脚本和论文图表输出。

当前项目目录如下：

```text
app/
  core/
    data.py                 数据读取与场景生成
    environment.py          VPP环境
    semantic.py             本地语义编码器
    simulation.py           策略运行与指标计算
    rl/                     SAC与LE-DRL训练模块
  backend/
    main.py                 FastAPI服务
  web/
    index.html              展示页面
scripts/
  crawl_priority1_sources.py
  build_priority1_dataset.py
  train_midterm_agents.py
  train_chapter4_sac.py
reports/
  chapter3_*.md
  chapter4_*.md
  chapter5_*.csv/json
```

## 5.3 后端服务实现

后端采用 FastAPI 实现，入口文件为：

`app/backend/main.py`

当前主要接口如下：

| 接口 | 方法 | 功能 |
|---|---|---|
| `/` | GET | 返回本地 Dashboard 页面 |
| `/api/health` | GET | 检查服务是否正常 |
| `/api/scenario` | POST | 根据起始时间、天数、随机种子生成场景预览 |
| `/api/run` | POST | 运行策略对比实验并返回轨迹、指标和解释 |

通过该接口，前端可以在不直接接触训练代码的情况下调用仿真服务。后续如果需要展示正式 SAC/LE-DRL 训练结果，可继续扩展 `/api/train`、`/api/evaluate` 和 `/api/results` 等接口。

## 5.4 前端展示系统

前端采用纯 HTML、CSS 和 JavaScript 实现，避免引入复杂构建链，保证答辩演示时启动简单、可控。前端文件位于：

`app/web/`

当前展示内容包括：

- 模型收益、CVaR、终止 SOC 等指标卡片。
- 电价、SOC、储能动作轨迹图。
- 文本事件列表。
- AI 风险预测与调度解释列表。

该展示系统服务于两个目的：一是作为研究过程中的调试面板，观察模型动作是否符合电价和文本事件变化；二是作为中期答辩和最终答辩中的工程实现展示材料。

## 5.5 AI解释接口设计

AI 接口位于：

`app/core/llm_provider.py`

当前默认使用本地规则预测器，不需要 API key。当后续提供主流 AI API key 后，可通过 `.env` 配置 OpenAI-compatible 接口：

```text
AI_PROVIDER=openai_compatible
AI_BASE_URL=https://api.openai.com/v1
AI_API_KEY=你的key
AI_MODEL=gpt-4.1-mini
```

本文对 AI 的定位是“语义理解与解释辅助”，而不是让大模型直接输出最终调度动作。最终动作仍由可复现的强化学习策略输出，以保证实验可重复和可评价[23-28]。

## 5.6 实验场景设计

正式实验设计四类场景。场景设计参考虚拟电厂调度、需求响应、新能源消纳和储能调度研究中的典型风险类型[1-3, 8, 13, 31, 38]：

| 场景编号 | 场景名称 | 目的 |
|---|---|---|
| S1 | 常规夏季运行场景 | 验证模型基础经济性 |
| S2 | 高温负荷压力场景 | 验证文本语义对提前储能和风险控制的作用 |
| S3 | 价格尖峰场景 | 测试市场公告和高价信号下的套利能力 |
| S4 | 新能源消纳压力场景 | 测试储能对午间光伏消纳和弃光惩罚的响应 |

场景配置代码位于：

`app/core/experiment_design.py`

场景矩阵输出文件为：

`reports/chapter5_scenarios.csv`

## 5.7 对比模型设计

本文计划比较以下模型：

| 模型 | 用途 |
|---|---|
| Rule-Based | 固定阈值规则基线 |
| Soft-Q-Numeric | 中期离散动作最大熵RL基线 |
| Soft-Q-Semantic | 中期语义增强RL基线 |
| SAC-Numeric | 正式连续动作数值状态基线 |
| LE-DRL-SAC | 本文核心方法 |
| LE-DRL w/o Text | 文本消融模型 |
| MILP Rolling Horizon | 滚动优化强基准或上界 |

其中，Rule-Based、Soft-Q、SAC-Numeric 和 LE-DRL-SAC 已完成初版实现。`LE-DRL w/o Text` 和 `MILP Rolling Horizon` 是后续第6章正式实验前需要补齐的两项。

## 5.8 评价指标体系

评价指标分为经济性、风险性、储能行为、约束满足和语义事件响应五类。总收益、尾部风险、储能吞吐量和高低价动作比例能够同时反映经济性、风险性和储能行为质量[31, 39, 44, 45]。

| 指标 | 单位 | 含义 |
|---|---:|---|
| 总收益 | 元 | 全测试期累计奖励 |
| 平均奖励 | 元/步 | 单调度步平均收益 |
| CVaR 5% | 元 | 最差 5% 调度步收益均值，衡量尾部风险 |
| 储能吞吐量 | MWh | 储能充放电总强度 |
| 高价放电率 | 比例 | 高电价时段放电占比 |
| 低价充电率 | 比例 | 低电价时段充电占比 |
| 终止 SOC | 比例 | 测试结束时储能状态 |
| 文本事件数 | 次 | 场景中文本事件触发数量 |
| 约束惩罚 | 元 | 因动作不可执行产生的惩罚 |
| 弃光惩罚 | 元 | 新能源消纳压力导致的惩罚 |

第6章正式实验应至少报告总收益、CVaR 5%、储能吞吐量、高价放电率和低价充电率。若篇幅允许，应进一步报告不同文本事件触发前后的动作变化。

## 5.9 实验流程

完整实验流程如下：

1. 运行数据抓取脚本，获得公开指标。
2. 构建广东 VPP 样例数据。
3. 校验数据质量和环境物理约束。
4. 训练 SAC-Numeric 与 LE-DRL-SAC。
5. 在四类场景上评估所有模型。
6. 汇总总收益、CVaR、储能行为和文本事件响应指标。
7. 生成对比图、轨迹图和消融实验表。

建议正式实验命令顺序如下：

```bash
python scripts/crawl_priority1_sources.py
python scripts/build_substitute_utilization_data.py
python scripts/build_priority1_dataset.py
python scripts/validate_chapter3_data_env.py
python scripts/train_chapter4_sac.py --episodes 50
python scripts/generate_chapter5_experiment_matrix.py
```

## 5.10 可复现性设计

为了保证实验可复现，本文系统采用以下设计：

- 所有场景配置固定随机种子。
- 原始网页、结构化 CSV、样例数据和结果 JSON 均保存在项目目录。
- 模型检查点保存到 `outputs/`。
- 训练脚本和评估脚本分离，便于重复运行。
- 数据字段字典和校验报告单独保存，便于论文审查。

## 5.11 本章小结

本章完成了虚拟电厂实验系统的实现说明和正式实验设计。系统层面，本文构建了包含数据层、环境层、模型层、服务层和展示层的完整实验平台；实验层面，本文设计了四类典型运行场景、七类对比模型和多维评价指标。该设计为第6章开展 SAC 与 LE-DRL-SAC 的正式训练、对比实验和消融分析奠定了基础。
