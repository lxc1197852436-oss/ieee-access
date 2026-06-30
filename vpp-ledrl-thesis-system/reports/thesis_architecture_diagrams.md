# 论文架构图说明

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
