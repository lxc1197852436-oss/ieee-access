# 下一步开发路线

## 当前版本定位

当前项目是论文系统的第一版干净工程包，已经包含：

- 中国虚拟电厂风格场景数据生成。
- VPP 储能调度环境。
- Rule、Random、LE-DRL-Semantic 三类策略接口。
- 本地语义风险预测器。
- OpenAI-compatible 主流 AI 接入预留。
- FastAPI 后端和本地 Dashboard。

当前 `LE-DRL-Semantic` 是语义增强启发式策略，不是最终训练版 SAC。它用于先跑通系统闭环和答辩展示面。

## 优先级 1：把中国数据接进来

建议先确定一个省级或区域实验对象，例如广东、山东或山西。

需要准备 CSV 字段：

```text
timestamp,load_mw,pv_mw,price_yuan_mwh,temperature_c,event_type,event_text
```

如果真实数据缺少光伏或负荷，可先用公开统计和气象数据校准仿真参数，但论文中必须明确“公开数据 + 仿真构造”的边界。

## 优先级 2：接入主流 AI

后续拿到 API key 后，修改 `.env`：

```text
AI_PROVIDER=openai_compatible
AI_BASE_URL=https://api.openai.com/v1
AI_API_KEY=你的key
AI_MODEL=gpt-4.1-mini
```

如果使用 DeepSeek、通义千问、智谱等 OpenAI-compatible 接口，只需要替换 `AI_BASE_URL` 和 `AI_MODEL`。

AI 在本论文中建议承担两个角色：

- 文本事件风险预测：把市场公告、气象预警、调度通知转为风险等级和趋势判断。
- 决策解释：解释为什么当前策略选择充电、放电或保持。

不建议直接让大模型输出最终调度动作，最终动作仍应由可复现的强化学习策略产生。

## 优先级 3：替换为训练版 SAC / LE-DRL

推荐训练模块结构：

```text
app/core/rl/
  replay_buffer.py
  sac.py
  ledrl_agent.py
scripts/train_sac.py
scripts/train_ledrl.py
scripts/evaluate_policies.py
```

训练版论文对比至少包括：

- Rule-based
- SAC，仅数值状态
- LE-DRL，数值状态 + 文本语义状态
- LE-DRL w/o Text，文本消融
- MILP rolling horizon，上界或强基准

核心评价指标：

- 总净收益
- CVaR 5%
- SOC 越限率
- 高电价放电比例
- 低电价/高光伏充电比例
- 极端文本事件下的响应收益

## 优先级 4：论文展示图表

前端展示面建议保留以下图：

- 电价、SOC、储能动作三联图。
- 不同模型收益对比柱状图。
- CVaR 风险对比。
- 文本事件时间线。
- AI 解释列表。

这些图可以直接作为论文“系统实现与结果分析”章节的截图或复绘依据。

