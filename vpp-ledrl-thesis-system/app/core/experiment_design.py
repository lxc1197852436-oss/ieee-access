from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ExperimentScenario:
    scenario_id: str
    name: str
    description: str
    start: str
    days: int
    seed: int
    region: str = "广东省"
    stress_type: str = "normal"

    @property
    def periods(self) -> int:
        return self.days * 96


SCENARIOS: list[ExperimentScenario] = [
    ExperimentScenario(
        scenario_id="S1",
        name="常规夏季运行场景",
        description="负荷、电价和光伏出力均处于常规波动范围，用于验证模型基础经济性。",
        start="2025-07-01 00:00:00",
        days=7,
        seed=2026,
        stress_type="normal",
    ),
    ExperimentScenario(
        scenario_id="S2",
        name="高温负荷压力场景",
        description="增强高温预警与晚高峰负荷压力，用于测试文本语义对提前保留储能的作用。",
        start="2025-07-08 00:00:00",
        days=7,
        seed=2031,
        stress_type="heat_load",
    ),
    ExperimentScenario(
        scenario_id="S3",
        name="价格尖峰场景",
        description="增强市场价格尖峰文本事件和晚高峰高价，用于测试语义增强策略的套利与风险控制能力。",
        start="2025-07-15 00:00:00",
        days=7,
        seed=2042,
        stress_type="price_spike",
    ),
    ExperimentScenario(
        scenario_id="S4",
        name="新能源消纳压力场景",
        description="增强午间光伏出力和新能源消纳提示，用于测试储能对弃光惩罚的响应。",
        start="2025-07-22 00:00:00",
        days=7,
        seed=2053,
        stress_type="renewable_curtailment",
    ),
]


BASELINE_MODELS = [
    {
        "model_id": "rule",
        "name": "Rule-Based",
        "description": "固定阈值策略，低价/高光伏充电，高价放电。",
        "chapter": "中期与正式实验基线",
    },
    {
        "model_id": "softq_numeric",
        "name": "Soft-Q-Numeric",
        "description": "离散动作最大熵RL过渡基线，仅使用数值状态。",
        "chapter": "中期基线",
    },
    {
        "model_id": "softq_semantic",
        "name": "Soft-Q-Semantic",
        "description": "离散动作最大熵RL过渡基线，使用数值状态和语义状态。",
        "chapter": "中期基线",
    },
    {
        "model_id": "sac_numeric",
        "name": "SAC-Numeric",
        "description": "连续动作SAC，仅使用数值状态。",
        "chapter": "正式核心基线",
    },
    {
        "model_id": "ledrl_sac",
        "name": "LE-DRL-SAC",
        "description": "本文核心方法，使用数值状态和文本语义增强状态。",
        "chapter": "正式核心方法",
    },
    {
        "model_id": "ledrl_without_text",
        "name": "LE-DRL w/o Text",
        "description": "文本消融模型，网络结构与LE-DRL一致但移除文本语义输入。",
        "chapter": "正式消融实验",
    },
    {
        "model_id": "milp",
        "name": "MILP Rolling Horizon",
        "description": "滚动优化上界或强基准，用于解释RL策略与优化方法差距。",
        "chapter": "正式强基准",
    },
]


METRICS = [
    ("total_reward_yuan", "总收益", "元", "经济性指标，越高越好"),
    ("mean_reward_yuan", "平均奖励", "元/步", "单位调度步收益"),
    ("cvar_5_yuan", "CVaR 5%", "元", "尾部风险指标，越高表示极端亏损越小"),
    ("battery_throughput_mwh", "储能吞吐量", "MWh", "储能使用强度"),
    ("high_price_discharge_rate", "高价放电率", "比例", "高电价时段放电动作占比"),
    ("low_price_charge_rate", "低价充电率", "比例", "低电价时段充电动作占比"),
    ("final_soc", "终止SOC", "比例", "期末储能状态"),
    ("event_count", "文本事件数", "次", "场景中文本事件触发数量"),
    ("violation_yuan", "约束惩罚", "元", "动作因SOC边界不可执行导致的惩罚"),
    ("curtailment_cost_yuan", "弃光惩罚", "元", "新能源消纳压力相关成本"),
]


def scenario_records() -> list[dict]:
    return [asdict(s) | {"periods": s.periods} for s in SCENARIOS]

