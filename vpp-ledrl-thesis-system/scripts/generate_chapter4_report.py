from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "outputs" / "chapter4" / "chapter4_sac_results.json"
REPORT = ROOT / "reports" / "chapter4_training_status.md"


def main() -> None:
    if not RESULT.exists():
        raise FileNotFoundError(f"Run scripts/train_chapter4_sac.py first: {RESULT}")
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = []
    for agent in data["agents"]:
        m = agent["metrics"]
        rows.append(
            {
                "name": agent["name"],
                "total": m["total_reward_yuan"],
                "cvar": m["cvar_5_yuan"],
                "final_soc": m["final_soc"],
                "throughput": m["battery_throughput_mwh"],
                "high_discharge": m["high_price_discharge_rate"],
                "low_charge": m["low_price_charge_rate"],
            }
        )

    lines = [
        "# 第4章 SAC/LE-DRL 训练状态报告",
        "",
        "## 训练说明",
        "",
        "本报告来自 `scripts/train_chapter4_sac.py` 的短训练结果。当前运行目的是验证连续动作 SAC 与 LE-DRL-SAC 训练闭环，而不是最终论文数值结论。正式第6章实验需要增加训练轮数、随机种子和消融实验。",
        "",
        f"- 数据集：`{data['dataset']}`",
        f"- 训练样本数：{data['train_rows']}",
        f"- 测试样本数：{data['test_rows']}",
        "",
        "## 测试集结果",
        "",
        "| 模型 | 总收益(元) | CVaR 5%(元) | 终止SOC | 储能吞吐(MWh) | 高价放电率 | 低价充电率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['total']:.2f} | {r['cvar']:.2f} | {r['final_soc']:.3f} | {r['throughput']:.2f} | {r['high_discharge']:.3f} | {r['low_charge']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 阶段性解释",
            "",
            "短训练结果显示，LE-DRL-SAC 已经可以利用语义增强状态完成连续动作调度，并在当前短训练设置下取得优于 SAC-Numeric 的总收益与尾部风险指标。但从终止 SOC 和高价放电率看，策略仍未充分收敛，存在过度保留电量或动作分布偏移问题。",
            "",
            "下一步需要：",
            "",
            "1. 将训练轮数从短训练提升到 50-100 episode。",
            "2. 增加自动温度系数或调小固定熵系数。",
            "3. 增加多随机种子实验。",
            "4. 增加 LE-DRL w/o Text 消融。",
            "5. 与 Rule-Based、Soft-Q、MILP rolling horizon 同表比较。",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {REPORT}")


if __name__ == "__main__":
    main()

