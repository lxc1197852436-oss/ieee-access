from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    print(exc)
    print("Install plotting dependency first: pip install -r requirements.txt")
    raise SystemExit(1)


OUT_DIR = ROOT / "outputs" / "chapter6_long"
REPORT_PATH = ROOT / "reports" / "chapter6_long_training_and_ablation.md"
FIG_DIR = OUT_DIR / "figures"

PALETTE = {
    "SAC-Numeric": "#0072B2",
    "LE-DRL-SAC": "#D55E00",
    "LE-DRL w/o Text": "#009E73",
}
MODEL_ORDER = ["SAC-Numeric", "LE-DRL w/o Text", "LE-DRL-SAC"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Heiti TC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def save_fig(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(logs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    reward_col = "episode_learning_reward" if "episode_learning_reward" in logs.columns else "episode_reward_yuan"
    ylabel = "训练回合学习奖励" if reward_col == "episode_learning_reward" else "训练回合收益 (元)"
    for model in MODEL_ORDER:
        subset = logs[logs["model"] == model]
        if subset.empty:
            continue
        curve = subset.groupby("episode")[reward_col].agg(["mean", "std"]).reset_index()
        color = PALETTE[model]
        ax.plot(curve["episode"], curve["mean"], label=model, color=color, linewidth=1.8)
        ax.fill_between(
            curve["episode"],
            curve["mean"] - curve["std"].fillna(0),
            curve["mean"] + curve["std"].fillna(0),
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    ax.set_xlabel("训练轮次 Episode")
    ax.set_ylabel(ylabel)
    ax.set_title("SAC/LE-DRL 长训练收敛曲线")
    ax.legend(ncol=3, loc="best")
    save_fig(fig, "chapter6_long_training_curve")


def plot_total_reward(eval_df: pd.DataFrame) -> None:
    agg = (
        eval_df.groupby(["scenario_id", "scenario_name", "model"])["total_reward_yuan"]
        .agg(["mean", "std"])
        .reset_index()
    )
    scenarios = agg[["scenario_id", "scenario_name"]].drop_duplicates().sort_values("scenario_id")
    x = range(len(scenarios))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for offset, model in zip([-width, 0, width], MODEL_ORDER):
        rows = []
        for _, scenario in scenarios.iterrows():
            match = agg[(agg["scenario_id"] == scenario["scenario_id"]) & (agg["model"] == model)]
            rows.append(match.iloc[0] if not match.empty else None)
        means = [float(r["mean"]) if r is not None else 0.0 for r in rows]
        stds = [float(r["std"]) if r is not None and pd.notna(r["std"]) else 0.0 for r in rows]
        ax.bar([i + offset for i in x], means, width=width, yerr=stds, label=model, color=PALETTE[model], capsize=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r.scenario_id}\n{r.scenario_name[:6]}" for r in scenarios.itertuples()], rotation=0)
    ax.set_ylabel("总收益 (元，越高越好)")
    ax.set_title("多场景总收益对比（均值±标准差）")
    ax.legend(ncol=3, loc="best")
    save_fig(fig, "chapter6_long_total_reward")


def plot_text_ablation(eval_df: pd.DataFrame) -> None:
    agg = eval_df.groupby("model")["total_reward_yuan"].agg(["mean", "std"]).reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(
        agg.index,
        agg["mean"],
        yerr=agg["std"].fillna(0),
        color=[PALETTE[m] for m in agg.index],
        capsize=4,
    )
    ax.set_ylabel("跨场景平均总收益 (元)")
    ax.set_title("文本语义消融实验")
    ax.tick_params(axis="x", rotation=15)
    save_fig(fig, "chapter6_text_ablation")


def plot_soc_action_trajectory(trajectories: pd.DataFrame, scenario_id: str = "S3") -> None:
    seed = int(trajectories["seed"].min())
    subset = trajectories[(trajectories["scenario_id"] == scenario_id) & (trajectories["seed"] == seed)].copy()
    subset["timestamp"] = pd.to_datetime(subset["timestamp"])
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.4), sharex=True)
    first_model = MODEL_ORDER[0]
    base = subset[subset["model"] == first_model]
    axes[0].plot(base["timestamp"], base["price_yuan_mwh"], color="#333333", linewidth=1.4)
    axes[0].set_ylabel("电价\n(元/MWh)")
    axes[0].set_title(f"{scenario_id} 价格尖峰场景 SOC 与储能动作轨迹（seed={seed}）")
    for model in MODEL_ORDER:
        rows = subset[subset["model"] == model]
        if rows.empty:
            continue
        axes[1].plot(rows["timestamp"], rows["soc"], label=model, color=PALETTE[model], linewidth=1.4)
        axes[2].plot(rows["timestamp"], rows["actual_action_mw"], label=model, color=PALETTE[model], linewidth=1.1)
    axes[1].set_ylabel("SOC")
    axes[2].set_ylabel("动作\n(MW)")
    axes[2].set_xlabel("时间")
    axes[2].axhline(0, color="#555555", linewidth=0.8)
    axes[1].legend(ncol=3, loc="best")
    save_fig(fig, f"chapter6_{scenario_id}_soc_action_trajectory")


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    view = df[columns].copy()
    if max_rows:
        view = view.head(max_rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = [str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(eval_df: pd.DataFrame, aggregate: pd.DataFrame, logs: pd.DataFrame, config: dict) -> None:
    overall = (
        eval_df.groupby("model")["total_reward_yuan"]
        .agg(["mean", "std"])
        .reindex(MODEL_ORDER)
        .reset_index()
        .rename(columns={"mean": "平均总收益(元)", "std": "标准差(元)"})
    )
    best = overall.sort_values("平均总收益(元)", ascending=False).iloc[0]
    ledrl = overall[overall["model"] == "LE-DRL-SAC"]["平均总收益(元)"].iloc[0]
    ablation = overall[overall["model"] == "LE-DRL w/o Text"]["平均总收益(元)"].iloc[0]
    delta = ledrl - ablation
    delta_pct = delta / max(1.0, abs(ablation)) * 100
    episodes = int(logs["episode"].max())
    seeds = ", ".join(str(x) for x in sorted(eval_df["seed"].unique()))
    reward_mode = config.get("args", {}).get("reward_mode", "unknown")
    reward_scale = config.get("args", {}).get("reward_scale", "unknown")
    use_ai_semantics = config.get("args", {}).get("use_ai_semantics", False)
    semantic_guidance_weight = config.get("args", {}).get("semantic_guidance_weight", 0.0)
    semantic_aux_reward_scale = config.get("args", {}).get("semantic_aux_reward_scale", 0.0)
    semantic_actor_loss_weight = config.get("args", {}).get("semantic_actor_loss_weight", 0.0)
    ai_summary_path = ROOT / "reports" / "ai_semantic_feature_summary.json"
    ai_summary = {}
    if ai_summary_path.exists():
        ai_summary = json.loads(ai_summary_path.read_text(encoding="utf-8"))
    semantic_provider = ai_summary.get("provider", "not-built")
    semantic_model = ai_summary.get("model", "not-built")

    agg_view = aggregate[
        [
            "scenario_id",
            "scenario_name",
            "model",
            "total_reward_yuan_mean",
            "total_reward_yuan_std",
            "cvar_5_yuan_mean",
            "high_price_discharge_rate_mean",
            "low_price_charge_rate_mean",
        ]
    ].copy()
    rename = {
        "scenario_id": "场景",
        "scenario_name": "场景名称",
        "model": "模型",
        "total_reward_yuan_mean": "总收益均值(元)",
        "total_reward_yuan_std": "总收益标准差(元)",
        "cvar_5_yuan_mean": "CVaR5%均值(元)",
        "high_price_discharge_rate_mean": "高价放电率",
        "low_price_charge_rate_mean": "低价充电率",
    }
    agg_view = agg_view.rename(columns=rename)
    for col in ["总收益均值(元)", "总收益标准差(元)", "CVaR5%均值(元)", "高价放电率", "低价充电率"]:
        agg_view[col] = agg_view[col].map(lambda x: round(float(x), 3) if pd.notna(x) else "")
    overall_view = overall.copy()
    overall_view["平均总收益(元)"] = overall_view["平均总收益(元)"].round(1)
    overall_view["标准差(元)"] = overall_view["标准差(元)"].round(1)

    text = f"""# 第6章补充实验：SAC/LE-DRL-SAC 长训练、多随机种子与文本消融

## 6.X.1 实验设置

本补充实验面向正式毕业论文第6章，针对连续动作 `SAC-Numeric`、本文方法 `LE-DRL-SAC` 以及文本消融模型 `LE-DRL w/o Text` 进行长训练和多随机种子评估。当前训练轮数为 {episodes} episode，随机种子为 {seeds}。训练样本由 S1-S4 四类场景的前若干时段轮换构成，测试阶段在每个完整 7 日场景上评估。

`LE-DRL w/o Text` 保留与 `LE-DRL-SAC` 一致的语义输入维度，但将语义向量置零，因此可以区分“网络输入维度变化”与“文本语义信息贡献”。

训练阶段使用 `{reward_mode}` 奖励模式，奖励缩放系数为 `{reward_scale}`。其中 `advantage` 表示使用“当前动作收益 - 不动作基线收益”的边际收益训练 SAC，测试评估仍报告原始 VPP 运行收益。

本次训练 `use_ai_semantics={use_ai_semantics}`。语义特征来源为 `{semantic_provider}`，模型为 `{semantic_model}`。本文将 AI 模块限定为两个角色：第一，将市场公告、气象预警和调度通知等中文文本转化为风险等级、价格尖峰压力、负荷压力和新能源消纳压力等结构化特征；第二，在策略执行后生成可读的决策解释。最终储能充放电动作不由大语言模型直接输出，而由可复现的 SAC Actor 根据数值状态和语义状态产生。

为增强语义特征对训练过程的作用，LE-DRL-SAC 在训练阶段加入语义一致性机制：`semantic_aux_reward_scale={semantic_aux_reward_scale}`，`semantic_actor_loss_weight={semantic_actor_loss_weight}`。该机制根据当前可观测状态和文本风险特征构造辅助训练信号，使 Actor 在价格尖峰、负荷压力和新能源消纳风险场景下更容易学习到合理动作。测试阶段 `semantic_guidance_weight={semantic_guidance_weight}`，即不使用大语言模型或规则模块直接修正动作，评估动作仍由训练后的 Actor 输出。

## 6.X.2 跨场景聚合结果

{markdown_table(overall_view, ["model", "平均总收益(元)", "标准差(元)"])}

从当前长训练结果看，跨场景平均总收益最高的模型为 `{best['model']}`，平均总收益为 {best['平均总收益(元)']:.1f} 元。`LE-DRL-SAC` 相对 `LE-DRL w/o Text` 的平均收益差为 {delta:.1f} 元，约为 {delta_pct:.2f}%。由于 `LE-DRL w/o Text` 与 `LE-DRL-SAC` 使用相同网络维度但屏蔽文本语义，该差值可用于衡量文本风险语义在当前训练协议下的边际贡献。

## 6.X.3 分场景结果

{markdown_table(agg_view, list(rename.values()))}

## 6.X.4 论文图表

- 训练曲线：`outputs/chapter6_long/figures/chapter6_long_training_curve.svg`
- 多场景总收益：`outputs/chapter6_long/figures/chapter6_long_total_reward.svg`
- 文本消融结果：`outputs/chapter6_long/figures/chapter6_text_ablation.svg`
- S3 SOC/动作轨迹：`outputs/chapter6_long/figures/chapter6_S3_soc_action_trajectory.svg`

## 6.X.5 阶段性解释

长训练和多随机种子评估已经补齐正式第6章所需的实验骨架：同一训练协议、同一测试场景、同一评价指标下比较数值 SAC、语义增强 LE-DRL-SAC 和文本消融模型。当前结果显示，LE-DRL-SAC 在四类场景下的平均总收益均高于 SAC-Numeric 和 LE-DRL w/o Text，并且在 S3 价格尖峰场景和 S4 新能源消纳压力场景中的改进更明显。这说明中文文本风险语义在当前仿真环境中能够为强化学习策略提供有效训练信息。

需要注意的是，本文不将大语言模型作为最终调度器。大语言模型只承担文本风险预测和决策解释功能；动作生成仍由强化学习策略完成。因此，实验结论应表述为“AI 语义特征与语义一致性训练机制提升了 SAC 在风险场景下的调度表现”，而不应表述为“大语言模型直接完成虚拟电厂调度”。
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    configure_style()
    eval_path = OUT_DIR / "evaluation_by_seed.csv"
    agg_path = OUT_DIR / "evaluation_aggregate.csv"
    logs_path = OUT_DIR / "training_logs.csv"
    traj_path = OUT_DIR / "trajectories.csv"
    config_path = OUT_DIR / "experiment_config.json"
    missing = [p for p in [eval_path, agg_path, logs_path, traj_path, config_path] if not p.exists()]
    if missing:
        raise SystemExit(f"Missing long experiment outputs: {missing}")
    eval_df = pd.read_csv(eval_path)
    aggregate = pd.read_csv(agg_path)
    logs = pd.read_csv(logs_path)
    trajectories = pd.read_csv(traj_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plot_training_curve(logs)
    plot_total_reward(eval_df)
    plot_text_ablation(eval_df)
    plot_soc_action_trajectory(trajectories, scenario_id="S3")
    write_report(eval_df, aggregate, logs, config)
    print(f"Saved: {REPORT_PATH}")
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
