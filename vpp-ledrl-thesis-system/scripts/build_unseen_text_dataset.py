"""Build unseen event-expression test set and evaluate LLM generalization.

Training uses five fixed event templates. To test whether the DeepSeek
semantic encoder generalizes beyond rote template matching, this script:

1. Defines ~10 unseen expressions per event category (different wording,
   same underlying operational meaning).
2. Queries DeepSeek for each unseen expression and caches the scores.
3. Builds an unseen-text OOD scenario whose numerical trajectory reuses the
   real Guangzhou weather OOD data, but whose event_text is replaced by the
   unseen expressions.
4. Saves the enriched unseen-text dataset for downstream evaluation.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.llm_provider import LLMProvider
from scripts.run_chapter6_experiments import scenario_data
from scripts.build_ood_dataset import OOD_SCENARIOS, load_real_weather, build_ood_scenario, WEATHER_CSV

OUT_FEATURES = ROOT / "data" / "processed" / "unseen_event_semantic_features.csv"
OUT_SCENARIOS = ROOT / "data" / "processed" / "unseen_text_scenarios.csv"
RAW_DIR = ROOT / "data" / "raw_sources" / "ai_semantic_cache"

# Unseen expressions: same operational meaning, different wording from training.
UNSEEN_EVENTS = {
    "正常运行": [
        "系统当前处于平稳运行状态，无异常气象或市场信号。",
        "当前调度时段未触发任何预警或邀约，按常规模式执行。",
        "运行状态正常，负荷与电价均在预期区间内波动。",
        "无极端天气与市场异动，维持基础调度策略。",
        "日内无突发事件，各资源出力稳定。",
    ],
    "高温预警": [
        "南方区域持续高温，空调负荷预计在晚高峰快速攀升。",
        "气象部门提示未来数小时出现极端炎热天气，居民用电负荷可能显著增加。",
        "多地发布高温红色预警，降温负荷叠加晚高峰，整体用电需求走高。",
        "受高温影响，午后至夜间负荷曲线明显抬升，需防范供需偏紧。",
        "炎热天气持续，空调类负荷占比上升，运行备用面临压力。",
    ],
    "需求响应": [
        "交易中心发起削峰邀约，请聚合商在晚高峰时段降低用电。",
        "调度启动可调负荷响应，鼓励储能与柔性负荷参与晚间顶峰。",
        "为缓解晚高峰供应紧张，现开展需求响应调用，参与方可获补偿。",
        "电力供需偏紧，建议负荷集成商在高峰段主动压降用电。",
        "今日执行需求响应事件，请于傍晚高峰减少非关键负荷。",
    ],
    "价格尖峰": [
        "日前市场出清价格出现异常抬升，晚高峰存在尖峰风险。",
        "现货价格波动加剧，晚间时段报价明显高于平日。",
        "市场公告显示近期结算价格走高，建议把握高价放电窗口。",
        "预计晚间现货价格大幅上涨，储能宜预留容量待价而沽。",
        "受供需与天气影响，日前出清价格在夜间达到峰值。",
    ],
    "新能源消纳": [
        "午间光伏出力充裕，存在弃光风险，建议储能优先吸纳。",
        "新能源出力大于本地负荷，调度提示提升午间充电以缓解消纳压力。",
        "日照充足导致光伏过剩，需调用储能减少弃电。",
        "午间可再生发电富余，鼓励增加储能充电消纳绿电。",
        "光伏高位运行，消纳形势紧张，建议午间加强储能吸纳。",
    ],
}


class ProgressBar:
    def __init__(self, total, label="", width=28):
        self.total = max(1, total)
        self.label = label
        self.width = width
        self.count = 0
        self.start = time.time()

    def update(self, n=1):
        self.count = min(self.total, self.count + n)
        elapsed = time.time() - self.start
        frac = self.count / self.total
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self.count / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.count) / rate if rate > 0 else 0.0
        sys.stdout.write(
            f"\r{self.label} [{bar}] {self.count}/{self.total} "
            f"({frac*100:5.1f}%) {elapsed:5.1f}s elapsed, eta {eta:5.1f}s  "
        )
        sys.stdout.flush()

    def finish(self):
        elapsed = time.time() - self.start
        sys.stdout.write(f"\r{self.label} [{'#'*self.width}] {self.total}/{self.total} (100.0%) {elapsed:5.1f}s done\n")
        sys.stdout.flush()


def collect_unseen_rows() -> list[dict]:
    provider = LLMProvider()
    rows = []
    raw_rows = []
    all_texts = []
    for event_type, texts in UNSEEN_EVENTS.items():
        for text in texts:
            all_texts.append((event_type, text))

    bar = ProgressBar(len(all_texts), label="Unseen LLM")
    for event_type, text in all_texts:
        assessment = provider.assess_event(text, context={"event_type": event_type})
        rows.append(
            {
                "event_type": event_type,
                "event_text": text,
                "ai_risk_score": assessment.risk_score,
                "ai_price_spike_score": assessment.price_spike_score,
                "ai_load_pressure_score": assessment.load_pressure_score,
                "ai_renewable_curtailment_score": assessment.renewable_curtailment_score,
                "ai_recommended_storage_bias": assessment.recommended_storage_bias,
                "ai_event_summary": assessment.event_summary,
                "ai_explanation": assessment.explanation,
                "ai_provider": assessment.provider,
                "ai_model": assessment.model,
            }
        )
        raw_rows.append({"event_type": event_type, "event_text": text, "assessment": rows[-1]})
        bar.update()
        print(
            f"\n  {event_type} risk={assessment.risk_score:.2f} price={assessment.price_spike_score:.2f} "
            f"load={assessment.load_pressure_score:.2f} renewable={assessment.renewable_curtailment_score:.2f}",
            flush=True,
        )
    bar.finish()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "unseen_event_semantic_features.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in raw_rows), encoding="utf-8"
    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_unseen_scenarios(features: pd.DataFrame) -> pd.DataFrame:
    """Build unseen-text scenarios on top of the real-weather OOD trajectories.

    Each event slot is replaced by a rotating unseen expression of the same
    category, then enriched with the cached DeepSeek scores.
    """
    frames = []
    for scenario_id, name, stress, seed, periods in OOD_SCENARIOS:
        weather = load_real_weather(WEATHER_CSV, "2024-07-01", periods)
        df = build_ood_scenario(weather, stress, seed)
        # Replace event_text at event slots with unseen expressions of same type.
        # Stress type -> training event_type mapping (see apply_stress).
        stress_to_type = {
            "normal": "正常运行",
            "heat_load": "高温预警",
            "price_spike": "价格尖峰",
            "renewable_curtailment": "新能源消纳",
        }
        target_type = stress_to_type.get(stress, "正常运行")
        unseen_pool = features[features["event_type"] == target_type].reset_index(drop=True)
        if unseen_pool.empty:
            unseen_pool = features[features["event_type"] == "正常运行"].reset_index(drop=True)
        event_mask = df["event_type"] != "正常运行"
        slot_idx = 0
        for i in df.index[event_mask].tolist():
            row = unseen_pool.iloc[slot_idx % len(unseen_pool)]
            df.at[i, "event_type"] = row["event_type"]
            df.at[i, "event_text"] = row["event_text"]
            slot_idx += 1
        # Also rotate normal slots among unseen normal expressions.
        normal_pool = features[features["event_type"] == "正常运行"].reset_index(drop=True)
        if not normal_pool.empty:
            nidx = 0
            for i in df.index[~event_mask].tolist():
                df.at[i, "event_text"] = normal_pool.iloc[nidx % len(normal_pool)]["event_text"]
                nidx += 1
        df["scenario_id"] = scenario_id
        df["scenario_name"] = name.replace("(真实气象)", "(unseen 文本)")
        df["stress_type"] = stress
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    # Enrich with cached AI scores.
    keep = [
        "event_type",
        "event_text",
        "ai_risk_score",
        "ai_price_spike_score",
        "ai_load_pressure_score",
        "ai_renewable_curtailment_score",
        "ai_recommended_storage_bias",
        "ai_event_summary",
        "ai_explanation",
        "ai_provider",
        "ai_model",
    ]
    enriched = combined.merge(features[keep], on=["event_type", "event_text"], how="left")
    return enriched


def main() -> None:
    rows = collect_unseen_rows()
    write_csv(OUT_FEATURES, rows)
    features = pd.DataFrame(rows)
    print(f"\nUnseen feature rows: {len(features)}")
    scenarios = build_unseen_scenarios(features)
    scenarios.to_csv(OUT_SCENARIOS, index=False, encoding="utf-8-sig")
    print(f"Unseen-text scenario rows: {len(scenarios)}  Saved: {OUT_SCENARIOS}")
    print(f"Unseen feature file: {OUT_FEATURES}")


if __name__ == "__main__":
    main()
