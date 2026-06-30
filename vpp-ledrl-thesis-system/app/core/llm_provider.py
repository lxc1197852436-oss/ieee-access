from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

from app.core.semantic import LocalSemanticEncoder


@dataclass
class AIPrediction:
    risk_level: str
    predicted_price_trend: str
    suggested_action: str
    explanation: str


@dataclass
class AISemanticAssessment:
    risk_score: float
    price_spike_score: float
    load_pressure_score: float
    renewable_curtailment_score: float
    recommended_storage_bias: float
    event_summary: str
    explanation: str
    provider: str
    model: str


class LLMProvider:
    """OpenAI-compatible provider with local fallback.

    Later, when an API key is supplied, this class can call a mainstream AI
    model for event-risk prediction and natural-language decision explanation.
    No key is required for the current demo.
    """

    def __init__(self):
        load_dotenv()
        self.provider = os.getenv("AI_PROVIDER", "local")
        self.base_url = os.getenv("AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("AI_API_KEY", "")
        self.model = os.getenv("AI_MODEL", "gpt-4.1-mini")
        self.use_response_format = os.getenv("AI_USE_RESPONSE_FORMAT", "true").lower() not in {"0", "false", "no"}
        self.local_encoder = LocalSemanticEncoder()

    def predict(self, state: dict, action_mw: float) -> AIPrediction:
        if self.provider != "local" and self.api_key:
            try:
                return self._predict_remote(state, action_mw)
            except Exception as exc:
                return self._predict_local(state, action_mw, fallback_error=str(exc))
        return self._predict_local(state, action_mw)

    def _predict_local(self, state: dict, action_mw: float, fallback_error: str | None = None) -> AIPrediction:
        signal = self.local_encoder.encode(state.get("event_text", ""))
        risk_level = "高" if signal.risk_score >= 0.65 else "中" if signal.risk_score >= 0.3 else "低"
        if signal.price_spike_score > 0.4 or state.get("price_yuan_mwh", 0) > 520:
            trend = "上行或尖峰风险"
        elif signal.renewable_curtailment_score > 0.4:
            trend = "午间低价或消纳压力"
        else:
            trend = "常规波动"

        if action_mw > 0.2:
            suggested = "放电/售电"
        elif action_mw < -0.2:
            suggested = "充电/吸纳新能源"
        else:
            suggested = "保持观望"

        explanation = (
            f"本地规则预测：风险等级为{risk_level}，价格趋势为{trend}。"
            f"当前动作建议为{suggested}，依据是：{signal.explanation_hint}。"
        )
        if fallback_error:
            explanation += f" 远程AI调用失败，已回退到本地预测器：{fallback_error}"
        return AIPrediction(risk_level, trend, suggested, explanation)

    def assess_event(
        self,
        event_text: str,
        context: dict | None = None,
        allow_fallback: bool = True,
    ) -> AISemanticAssessment:
        """Convert Chinese event text into numeric features for LE-DRL.

        The remote branch is intentionally OpenAI-compatible, so the same code
        can work with OpenAI, DeepSeek, Qwen-compatible gateways, or a private
        proxy as long as they expose /chat/completions.
        """
        if self.provider != "local" and self.api_key:
            try:
                return self._assess_event_remote(event_text, context or {})
            except Exception as exc:
                if allow_fallback:
                    return self._assess_event_local(event_text, fallback_error=str(exc))
                raise
        if not allow_fallback:
            raise RuntimeError("Remote AI is not configured. Set AI_PROVIDER, AI_BASE_URL, AI_API_KEY, and AI_MODEL.")
        return self._assess_event_local(event_text)

    def _assess_event_local(self, event_text: str, fallback_error: str | None = None) -> AISemanticAssessment:
        signal = self.local_encoder.encode(event_text)
        explanation = f"本地规则语义评分：{signal.explanation_hint}。"
        if fallback_error:
            explanation += f" 远程AI调用失败，已回退到本地评分：{fallback_error}"
        bias = signal.renewable_curtailment_score - signal.price_spike_score
        if signal.load_pressure_score > 0.4:
            bias -= 0.2
        return AISemanticAssessment(
            risk_score=signal.risk_score,
            price_spike_score=signal.price_spike_score,
            load_pressure_score=signal.load_pressure_score,
            renewable_curtailment_score=signal.renewable_curtailment_score,
            recommended_storage_bias=float(max(-1.0, min(1.0, bias))),
            event_summary=event_text[:80] if event_text else "正常运行",
            explanation=explanation,
            provider="local",
            model="local-rule",
        )

    def _predict_remote(self, state: dict, action_mw: float) -> AIPrediction:
        prompt = (
            "你是虚拟电厂调度助手。请基于状态和文本事件预测风险、价格趋势，并解释动作。"
            "只返回JSON，字段为 risk_level, predicted_price_trend, suggested_action, explanation。"
            f"\n状态: {json.dumps(_jsonable_state(state), ensure_ascii=False)}"
            f"\n动作MW: {action_mw:.3f}"
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是电力系统和虚拟电厂优化调度专家。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed = parse_json_content(content)
        return AIPrediction(
            risk_level=str(parsed.get("risk_level", "未知")),
            predicted_price_trend=str(parsed.get("predicted_price_trend", "未知")),
            suggested_action=str(parsed.get("suggested_action", "未知")),
            explanation=str(parsed.get("explanation", content)),
        )

    def _assess_event_remote(self, event_text: str, context: dict) -> AISemanticAssessment:
        prompt = (
            "请把中文虚拟电厂运行文本转成强化学习可用的结构化语义特征。"
            "只返回JSON，不要返回Markdown。字段必须包括："
            "risk_score, price_spike_score, load_pressure_score, renewable_curtailment_score, "
            "recommended_storage_bias, event_summary, explanation。"
            "所有score范围为0到1；recommended_storage_bias范围为-1到1，"
            "-1表示倾向放电或保留高价套利能力，1表示倾向充电或吸纳新能源，0表示中性。"
            f"\n文本事件: {event_text or '正常运行'}"
            f"\n上下文: {json.dumps(context, ensure_ascii=False)}"
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是电力市场、虚拟电厂和需求响应调度专家。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        if self.use_response_format:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed = parse_json_content(content)
        return AISemanticAssessment(
            risk_score=_clamp01(parsed.get("risk_score", 0.0)),
            price_spike_score=_clamp01(parsed.get("price_spike_score", 0.0)),
            load_pressure_score=_clamp01(parsed.get("load_pressure_score", 0.0)),
            renewable_curtailment_score=_clamp01(parsed.get("renewable_curtailment_score", 0.0)),
            recommended_storage_bias=_clamp(parsed.get("recommended_storage_bias", 0.0), -1.0, 1.0),
            event_summary=str(parsed.get("event_summary", event_text[:80] if event_text else "正常运行")),
            explanation=str(parsed.get("explanation", "")),
            provider=self.provider,
            model=self.model,
        )


def _jsonable_state(state: dict) -> dict:
    keep = {
        "timestamp",
        "hour",
        "load_mw",
        "pv_mw",
        "price_yuan_mwh",
        "temperature_c",
        "event_type",
        "event_text",
        "soc",
    }
    return {k: str(v) if k == "timestamp" else v for k, v in state.items() if k in keep}


def _clamp01(value) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value, low: float, high: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = low
    return float(max(low, min(high, numeric)))


def parse_json_content(content: str) -> dict:
    text = (content or "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"AI response did not contain valid JSON: {text[:300]}")
