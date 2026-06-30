from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.llm_provider import LLMProvider


def mask_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def main() -> None:
    provider = LLMProvider()
    config = {
        "AI_PROVIDER": provider.provider,
        "AI_BASE_URL": provider.base_url,
        "AI_API_KEY": mask_key(os.getenv("AI_API_KEY", "")),
        "AI_MODEL": provider.model,
        "AI_USE_RESPONSE_FORMAT": provider.use_response_format,
    }
    print(json.dumps({"config": config}, ensure_ascii=False, indent=2))
    if provider.provider == "local" or not provider.api_key:
        print("Remote AI is not configured. Current mode will use local-rule fallback.")
        return

    sample_text = "广东气象台发布高温橙色预警，预计晚高峰空调负荷显著上升，现货市场可能出现尖峰电价。"
    sample_context = {
        "scenario_id": "CHECK",
        "event_type": "高温与价格尖峰",
        "temperature_c": 36.5,
        "price_yuan_mwh": 580.0,
    }
    assessment = provider.assess_event(sample_text, context=sample_context, allow_fallback=False)
    payload = {
        "remote_ai_ok": True,
        "assessment": assessment.__dict__,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
