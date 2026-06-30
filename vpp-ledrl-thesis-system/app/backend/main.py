from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core.config import ScenarioConfig
from app.core.data import generate_china_vpp_scenario
from app.core.simulation import run_experiment, _records_for_json


APP_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = APP_ROOT / "web"

app = FastAPI(
    title="VPP LE-DRL Thesis System",
    description="大语言模型与深度强化学习融合的虚拟电厂动态优化与决策研究演示后端",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class ScenarioRequest(BaseModel):
    start: str = Field(default="2025-07-01 00:00:00")
    periods: int = Field(default=96 * 7, ge=96, le=96 * 60)
    freq: str = Field(default="15min")
    seed: int = Field(default=42)
    region: str = Field(default="广东省")
    policies: list[str] = Field(default_factory=lambda: ["rule", "ledrl", "random"])


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "service": "vpp-ledrl-thesis-system"}


@app.post("/api/scenario")
def scenario(req: ScenarioRequest):
    cfg = ScenarioConfig(start=req.start, periods=req.periods, freq=req.freq, seed=req.seed, region=req.region)
    df = generate_china_vpp_scenario(cfg)
    return {
        "scenario": cfg.__dict__,
        "rows": len(df),
        "preview": _records_for_json(df.head(200)),
        "event_counts": df["event_type"].value_counts().to_dict(),
    }


@app.post("/api/run")
def run(req: ScenarioRequest):
    cfg = ScenarioConfig(start=req.start, periods=req.periods, freq=req.freq, seed=req.seed, region=req.region)
    return run_experiment(policy_names=req.policies, scenario=cfg)


def main():
    uvicorn.run("app.backend.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
