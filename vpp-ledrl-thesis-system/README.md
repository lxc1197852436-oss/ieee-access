# VPP LE-DRL Thesis System

面向硕士论文《大语言模型与深度强化学习融合的虚拟电厂动态优化与决策研究》的实验与展示项目包。

第一版目标：

- 本地生成“中国虚拟电厂”风格的可复现实验数据。
- 提供 VPP 储能调度仿真环境。
- 提供 Rule-based、SAC 占位策略、LE-DRL 语义增强策略接口。
- 预留主流 AI API 接入点，用于文本事件风险预测与调度解释。
- 提供 FastAPI 后端和静态 Dashboard 展示面。

## 快速运行

```bash
cd "/Users/xingchengli/Documents/New project/vpp-ledrl-thesis-system"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py
python -m app.backend.main
```

打开浏览器访问：

```text
http://127.0.0.1:8000
```

## 后续接入 AI API

复制环境变量模板：

```bash
cp .env.example .env
```

后续填入 API key：

```text
AI_PROVIDER=openai_compatible
AI_BASE_URL=https://api.openai.com/v1
AI_API_KEY=你的key
AI_MODEL=gpt-4.1-mini
AI_USE_RESPONSE_FORMAT=true
```

当前没有 key 时，系统会使用本地规则预测器，不影响演示和论文实验流程。

## 生成真正的 LLM 语义特征

论文题目中的“大语言模型融合”应体现在文本事件理解进入强化学习状态。流程如下：

1. 在 `.env` 中填写主流 AI 的 OpenAI-compatible 接口信息，不要把 key 写入代码、论文或截图。
2. 先测试 AI API 是否可用：

```bash
source .venv/bin/activate
python scripts/check_ai_provider.py
```

如果某些兼容接口不支持 `response_format`，把 `.env` 中的 `AI_USE_RESPONSE_FORMAT` 改为 `false` 后再测试。

3. 生成 AI 语义特征：

```bash
python scripts/build_ai_semantic_features.py
```

该脚本会输出：

- `data/processed/ai_event_semantic_features.csv`
- `data/processed/chapter6_ai_semantic_scenarios.csv`
- `reports/ai_semantic_feature_summary.json`

4. 使用 AI 语义特征重新训练第六章模型：

```bash
python scripts/train_chapter6_long_sac.py \
  --episodes 50 \
  --seeds 2026,2031,2042 \
  --train-periods-per-scenario 288 \
  --update-interval 8 \
  --batch-size 64 \
  --warmup-steps 256 \
  --hidden-dim 64 \
  --reward-mode advantage \
  --reward-scale 0.01 \
  --use-ai-semantics

python scripts/generate_chapter6_long_report.py
```

此时 `LE-DRL-SAC` 使用的是 AI API 生成的文本语义评分，`LE-DRL w/o Text` 是文本消融模型，`SAC-Numeric` 是不使用文本的数值基线。
