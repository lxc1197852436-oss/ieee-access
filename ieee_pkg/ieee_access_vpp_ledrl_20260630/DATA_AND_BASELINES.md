# Data and Baseline Boundary for the IEEE Access Draft

## Data

The current manuscript uses a **public-data-calibrated 15-minute VPP dispatch dataset**. The current project metadata do not support describing it as verified private real-world VPP telemetry.

### Public disclosures used for calibration

- National electricity consumption and renewable-energy background statistics.
- Guangdong electricity market and independent energy-storage public disclosures.
- Publicly available market/event categories used to design weather, market-price, demand-response, and renewable-curtailment text events.

### 15-minute dispatch trajectories

The 15-minute model input sequence includes:

- `timestamp`
- `load_mw`
- `pv_mw`
- `price_yuan_mwh`
- `temperature_c`
- `event_type`
- `event_text`
- semantic risk features when AI semantic extraction is enabled

Correct manuscript wording:

> This study uses a public-data-calibrated 15-minute VPP dispatch dataset. Public disclosures calibrate the market and power-system background, while the dispatch trajectories support reproducible algorithm evaluation.

Incorrect manuscript wording:

> This study uses real Guangdong VPP 15-minute operating records.

## Strong Baselines

The draft now includes the following stronger baselines:

| Baseline | Status | Role |
|---|---|---|
| Rule-Based | implemented and evaluated | Strong deterministic operational baseline |
| Rolling-Horizon | implemented and evaluated | Solver-free rolling dynamic-programming/look-ahead baseline |
| Enhanced Rolling-Horizon | implemented and evaluated (`light_40_10`) | Engineering-regularized MPC-like baseline with terminal SOC, action smoothing, and cycling penalty |
| SAC-Numeric | implemented and evaluated | Numerical-state reinforcement-learning baseline |
| SAC-Numeric + numeric safety layer | implemented and evaluated (`numeric_guidance_100`) | Text-agnostic safety-layer ablation isolating mechanism vs semantics |
| LE-DRL w/o Text | implemented and evaluated | Text-ablation baseline with identical semantic input dimension |
| LE-DRL-SAC + semantic safety layer | implemented and evaluated | Final proposed controller; main result uses `w=0.9` with prior power 2.0 |
| Random | implemented and evaluated | Weak sanity-check baseline |
| Soft-Q-Numeric / Soft-Q-Semantic | implemented and evaluated | Transitional discrete-action RL baseline |
| Full MILP/MPC | not yet implemented | Recommended next strong baseline before final submission |

## Current Evidence

Cross-scenario average total reward:

| Model | Total reward (yuan) |
|---|---:|
| LE-DRL-SAC + semantic safety layer (`w=0.9`, power=2.0, DeepSeek) | -208,468.9 |
| Rule-Based | -208,969.9 |
| SAC-Numeric + numeric safety layer (`numeric_guidance_100`) | -209,125.3 |
| LE-DRL-SAC actor only (`w=0`) | -210,447.8 |
| Enhanced Rolling-Horizon (`light_40_10`) | -209,988.3 |
| Rolling-Horizon | -210,616.7 |
| SAC-Numeric | -211,023.2 |
| LE-DRL w/o Text | -211,023.1 |

Correct claim:

> LE-DRL-SAC with a semantic safety layer (`w=0.9`, prior power 2.0, DeepSeek LLM semantic encoder) outperforms numerical SAC, text-ablation SAC, a numeric-safety-layer SAC variant, Rule-Based control, the Rolling-Horizon optimizer, and the Enhanced Rolling-Horizon optimizer in average total reward over three seeds. Statistical uncertainty should be reported with the paired-bootstrap procedure in `scripts/bootstrap_stats_paired.py`, which resamples matched `(seed, scenario)` reward differences rather than three seed-level averages. Under this corrected procedure, the proposed controller exceeds SAC-Numeric by +2554.2 yuan with 95% CI[+2060.4,+3043.8] and Wilcoxon p=0.0005. The numeric safety layer already improves SAC-Numeric, which confirms the blending mechanism carries value; the language-enhanced controller still exceeds the numeric-safety-layer variant, so the residual gain is attributable to textual semantics rather than the mechanism alone. The pure learned actor alone is not sufficient; the semantic safety layer is necessary for the final result.

Incorrect claim:

> A pure LE-DRL-SAC actor without the semantic safety layer is the best overall dispatch method.

## Recommended Next Experiment Before Submission

For a stronger IEEE Access submission, add either:

- a full MILP or MPC rolling optimization baseline with explicit forecast horizon and constraints, or
- a tuned rolling-horizon baseline with terminal SOC penalty and degradation-aware action smoothing.

This would address the most likely reviewer concern: whether the proposed semantic RL method adds value beyond well-designed operational optimization.
