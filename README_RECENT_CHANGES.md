# 近期改动说明（2026-07-05）

这份文件记录了 2026-07-05 对 VPP LE-DRL-SAC 论文（`ieee_pkg/ieee_access_vpp_ledrl_20260630/main.tex`）的一轮审稿式修订。换电脑后先看这份，再对照 `main.tex` 继续。

## 一、论文文本改动（main.tex）

### 1. 货币单位统一
**问题**：德国 DE-LU 真实电价实验里，电价是 EUR/MWh，但 reward gap 却标成 yuan，审稿人会抓这个矛盾。

**改法**（按"reward 是相对 no-action 基线的优势值"口径）：
- 在 reward 定义处（`sec:data` 的 VPP Environment 小节）加了总说明：reward 以对应电价的计价货币为单位，S1–S5（yuan/MWh）记 yuan，德国 DE-LU（EUR/MWh）记 EUR，不做汇率换算。
- 德国实验 4 处 `yuan` → `EUR`：摘要、`Real-price validation` 段、`tab:real_price` 表标题与表头、结论。
- 修正了 `−11.8 vs −210 yuan/MWh` 的跨币种直接比较，改为各带单位（−11.8 EUR/MWh vs −210 yuan/MWh）并说明是"相对深度"。

### 2. 数字一致性
- 讨论段里陈旧的 `5,127 yuan / CI [+4,420, +5,655]` → 改为与摘要、贡献列表、`tab:s5_unseen` 一致的 `5,482 yuan / CI [+4,875, +6,060]`，对比对象统一为 `SAC-Numeric`。

### 3. S5 术语冲突
- `tab:gated_moe` 里自相矛盾的 `known-train` → 改为 `neg-price, in train` / Type 列 `neg-price`。
- 门控段正文补一句：S5 与 V1–V4 同属"先验无分支的负电价结构"，区别仅在 S5 在训练集内、V1–V4 留出。

### 4. 弱化 "first reported evidence" 强断言
- 摘要：`providing the first reported evidence` → `providing preliminary evidence ... can help`
- 讨论（`sec:unseen_event` Interpretation 段）：`the first reported evidence` → `to our knowledge this provides preliminary evidence ..., which future work on measured operational data should confirm`

### 5. 把 DE-LU 真实电价实验提到核心位置（让标题主张自证）
- 贡献列表第 3 条：补入 DE-LU 真实电价+真实天气结果，定位为"最接近部署的证据，标题 event-awareness 主张的实证锚点"。
- Discussion 数据有效性段：重写为"递增真实性阶梯"（合成 → 真实天气 → 真实电价+真实天气），DE-LU 为阶梯顶端，消除原文"真实数据是下一步"与已有真实实验的矛盾。
- 结论：DE-LU 从附属列举提升为 `Most importantly ... 最接近真实市场的证据 ... grounding the event-awareness claim of the title in real data`。

### 6. 补算法伪代码
- preamble 加 `\usepackage{algorithm}` + `\usepackage{algpseudocode}`。
- **算法 1**：LE-DRL-SAC 训练 + 语义安全层推理（Stage 1 LLM 缓存 / Stage 2 SAC 训练 / Stage 3 `Dispatch()` 推理），放在 IV-C 节末。
- **算法 2**：事件覆盖门控训练（Stage 1 两 expert 预训练 / Stage 2 per-batch 归一化 Q 差分目标 / Stage 3 BCE 训练门控 / Stage 4 `GatedDispatch()`），放在 IV-G 节。
- 用 `scripts/check_alg.py` 静态检查过：环境配对、For/Function 配对、`$` 闭合全部正确。

### 7. 修正 bootstrap 统计方法（**重要**）
**问题**：旧 `bootstrap_significance.py` 在 n=3 个 seed-级跨场景均值上做 bootstrap，CI 因此窄到不可信（`proposed−SAC-Numeric = [+2517.9, +2573.6]`，宽度仅 55）。审稿人会指出 n=3 bootstrap 无意义。

**改法**：
- 新建 `vpp-ledrl-thesis-system/scripts/bootstrap_stats_paired.py`，用 **seed-paired bootstrap**（每个 seed 内 proposed−baseline 的差值序列上重采样，3-seed 主比较 n=12 配对差值，5-seed S5 n=5），并加 Wilcoxon signed-rank 交叉验证。
- 论文 Discussion 段重写：说明 paired bootstrap 方法，新 CI `[+2060.4, +3043.8]`（宽度 983）替换旧窄 CI；点估计 2554 不变，结论（显著优于 SAC-Numeric，Wilcoxon p=0.0005）不倒。
- 5-seed S5/变体 CI 经核验与论文基本一致（`[+4875, +6060]`），微调了 V1/V4 的 CI 上限以对齐新算的值。

### 8. 补充 LLM 相关工作讨论
- `Language Models and Semantic Information in Power Systems` 小节扩写：把 b4（Majumder Joule 2024）、b5（Cheng Sci Reports 2025）、b6（Amjad IEEE Access 2025）三篇的具体贡献展开分析，而不是列表式引用。
- 加了一句范围限定："现有 LLM-for-power-systems 文献在此按架构模式综述，而非系统编目每个 LLM+dispatch 配对"——把"文献偏薄"转化为有意识的范围界定。

> ⚠️ **未做**：因网络受限无法在线核实，**没有**新加 bibitem。如要补 2024–2025 年 LLM+能源/LLM+RL 近作，需在能联网的电脑上自己找几篇真实存在的加进 `thebibliography`（当前编号到 b35）。**不要凭记忆编造引用**。

## 二、架构图重画

旧图（四层横带 + 三条交叉斜线）→ 新图（四泳道 (a)(b)(c)(d)、主流程水平、连线全部正交无斜线交叉、紫色专留给 gate、encoder 改棕色避免撞色、底部加 gate 语义说明）。

- 生成脚本：`ieee_pkg/ieee_access_vpp_ledrl_20260630/scripts/draw_architecture_clean.py`
- 输出：`ieee_pkg/ieee_access_vpp_ledrl_20260630/figures/ieee_vpp_ledrl_architecture.png` + `.svg`
- 旧图备份：`ieee_vpp_ledrl_architecture_OLD.png` / `.svg`
- 论文 caption 与新图结构已核对一致，无需改动。
- 生成命令：`python scripts/draw_architecture_clean.py`（需 matplotlib，环境已装 3.7.5）

## 三、统计脚本与输出（在 `vpp-ledrl-thesis-system/` 下）

| 文件 | 作用 |
|---|---|
| `scripts/bootstrap_stats_paired.py` | **新建**，修正版 paired bootstrap + Wilcoxon，重算 3-seed 主比较与 5-seed S5/变体 CI |
| `scripts/bootstrap_significance.py` | 旧版（n=3 seed-级 bootstrap），**保留但已不用**，论文已改用 paired 版结果 |
| `scripts/check_alg.py` | **新建**，静态检查 main.tex 里两个 algorithm 块的语法配对 |
| `outputs/chapter6_long/bootstrap_paired_3seed.csv` | 新生成，3-seed paired bootstrap 结果 |
| `outputs/chapter6_long/bootstrap_paired_5seed_s5.csv` | 新生成，5-seed S5/V1–V4 paired bootstrap 结果 |

> 注：`outputs/` 在 `.gitignore` 里，这些 csv **不会被 git 跟踪**。换电脑后要重新跑 `python scripts/bootstrap_stats_paired.py` 生成。脚本依赖 `pandas`、`scipy`。

## 四、还没做的审稿项（换电脑后继续）

1. **补 LLM 近年文献**（需联网）：相关工作 LLM 部分虽已扩写，但 bibitem 仍只有 b4/b5/b6 三篇。建议补 2–4 篇 2024–2025 真实存在的 LLM+能源/LLM+RL 近作，编号 b36 起。
2. **摘要精简**：现在 ~450 词，IEEE Access 上限通常 250 词，超标严重。投稿前必改。
3. **作者信息**：邮箱仍是 `to be added`，缺 `\begin{IEEEbiography}` 作者简介和照片。
4. **DE-LU 真实电价 5-seed 数据核查**：论文里 `[+593, +3,627]` 那个 CI，我没在 `outputs/` 下找到对应 csv，无法用 paired bootstrap 复核。换电脑后找一下生成这个 CI 的脚本，确认它用的是 paired 方法（5-seed S5 已验证 paired 方法与论文一致，可推断 DE-LU 也是同一套，但最好复核）。
5. **代码/数据可用性声明**：这篇没有，IEEE Access 看重。建议补一个 "Data and Code Availability" 小节。
6. **本地无 LaTeX 编译器**：所有 .tex 改动未做端到端编译验证。换电脑后用 Overleaf 编译一次，重点看 `algorithm` 浮动体位置和 `algpseudocode` 在 IEEE Access 模板下的缩进。

## 五、换电脑后的操作

```bash
# 1. 拉取
git pull origin main

# 2. 安装依赖（统计脚本用）
cd vpp-ledrl-thesis-system
pip install -r requirements.txt -r requirements-ml.txt   # 含 pandas / scipy / matplotlib

# 3. 复跑统计（生成 outputs 下的 csv，gitignore 不跟踪）
python scripts/bootstrap_stats_paired.py

# 4. 复画架构图（如需改动）
cd ../ieee_pkg/ieee_access_vpp_ledrl_20260630
python scripts/draw_architecture_clean.py

# 5. 用 Overleaf 编译 main.tex 确认无语法错
```

## 六、关键文件速查

- 论文：`ieee_pkg/ieee_access_vpp_ledrl_20260630/main.tex`
- 架构图脚本：`ieee_pkg/ieee_access_vpp_ledrl_20260630/scripts/draw_architecture_clean.py`
- 统计脚本：`vpp-ledrl-thesis-system/scripts/bootstrap_stats_paired.py`
- 3-seed 评估数据：`vpp-ledrl-thesis-system/outputs/chapter6_long/evaluation_by_seed.csv`
- 5-seed S5 数据：`vpp-ledrl-thesis-system/outputs/chapter6_long/s5_and_variants_5seed.csv`
- 安全层 sweep 数据：`vpp-ledrl-thesis-system/outputs/chapter6_long/prior_weight_sweep_by_seed.csv`
