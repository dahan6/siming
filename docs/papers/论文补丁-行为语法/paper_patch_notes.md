# 论文补丁说明（paper patch notes）

2026-09-08 ｜ 对应审稿风险清单逐条处理
论文主文件：`paper_behavioral_grammar_detection.md`

---

## P1-1 基线对比节的两个测量条件（必须并列写清）

**插入位置**：§6 Evaluation 新增小节 "Baseline Comparison with DeepLog"（建议放在 6.1 四代对比之后）。

**可直接使用的英文段落（草稿）：**

> **Measurement conditions.** Two distinct evaluation conditions appear in this paper and must not be conflated:
> (i) *In-vocab closed-world*: the AAA's active-phase events on the target VM, where every event token is within the benign vocabulary — the main results (93% detection at 3.84% onboarding FPR, §6.1) are measured under this condition;
> (ii) *Open-vocab traces*: exported attack trajectories (rove/bee) containing out-of-vocabulary events (e.g., reconnaissance processes never seen in the benign host baseline; the rove trace set is 100% EXEC-type events versus 83% CONN in benign, with a process vocabulary of 37 versus 88). Under condition (ii), all full-grammar models reach 100% event-level detection — this number reflects distribution mismatch and the UNK mechanism, not model discrimination per se. We report the baseline comparison under both conditions: condition (ii) quantifies the representation advantage of the 8-token grammar (DeepLog-keys 76.7% vs 100% for full-grammar models), while condition (i) quantifies discrimination when the adversary achieves perfect vocabulary conformance.

**中文要点**：主结果 = VM 内全词表（93% @ 3.84% FPR）；100%/0% = 含词表外事件的导出轨迹。
两数字并列处必须带本段说明。

---

## P1-1b 硬子集分析（词表内检出率）⭐ 回应审稿人的决定性证据

审稿人可能质疑"100%/0% 是分布失配（OOV）撑起来的"。用**纯词表内子集**（只保留全部 token
都在良性词表内的事件）实测，该质疑对完整语法模型不成立：

| 架构 | rove 词表内检出 | bee 词表内检出 | 是否 OOV 驱动 |
|---|---|---|---|
| DeepLog-keys | **15.9%**（vs 全量 59.6%） | 56.3%（vs 68.1%） | 是（崩塌） |
| DeepLog-full | **99.8%**（vs 99.9%） | 96.3%（vs 97.3%） | 否 |
| TinyGPT（重训子采样） | **100%** | **100%** | 否 |
| **TinyGPT（生产模型 host-r3-clean, vocab 240）** | **100%**（词表内占 57.6%） | **100%**（占 80.9%） | 否 |

**结论**：真正依赖 OOV 的是 keys-only 基线（词表内 59.6%→15.9% 崩塌）。完整 8-token 语法
模型（无论 LSTM 还是 Transformer）在纯词表内事件上仍 ~100% 检出——它抓的是上下文字段
组合违规（高 NLL），不是词表外 token。生产模型在 bee 词表内子集（80.9% 事件）上 100% 检出、
分数 p50=21.8（τ=2.6，8.4× 阈值）。**这条把"分布失配"质疑转化为加分项。**

**建议入文英文句：**

> *OOV robustness.* To rule out that detection is driven merely by out-of-vocabulary tokens, we re-score attack traces on the in-vocabulary subset only (events whose tokens all appear in the benign vocabulary). Keys-only DeepLog collapses from 59.6% to 15.9% detection, confirming its reliance on OOV; full-grammar models maintain ~100% (LSTM-full 99.8%, TinyGPT 100%), and the production TinyGPT attains 100% on the 80.9% in-vocabulary subset of bee active-phase events (p50 surprise 21.8 vs τ=2.6). The grammar captures field-combination violations, not vocabulary membership.

数据：`results/invocab_analysis.json` + `results/invocab_production_model.json`
（脚本 `scripts/invocab_analysis.py` / `scripts/score_only_invocab.py`）。

---

## P1-2 图的英文版与 ADFA 宏平均标注

- 四张图已全部重出英文版：`figures/en/` 目录下同名文件（投稿用）。
- `fig_adfa_perclass` 右侧三条宏平均虚线标注已纵向拉开（+2.0 / −7.0 / −16.0 偏移），
  实测 FPR 注记移至左下角。
- 中文版保留在 `figures/`（内部用）。

---

## P2-1 数字口径统一（以实测 JSON 为准）

| 项 | 旧口径 | 本补丁实测 | 处理 |
|---|---|---|---|
| TinyGPT 参数量 | 0.88M（摘要） | **871,408（0.87M）**，vocab=240 | 摘要改 0.87M；或注明主实验 0.88M（词表 167）、对比实验 0.87M（词表 240） |
| TinyGPT rove 分数 p50 | README 曾写 19.4（seed0 单点） | **18.34（5-seed 均值）** | 引用一律用 18.3 |
| DeepLog-full rove p50 | 14.6 | **14.66**（5-seed 均值） | 确认无误 |
| DeepLog-keys rove p50 | 3.4（子采样） | **3.57**（全量 5-seed 均值） | 引用一律用 3.6 |
| 脚本名 | run_matrix_cloud.sh | 实际：`run_matrix_cloud.sh`（DeepLog 矩阵）、`run_adfa_matrix_cloud.sh`（ADFA）、`run_dist.sh`（分布） | 已修正 |

## P2-2 良性分布同质性（Limitations 必加一句）

实测（host_tokens_clean 前 100K 事件）：**95.1% 事件 PROC=ssh、83.3% ET=CONN**。
DeepLog-keys 的 0% FPR 部分受益于该同质性，非纯属模型能力。

**建议加入 Limitations（英文草稿）：**

> *Benign baseline homogeneity.* Our benign baseline is dominated by a narrow workload (95.1% of events are ssh-related connections, 83.3% are network-connect events), which inflates the apparent separability for all evaluated models — including the baselines' near-zero FPR. Detection claims under the open-vocab condition should be read with this caveat; the in-vocab closed-world results (§6.1) are the conservative reference.

---

## P3-1 ADFA 上的 DeepLog 参照（已实测）

ADFA-LD 上 LSTM（DeepLog 风格）参照，同 bigram 表示、同阈值纪律（良性 q99），3 seeds：

| 模型 | ADFA bigram 宏检出(融合) | FPR |
|---|---|---|
| TinyGPT（司命先验） | 31.56% ±0.32% (CI95 ±0.40%, n=5) | 1.83% |
| **DeepLog-LSTM 参照** | **32.16% ±2.13% (CI95 ±5.29%, n=3)** | 1.83% |

**结论**：公开基准上 LSTM 与 Transformer 差异不显著（32.2% vs 31.6%），与内部结论
"表示 > 架构"一致。Creech 2013 SOTA（语义 n-gram + one-class）约 60-70% @ FPR 1%，
差距来自特征工程（非连续 syscall 模式）而非序列模型架构。

数据：`results/adfa_lstm_skip0_seed{0,1,2}.json`，聚合并入 `aggregate_summary.json`。

## P3-2 更新数据集（future work 一句）

> Future work includes evaluation on OpTC (DARPA Transparent Computing) and DARPA TC Engagement 5, whose richer multi-host telemetry would stress-test the cross-machine generalization claims (§4.5).

---

## 附：本补丁产物清单

- `results/deeplog_{lstm_keys,lstm_full,tinygpt}_seed{0..4}.json` — 15 轮基线矩阵
- `results/adfa_{skip0,skip1,skip2}_seed{0..4}.json` — 15 轮 ADFA 矩阵
- `results/adfa_lstm_skip0_seed{0,1,2}.json` — ADFA 上 LSTM 参照（3 轮）
- `results/invocab_analysis.json` + `invocab_production_model.json` — 硬子集分析 ⭐
- `results/scores_*.npz` — 分数数组（含 UNK 标记）
- `results/aggregate_summary.json` — 全聚合（mean ± std + 95% CI）
- `figures/`（中文）+ `figures/en/`（英文，投稿用）
- `reports/aaa_calibration.json` — AAA 拟态量化
- `scripts/` — 全部可复跑脚本（含 invocab_analysis.py / score_only_invocab.py）
