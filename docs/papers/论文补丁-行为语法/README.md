# 论文补丁包 · 行为语法检测器（司命）

日期：2026-09-11 ｜ 用途：补齐初投前的学术短板 + 实验补充方案
关联论文：`~/share/paper_behavioral_grammar_detection.md`

## 补丁对照（两轮）

### 第一轮（2026-09-08）：四条硬伤

| 审稿风险 | 补丁 | 证据位置 |
|---|---|---|
| 外部基线对比 = 0 | DeepLog（LSTM）双配置 vs TinyGPT，3 架构 × 5 seed | `figures/fig_baseline_comparison.png` + `results/deeplog_*.json` |
| 无公开数据集 | ADFA-LD 3 变体 × 5 seed（含 skip-gram 扩展） | `figures/fig_adfa_perclass.png` + `results/adfa_*.json` |
| 统计严谨性 | 全部关键数字 5-seed mean ± std + 95% CI | `results/aggregate_summary.json` |
| 对抗强度校准 | AAA 拟态量化（逐字段 KL + 时序分布迁移） | `figures/fig_aaa_mimicry.png` + `reports/aaa_calibration.json` |

### 第二轮（2026-09-11，实验补充方案 v1.0）：E1/E3/E4

| 实验 | 科学问题 | 核心结论 | 报告 |
|---|---|---|---|
| **E1** CV 失效边界 | CV 可被绕过吗？绕过后暴露多少？ | CV 漏洞真实（σ≥1.25 逃逸），但聚合语法面 p≈1e-9 仍检出——多面融合必要性实测证明 | `reports/e1_cv_boundary_range.md` |
| **E3** 异质负载基线 | 结论在负载多样性下成立吗？ | LOVO 跨机 FPR=0%，逐角色检出 3/3——不过拟合单一负载；web 训迁移差印证"必须本机自训" | `reports/e3_heterogeneous_workload.md` |
| **E4** OpTC 公开基准 | 公开多主机数据上检测率/泛化？ | 3 红队场景检出 22~87%（均值 63%），良性 FPR=0%，跨日漂移 5-24% | `reports/e4_optc_benchmark.md` |

**E2（敌手变体统计纪律）**：红线内，待红方跑变体训练后做检测侧评估。

## 关键数字（可直接引用）

### 基线对比（良性=host_tokens_clean 884K 事件；攻击=rove 70K + bee_active 37K 轨迹）

| 架构 | 参数 | AAA rove 检出(EWMA) | AAA bee 检出(EWMA) | FPR(EWMA) |
|---|---|---|---|---|
| DeepLog-keys (LSTM, 事件键) | 0.30M | 76.7% ±9.0% (CI95 ±11.2%) | 93.1% ±1.6% | 0.00% ±0.00% |
| DeepLog-full (LSTM, 8-token) | 0.33M | 100.0% ±0.0% | 100.0% ±0.0% | 0.00% ±0.01% |
| TinyGPT (Transformer, 8-token) | 0.87M | 100.0% ±0.0% | 100.0% ±0.0% | 0.00% ±0.01% |

**核心论点**：检测能力差异由表示驱动而非架构——事件键表示下 AAA 惊讶度 p50=3.6
（5-seed 均值），8-token 语法下 LSTM 14.7 / Transformer 18.3（单调递增）。LSTM 在
完整语法表示上即可达到 100%，证明"行为语法"表示本身贡献了主要检测增益；
Transformer 提供更强的分数分离（fig_score_distribution.png 三面板可见）。

参数量注记：本对比实验 TinyGPT 为 871,408 参数（词表 240）；论文主实验模型为
0.88M（词表 167）。两处词表不同导致参数略异，论文中引用时保持同一口径。

**诚实注意（写论文时必带）**：本对比的攻击轨迹含词表外事件；论文消融中 prior-only
0.4% 的测量是在 VM 内 bee 活跃期、全词表内事件上做的。两个数字测量条件不同，
需在文中说明，避免审稿人误解为矛盾。

### 硬子集分析（词表内检出率，回应"分布失配"质疑）⭐ 关键

审稿人可能质疑 100% 检出是 OOV（词表外）撑起来的。实测**纯词表内子集**：

| 架构 | rove 词表内检出 | bee 词表内检出 | 是否 OOV 驱动 |
|---|---|---|---|
| DeepLog-keys | **15.9%**（vs 全量 59.6%） | 56.3%（vs 68.1%） | 是（崩塌） |
| DeepLog-full | **99.8%**（vs 99.9%） | 96.3%（vs 97.3%） | 否 |
| TinyGPT（重训） | **100%** | **100%** | 否 |
| **TinyGPT（生产模型 host-r3-clean）** | **100%**（词表内占 57.6%） | **100%**（占 80.9%） | 否 |

**结论**：完整语法模型在纯词表内事件上仍 ~100% 检出——它抓的是字段组合违规
（高 NLL），不是词表外 token。生产模型 bee 词表内子集分数 p50=21.8 vs τ=2.6（8.4×）。
只有 keys-only 基线靠 OOV。这一条把"分布失配"质疑转化为论文的加分项。

数据：`results/invocab_analysis.json` + `results/invocab_production_model.json`
（脚本 `scripts/invocab_analysis.py` / `scripts/score_only_invocab.py`）。

### ADFA-LD（阈值按良性 q99 标定）

| 变体 | 宏检出率(融合) | 实测 FPR |
|---|---|---|
| bigram（连续对） | **31.56% ±0.32% (CI95 ±0.40%)** | 1.83% |
| skip-1 bigram | 29.68% ±0.72% | 1.72% |
| skip-2 bigram | 27.83% ±0.82% | 1.72% |

负结果：跳步 bigram 不优于连续 bigram（原报告 §6.1 设想的扩展方向被证伪，可写）。

逐类（bigram, CI95）：Hydra_FTP 46.8%±0.8%、Hydra_SSH 40.6%±0.6%、
Web_Shell 29.5%±1.4%、Adduser 27.5%±1.0%、Java_Meterpreter 25.3%±0.9%、
Meterpreter 19.7%±0.7%。

### AAA 对抗强度量化

- 时序拟态证据：DT 分布 KL 散度 rove 1.05 → bee 0.54（向良性迁移，频率跟随生效）
- 最大破绽：进程词汇（KL 7.3；AAA 37 词 vs 良性 88 词）
- 与文献锚点对比可用的区分点：gym-malware/MERLIN/MAB-Malware 仅变异 API/字节
  序列、无时机适应；AAA 具备宿主频率跟随+渐稳减速+rebirth 自变异（运行时行为层适应）

---

## 实验补充方案（E1/E3/E4）关键数字

### E1 · CV 失效边界（defense-atomic-0 靶场，VM-local TinyGPT）

| σ 档 | 实测 CV | CV 面检出 | 单事件语法暴露 | 聚合显著性（binomial） |
|---|---|---|---|---|
| 0.3 | 0.297 | 是 | 8.6% | p=2.6e-8 |
| 1.13 | 1.470 | 是（压线） | 9.3% | p=2.7e-9 |
| 1.25 | 1.733 | **否（逃逸）** | 8.9% | p=7.0e-9 |
| 1.75 | 3.180 | **否（逃逸）** | 9.6% | p=1.2e-9 |

良性基线 FPR 2.7%。**CV 可被重尾间隔绕过，但逃逸留下统计显著语法指纹
（3.3× 异常率升高，聚合 p≈1e-9）**——多面融合必要性实测证明。
图：`figures/fig_e1_cv_boundary{,_en}.png`（Figure 1 候选）。

### E3 · 异质负载基线（7 VM 三角色，25,632 事件）

| 角色 | val_ppl | 留出 FPR | 攻击检出 |
|---|---|---|---|
| dev | 1.32 | 0.00% | 3/3 |
| web | 1.44 | 0.00% | 3/3 |
| desktop | 1.58 | 0.66% | 3/3 |

**LOVO 跨机泛化 FPR = 0.00% ±0.00%（7 折全过）**。跨角色矩阵揭示 web 特异性：
web 训 → 其他角色 19%/14.2%（迁移差），dev/desktop 训 → web 0.9%/0.6%。
结论：不过拟合单一负载，但跨域仍须本机自训。

### E4 · OpTC / DARPA TC 公开基准（500 Windows 端点）

val_ppl=1.17，τ=6.334，良性留出 FPR=0.00%。

| 场景 | 目标 | 检出率 | 良性 FPR |
|---|---|---|---|
| sc1 | SysClient0201 | 82.3% | 23.3% |
| sc1 | SysClient0402 | 81.5% | 24.4% |
| sc2 | SysClient0501 | 65.1% | 8.4% |
| sc2 | SysClient0005 | 87.0% | 6.8% |
| sc3 | SysClient0051 | 42.0% | 5.3% |
| sc3 | SysClient0351 | 22.7% | 4.7% |
| **均值** | — | **63.4%** | — |

**检出随对抗强度递减**（sc1 进程链攻击 82% > sc3 拟态低慢 22-42%）；
**跨日 FPR 漂移 0%→5-24%**（sep16→sep23-25 时间漂移）——印证定期重训必要性。
报告：`reports/e4_optc_benchmark.md`；结果：`results/optc_eval.json`。

## 复现

```bash
# 环境：torch>=2.8, numpy；数据路径见各脚本顶部常量/环境变量
# 1. DeepLog 对比矩阵（单轮）
python3 scripts/deeplog_baseline.py --arch tinygpt --seed 0 \
  --out results/deeplog_tinygpt_seed0.json
#    数据由环境变量指定：BENIGN_PATH / ATTACK_PATH / ATTACK_PATH2
#    双卡并行编排：scripts/run_matrix_cloud.sh

# 2. ADFA 多 seed
python3 scripts/adfa_bigram.py <ADFA-LD根> --epochs 3 --valsplit 3500 \
  --skip 0 --seed 0 --out results/adfa_skip0_seed0.json
#    双卡并行编排：scripts/run_adfa_matrix_cloud.sh

# 3. 聚合
python3 scripts/aggregate_stats.py            # -> results/aggregate_summary.json

# 4. AAA 校准
python3 scripts/aaa_calibration.py            # -> reports/aaa_calibration.json

# 5. 出图（需 matplotlib/seaborn，中文字体 Noto Sans CJK）
python3 scripts/make_paper_figures.py         # -> figures/
```

## 硬件记录

DeepLog/ADFA 矩阵：双 NVIDIA H800 80GB（云端），torch 2.8.0+cu128；
分布图数据（150K 事件子采样）：本机 AMD ROCm GPU。
司命原始实验环境见论文 §5 设施描述。
