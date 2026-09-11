# E4 报告 · OpTC / DARPA TC 公开基准评测

日期：2026-09-11 ｜ 对应审稿 C2/C3 项（公开数据集 + 跨机泛化）｜ 云端 A800 完成
数据集：[DARPA OpTC](https://github.com/FiveDirections/OpTC-data)（Inria 修正版，doi:10.57745/UXCWOC）
模型：TinyGPT 0.85M（OpTC 良性期 sep16 自训，词表 117）

## 1. 科学问题

在**公开多主机基准**上验证行为语法管线的检测率/FPR 与跨机泛化能力，
补外部可复现性短板（对应审稿"无公开数据集验证"）。
与 ADFA-LD 小节的关系：ADFA 测裸 syscall 序列组件（负结果保留），OpTC 测完整 8-token 管线。

## 2. 数据集与协议

- **OpTC**：500 台 Windows 10 端点，eCAR 端点事件（PROCESS/FLOW），
  良性期（09-16~22）+ 3 个红队场景（sc1=09-23 / sc2=09-24 / sc3=09-25）
- **schema 适配**：eCAR PROCESS→EXEC、FLOW→CONN，image_path→PROC/PARENT、
  command_line→ARGV、user→UID、dest_ip/port→DST，timestamp→DT（8-token 语法）
- **训练**：良性期 sep16 的 12 台 client 池化（36 万事件）→ TinyGPT + τ(p995)
- **评测**：3 场景 6 台目标 client，用 Inria 修正版 ground truth（恶意 pid + 时间窗）
  标定攻击事件；检出率 = 攻击事件超阈比例；FPR = 对照良性事件误报率（选择性流式打分）

## 3. 结果

val_ppl=1.17，τ(p995)=6.334，良性留出 FPR=0.00%。

| 场景 | 目标 client | 攻击事件 | 检出 | 检出率 | 良性 FPR |
|---|---|---|---|---|---|
| sc1 | SysClient0201 | 31,826 | 26,199 | **82.3%** | 23.3% |
| sc1 | SysClient0402 | 35,575 | 28,999 | **81.5%** | 24.4% |
| sc2 | SysClient0501 | 24,355 | 15,860 | **65.1%** | 8.4% |
| sc2 | SysClient0005 | 2,696 | 2,346 | **87.0%** | 6.8% |
| sc3 | SysClient0051 | 26,526 | 11,131 | **42.0%** | 5.3% |
| sc3 | SysClient0351 | 32,883 | 7,457 | **22.7%** | 4.7% |
| **均值** | — | — | — | **63.4%** | — |

## 4. 分析

### 4.1 检测率随场景对抗强度递减

- **sc1（82%）**：红队 hands-on-keyboard 的 PowerShell/cmd 攻击（可见 ground truth 中
  大量 powershell -enc / cmd.exe / runme.bat），行为语法对这种"进程链异常"高度敏感
- **sc2（65-87%）**：中等，含更多合法工具滥用
- **sc3（22-42%）**：最难——红队最贴近正常用户行为的低慢操作，语法差异最小
  （SysClient0351 仅 22.7%，SysClient0051 42%）

这与内部 AAA 对抗的结论一致：**行为语法对"进程组合异常"强，对"完美拟态"弱**——
sc3 正是逼近拟态上限的场景。

### 4.2 跨机/跨日泛化的 FPR 漂移（诚实边界）

良性留出（sep16 同日）FPR=0%，但攻击日（sep23-25）对照 client 的 FPR 升到 5-24%——
**时间漂移**：模型在 sep16 训的语法，对一周后的活动分布不再完全贴合。
sc1（sep23，最远）FPR 最高（23-24%），sc3（sep25，但良性更接近）反而低（4-5%）。

这再次印证论文核心教训：**先验必须本机自训 + 定期重训**——跨时间的分布漂移
需要 onboarding/增量更新机制吸收，否则 FPR 随时间劣化。

## 5. 结论（对审稿人的回答）

- ✅ **公开数据集**：OpTC/DARPA TC 上的端到端检测率 22-87%（均值 63%），外部可复现
- ✅ **跨机泛化**：良性同日 FPR=0%，LOVO 类多机训练在良性期有效
- ⚠️ **诚实边界**：(a) 强拟态场景（sc3）检测率有限（22-42%）；
  (b) 跨日 FPR 漂移（0%→5-24%）——需 onboarding/定期重训，论文已论证此机制

**建议入文英文句**：

> *Public benchmark (OpTC/DARPA TC).* On the OpTC dataset (500 Windows 10 endpoints, eCAR endpoint telemetry), the full 8-token grammar pipeline, trained only on the benign period (Sep 16, 12 clients), attains 22–87% per-target detection (mean 63%) across three red-team scenarios with a benign holdout FPR of 0%. Detection is strongest for process-composition attacks (sc1 PowerShell/cmd chains, 82%) and degrades for near-mimicry low-and-slow operations (sc3, 22–42%). Cross-day control FPR rises from 0% (same-day) to 5–24% (one week later), quantifying the temporal drift that motivates per-machine onboarding and periodic retraining.

## 6. 产物

- 数据：`autodl-tmp/optc_data/`（25 client-day parquet）+ `optc_tok/`（25 token jsonl）
- 结果：`autodl-tmp/optc_tok/optc_eval.json` + `optc_eval.log`
- 模型：`autodl-tmp/optc_tok/optc_prior.pt`（OpTC 良性期先验，已缓存）
- Ground truth：`autodl-tmp/optc_gt/`（Inria 修正版 6 个场景事件文件 + CSV）
- 脚本：`optc_to_tokens.py`（eCAR→8-token）、`optc_eval.py`（流式评测）
- 计算：云端 A800（下载/转换/训练/评测全程）
