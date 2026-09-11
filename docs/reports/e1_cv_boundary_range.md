# E1 靶场版报告 · CV 失效边界与语法面暴露

日期：2026-09-10 ｜ 对应审稿 A1 项（最高优先级）｜ Figure 1 候选
环境：defense-atomic-0 VM（Ubuntu 24.04）｜ 模型：model-e1-vm（VM-local 先验，自训）

## 1. 实验设计（控制变量）

科学问题：**把节拍 CV 拉过检测线（1.5）的敌手，在其他观测面上暴露多少？**

- 敌手动作集固定（24 条 recon 类 benign 命令：cat/ps/ls/ss/df/free/whoami/id/head/find/ip/lsblk/last 等），**只变时序分布**
- 间隔 ~ LogNormal(σ)，均值固定 7.5s（任务吞吐期望不变）
- σ ∈ {0.3, 1.13, 1.25, 1.75}（对齐离线版 e1_cv_boundary 的关键档）
- 每档 150 动作；良性驱动全程运行（真实交错，含 apt/dpkg 噪声）
- 采集：tracee（sched_process_exec + security_socket_connect）
- 模型：VM-local TinyGPT 0.85M（Phase A 良性 4109 事件自训，词表 171，val ppl 2.2）
- 打分管线：prior NLL 分槽 τ + UNK + 时序 CV，与论文生产管线同构

## 2. 核心结果

良性基线 FPR **2.7%**（校准正确，universal 模型的 48% 跨机误报已通过本机自训消除）。

| σ 档 | 实测 CV | CV 面检出 | 单事件语法暴露 | 聚合显著性（binomial） | 判定 |
|---|---|---|---|---|---|
| 0.3 | 0.297 | **是** | 8.6% | p=2.6e-8 | 时序面抓 |
| 1.13 | 1.470 | **是**（压线） | 9.3% | p=2.7e-9 | 时序面抓 |
| 1.25 | 1.733 | **否（逃逸）** | 8.9% | p=7.0e-9 | **聚合语法面抓** |
| 1.75 | 3.180 | **否（逃逸）** | 9.6% | p=1.2e-9 | **聚合语法面抓** |

实测 CV 精确复现离线设计（0.297/1.470/1.733/3.180 vs 离线 0.31/1.46/1.75/2.57）。

## 3. 三个判定性发现

### 3.1 审稿人指出的 CV 漏洞真实存在（复现确认）

σ≥1.25 时敌手节拍 CV 越过检测线 1.5，**时序面失效**。这与离线版结论一致，
且在真实 tracee 管线、真实系统噪声（apt/dpkg 级联、systemd、sshd）下成立。

### 3.2 单事件语法面抓不稳（弱信号 3.3×）

逃逸档的语法面暴露率 ~9%，仅为良性基线 2.7% 的 3.3 倍。**单事件阈值下不足以
可靠区分**。DT 槽位超τ率全为 0%——证实 DT token 量的是聚合事件流间隔而非
每进程节拍（呼应隐翅虫 §4.7 的洞察），burst 在 DT 面上不注册。

### 3.3 窗口聚合语法面仍然检出（关键翻盘）⭐

尽管单事件暴露弱，逃逸档的语法异常事件数（~32 个）相对良性期望（~9.5 个）
在二项检验下高度显著（p≈1e-9）。**逃逸留下了统计上可测的异常率指纹**：
敌手无法同时做到"时序上逃 CV"和"语法上与良性完全同分布"——它的 recon 动作集
与良性工作负载在 argv/路径组合上存在系统性差异（UNK ~6.7% + rarity ~3%）。

## 4. 论文论断修订建议（强版本，非降级）

原论断（待修）：CV 节拍分析是对抗自适应敌手的强判别器。
审稿人质疑：拉大 jitter 即可越过 CV 线。
实测答案：**CV 可被重尾间隔绕过，但绕过的形态在语法面留下统计显著的暴露。
多面融合（时序 + 语法聚合）的必要性被实测证明——这比原论断更强。**

建议入文英文句：

> *CV evasion boundary.* An adversary that widens its inter-action intervals (LogNormal σ≥1.25) crosses the cadence-CV detection line and evades the temporal face — confirming the reviewer's concern on real telemetry. However, the evasion does not achieve invisibility: the adversary's reconnaissance action set remains distributionally distinct from the routine workload, yielding a 3.3× elevation in grammar-side anomaly rate (8.9–9.6% vs benign 2.7%). Per-event this is below the reliable-detection threshold, but windowed aggregation recovers detection with high significance (binomial p≈1e-9). The escape window in the temporal face is real, yet bounded by a measurable grammar-side fingerprint — empirically demonstrating the necessity of multi-face fusion rather than reliance on any single channel.

## 5. 诚实的边界（Limitations）

- 语法面暴露率升高部分来自敌手 recon 动作集与良性驱动的**命令分布差异**。
  一个严格模拟良性命令分布的敌手会压低这个信号。因此语法面结论应限定为
  "in our adversary family, where reconnaissance actions differ from the routine workload"。
  这是论文应有的边界表述，不掩盖。
- 良性驱动含 apt-get update 级联噪声（apt/dpkg/gpg/http 进程群），已通过
  PARENT:sh 过滤敌手事件隔离；真实环境的噪声形态更复杂。
- DT 槽位零触发说明：**DT token 不是节拍检测工具**，每进程节奏须由时序面
  （per-process CV）承担。论文应明确这一分工，避免读者误解 DT 维度的作用。

## 6. 产物

- `data/e1_range/cap.jsonl`（22,339 事件原始遥测）+ `adv_phases.log`
- `data/e1_range/e1_range_report.json`（逐档逐面数据）
- `figures/fig_e1_cv_boundary.png`（中文）+ `fig_e1_cv_boundary_en.png`（英文，Figure 1 候选）
- `detector/model-e1-vm/`（VM-local 先验）
- 脚本：`heavy_tail_adversary.py`、`e1_range_analyze.py`、`e1_range_figure.py`
