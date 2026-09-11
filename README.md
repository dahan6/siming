# 司命 Siming v4 — Behavioral Grammar Host Anomaly Detection Engine

**行为语法：用微型语言模型先验与二阶时序分析检测自适应恶意软件**
**Behavioral Grammar: Detecting Adaptive Malware via Tiny Language Model Priors and Second-Order Temporal Analysis**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-2608.00745-b31b1b.svg)](https://arxiv.org/abs/2608.00745)

<p align="center"><b>Zihan Luo</b></p>

---

## 中文

### 这是什么？

司命（Siming）是一个**主机行为异常检测引擎**，属于防御性安全研究项目。它把运行时行为当作一门"结构化语言"：每个系统事件被离散化为 8-token 表示，用一个 0.88M 参数的因果 Transformer（TinyGPT）以纯自监督方式学习正常行为的"语法"。异常分数来自分槽位 NLL 统计，每条告警都能指出是哪个槽位违反了语法。

```
ET:EXEC  PROC:bash  ARGV:N1P  PC:NONE  PARENT:sshd  UID:1000  DST:NONE  DT3
```

### v4 六网融合架构

| 层 | 功能 | 优先级 |
|---|---|---|
| **统计层** | TinyGPT 先验分槽位 NLL（本机自训） | 基础 |
| **语义层** | 语义编码器 + 原型网络（对比学习） | 基础 |
| **时序层** | 二阶时序分析（FFT / CV，机器节拍检测，C2 信标 SNR=25.9） | 最强信号 |
| **自适应层** | 自学习模式提取（auto_pattern，人工审核后入库） | 增强 |
| **P0–P5 分级融合** | P0 自适应高危模式 > P1 ATT&CK 模式+原型 > P2 上下文异常 > P3 稀有度 > P4 时序 > P5 自适应低危 | 裁决 |

### 关键指标

| 指标 | 数值 |
|---|---|
| 良性误报率 FPR | **0.47%** |
| 合成攻击检出率 TPR | **92.8%** |
| FFT C2 信标信噪比 | **SNR = 25.9** |
| 模型规模 | 0.88M 参数（约 3.6 MB），INT8 量化零掉点 |
| 推理 | CPU 实时（单事件 < 1 ms） |

### 诚实的限制清单

- **真实攻击 TPR 未验证**：92.8% 来自合成攻击与靶场对抗数据，不代表对野外真实攻击的检出率。
- **单机数据**：核心结论来自单台主机的遥测与少量 VM；跨机泛化已知很弱（先验必须本机自训，跨机 FPR 高达 92.5%）。
- **仅 Ubuntu**：全部实验在 Ubuntu（auditd / tracee / conntrack）上完成，未适配其他发行版或操作系统。
- **无实时 daemon**：发布的是离线检测与标定流水线；生产级常驻守护进程、告警对接需自行工程化。

### 快速开始

```bash
pip install torch numpy scikit-learn pyyaml

# 1. auditd 采集（Ubuntu，需 root）
sudo python3 detector/collect_auditd.py --out data/my_audit.jsonl

# 2. 解析为 token 流
python3 detector/parse_events.py data/my_audit.jsonl data/my_tokens.jsonl

# 3. onboard 标定：在本机良性数据上训练先验 + 校准分槽位阈值
bash detector/onboard.sh          # 或：
python3 detector/train_prior.py data/my_tokens.jsonl models/my-host
python3 detector/onboard_v2.py models/my-host data/my_tokens.jsonl

# 4. 检测
python3 detector/detect_in_trace.py models/my-host data/new_trace.jsonl
python3 detector/alert_report.py  # 查看分级告警
```

### 目录结构

```
release-siming-v4/
├── detector/    # 检测引擎源码（训练/打分/校准/融合/时序 FFT/基线/评测，40+ 脚本）
├── models/      # 模型代际目录（host r0–r3、vm-universal、semantic 系列、hybrid、stat；含 INT8 量化版）
├── data/        # 实验数据（audit jsonl、合成攻击、对比对、语义语料、多 seed 实验结果 json）
├── patterns/    # 模式库 patterns.jsonl、原型 prototypes.jsonl、候选模式、rove 序列
├── docs/        # 技术文档 + 论文全文与图 + 实验报告
├── mocheng/     # 墨城防火墙：同架构的网络侧行为语法防火墙（蓝方资产，iptables 执法）
└── range/       # 靶场重建脚本与说明（Atomic 采集清单、防御 VM 创建/还原/隔离脚本）；
                 # VM 镜像不入包，请按文档指引从官方源下载基础镜像后用 range/ 脚本重建
```

### 数据说明

`data/` 中的大文件以 gzip 压缩分发（`*.jsonl.gz`），使用前请解压：

```bash
gunzip data/host_tokens.jsonl.gz data/host_tokens_clean.jsonl.gz
# Windows 用户也可用 7-Zip 解压 .gz 文件
```

### 论文

- **Behavioral Grammar: Detecting Adaptive Malware via Tiny Language Model Priors and Second-Order Temporal Analysis** — [arXiv:2608.00745](https://arxiv.org/abs/2608.00745)
- Companion papers: *SPECIES constitution* and *Whetstones technical report* (arXiv, forthcoming)

### 双用途声明

本项目仅用于**防御性安全研究与教学**，不支持、不授权任何非法用途；使用者须遵守当地法律法规。发布内容不含红方项目的任何代码或数据；攻击轨迹仅含已从隔离靶场导出的 token 化遥测，用于防御评估。

---

## English

### What is Siming?

Siming is a **host-based behavioral anomaly detection engine** — a defensive security research project. It treats runtime behavior as a structured language: every system event is discretized into an 8-token representation, and a compact 0.88M-parameter causal Transformer (TinyGPT) learns the "grammar" of normal behavior in a purely self-supervised manner. Anomaly scores are derived from per-slot NLL statistics, so every alert names the exact slot that violated the grammar.

### v4 Six-Net Fusion Architecture

| Layer | Function | Priority |
|---|---|---|
| **Statistical** | TinyGPT per-slot NLL prior (self-trained per host) | Base |
| **Semantic** | Semantic encoder + prototype network (contrastive) | Base |
| **Temporal** | Second-order timing analysis (FFT / CV; machine-cadence detection; C2 beacon SNR = 25.9) | Strongest signal |
| **Adaptive** | Self-learning pattern extraction (human-reviewed before entry) | Enhancement |
| **P0–P5 fusion** | P0 adaptive high-risk > P1 ATT&CK patterns + prototypes > P2 context anomaly > P3 rarity > P4 temporal > P5 adaptive low-risk | Arbitration |

### Key Metrics

| Metric | Value |
|---|---|
| Benign FPR | **0.47%** |
| Synthetic-attack TPR | **92.8%** |
| FFT C2 beacon SNR | **25.9** |
| Model size | 0.88M params (~3.6 MB); INT8 quantization with zero degradation |
| Inference | Real-time on CPU (< 1 ms per event) |

### Honest Limitations

- **Real-world attack TPR is unverified** — 92.8% was measured on synthetic attacks and cyber-range adversarial data, not in-the-wild malware.
- **Single-host data** — core conclusions come from one host's telemetry plus a few VMs; cross-host generalization is known to be weak (priors must be self-trained per host; naive cross-host FPR reaches 92.5%).
- **Ubuntu only** — all experiments ran on Ubuntu (auditd / tracee / conntrack); no other distro or OS has been tested.
- **No real-time daemon** — this release ships the offline detection and calibration pipeline; a production-grade resident daemon and alerting integration are left to the user.

### Quick Start

```bash
pip install torch numpy scikit-learn pyyaml

# 1. Collect with auditd (Ubuntu, root required)
sudo python3 detector/collect_auditd.py --out data/my_audit.jsonl

# 2. Parse into a token stream
python3 detector/parse_events.py data/my_audit.jsonl data/my_tokens.jsonl

# 3. Onboard: train the prior on your own benign data and calibrate per-slot thresholds
python3 detector/train_prior.py data/my_tokens.jsonl models/my-host
python3 detector/onboard_v2.py models/my-host data/my_tokens.jsonl

# 4. Detect
python3 detector/detect_in_trace.py models/my-host data/new_trace.jsonl
python3 detector/alert_report.py
```

### Data Notes

Large files under `data/` are distributed gzip-compressed (`*.jsonl.gz`). Decompress before use:

```bash
gunzip data/host_tokens.jsonl.gz data/host_tokens_clean.jsonl.gz
# On Windows, 7-Zip can extract .gz files as well
```

The `range/` directory contains cyber-range rebuild scripts and documentation (Atomic collection manifest, defense-VM create/reset/lockdown scripts). VM images are not included — download a base image from official sources as directed by the docs, then rebuild with the `range/` scripts.

### Paper

- **Behavioral Grammar: Detecting Adaptive Malware via Tiny Language Model Priors and Second-Order Temporal Analysis** — [arXiv:2608.00745](https://arxiv.org/abs/2608.00745)
- Companion papers: *SPECIES constitution* and *Whetstones technical report* (arXiv, forthcoming)

### Dual-Use Statement

**This project is for defensive security research and education only. It does not support or authorize any illegal use.** Users must comply with applicable local laws and regulations. This release contains no code or data from any red-team project; attack traces are tokenized telemetry exported from an isolated cyber range, included solely for defensive evaluation.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{luo2026behavioral,
    title={Behavioral Grammar: Detecting Adaptive Malware via Tiny Language Model Priors and Second-Order Temporal Analysis},
    author={Zihan Luo},
    year={2026},
    eprint={2608.00745},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2608.00745}
}
```
