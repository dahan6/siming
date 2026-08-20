# Siming — Behavioral Grammar Detection Engine

**Treating host runtime behavior as a language — and learning its grammar.**

Siming is a next-generation host-based intrusion detection engine that models endpoint telemetry as a structured behavioral language. By learning the conditional distribution of normal system behavior with a compact 0.88M-parameter causal Transformer, Siming achieves near-perfect detection of adaptive malware — including adversaries that actively learn to evade defenses — while maintaining a mathematically bounded false-positive rate of just 0.3%.

Unlike signature-based engines that are trivially evaded by polymorphic threats, and unlike heavyweight deep-learning models that resist auditability, Siming occupies a unique position: **small enough to deploy anywhere, expressive enough to detect anything.**

## Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Benign False-Positive Rate** | **0.3%** | Mathematically bounded via per-slot threshold calibration |
| **Attack Detection Rate (TPR)** | **92.8%** | Against adaptive adversarial agents |
| Exfiltration / Lateral Movement | **100%** | Zero misses across all test scenarios |
| Reconnaissance | **99.8%** | Including low-and-slow probing |
| Persistence | **87.7%** | Novel persistence techniques included |
| Privilege Escalation | **80.8%** | Multi-step escalation chains |
| FFT-based C2 Detection | **SNR = 25.9** | Periodic beacon detection under heavy noise |
| Adaptive Adversarial Tests | **7/7** | Against agents that learn survival strategies |

## Architecture: Six-Network Fusion Pipeline

Siming employs six independent detection networks operating in parallel, each analyzing a different dimension of host behavior. Their outputs are fused through a cross-validation layer that dramatically reduces false positives while preserving detection sensitivity.

### Layer 1: Statistical Layer (TinyGPT Prior)

A 0.88M-parameter causal Transformer (TinyGPT, 4 layers, 128 dimensions) learns the conditional probability distribution of normal system events. Each event is discretized into an 8-slot token representation spanning event type, process, argument skeleton, path category, parent process, user, destination, and inter-event timing. Anomaly scores derive from per-slot negative log-likelihood statistics, with thresholds calibrated at the 99.5th percentile of the baseline distribution — yielding a **provable per-dimension false-positive bound**.

### Layer 2: Semantic Layer (Contrastive Learning + Intent Classification)

A contrastive learning module (InfoNCE loss) maps behavioral events into a semantic embedding space, where a classifier head identifies six behavioral intent categories. This layer captures *what the behavior means*, not just *whether it is unusual* — enabling detection of novel attack variants that share semantic intent with known techniques.

### Layer 3: Temporal Layer (FFT + Coefficient of Variation)

The strongest discriminative signal against adaptive adversaries comes not from individual events but from **timing structure**. The coefficient of variation (CV) of inter-event intervals reveals a fundamental stealth-functionality trade-off: adaptive agents require regularized stepping cadences (CV ~ 0.3) for task responsiveness, while benign activity is burst-driven (CV ~ 9.8) — a **30x separation** that cannot be hidden without sacrificing operational capability. FFT-based periodicity detection further identifies C2 beaconing patterns with SNR = 25.9.

### Layer 4: Adaptive Detector (Variant-Tolerant Matching)

Purpose-built to handle polymorphic and metamorphic threats, this layer performs fuzzy behavioral matching that tolerates process renames, path mutations, and argument obfuscation — the exact evasion techniques that defeat signature-based systems.

### Layer 5: Prototype Network (Known-Attack Attribution)

Learns prototype representations for known attack patterns, enabling rapid attribution of detected anomalies to specific MITRE ATT&CK techniques and threat actor TTPs.

### Layer 6: Self-Learning Pattern Extractor

Continuously mines detected anomalies for novel behavioral patterns, automatically generating detection rules that feed back into the pattern library. The system **improves with every deployment** — no manual rule authoring required.

## Why Siming Works: The Coevolutionary Argument

Siming's design is grounded in a coevolutionary economics model of attacker-defender dynamics:

- **Rule-based detection** shifts evasion cost to *rule circumvention* — cheap, well-understood, endlessly repeatable.
- **Behavior-grammar detection** shifts evasion cost to *distribution matching* — the adversary must make malicious behavior statistically indistinguishable from benign behavior across 8 dimensions simultaneously.

This establishes a **structural asymmetry that favors the defender**: the cost of evasion scales superlinearly with the number of monitored dimensions, while the cost of detection scales linearly.

## Quick Start

```bash
# Install dependencies
pip install torch numpy scikit-learn
sudo apt install auditd

# Enable auditd execve monitoring
sudo auditctl -a always,exit -F arch=b64 -S execve -k exec_log
sudo auditctl -a always,exit -F arch=b32 -S execve -k exec_log

# Run the six-network fusion pipeline
python detector/fusion_pipeline.py --eval data/audit_all.jsonl

# Calibrate on a new machine
python detector/onboard_v2.py models/model-stat-v3 data/onboard_benign.jsonl
```

## Project Structure

```
siming-full/
├── docs/                              # Documentation
│   ├── siming-system-doc-v4.md        # Complete system documentation
│   ├── full-layer-upgrade-report.md   # Layer-by-layer upgrade analysis
│   ├── anti-adaptive-upgrade-report.md # Anti-adaptive defense report
│   ├── paper_*.md                     # Research paper (arXiv submission)
│   ├── figures/                       # 8 publication-quality figures
│   └── blue-team-dialogue-notes.md    # Blue team operational notes
├── detector/                          # Core detection engine (42 scripts)
│   ├── fusion_pipeline.py             # Six-network fusion orchestrator
│   ├── stat_layer_upgrade.py          # Statistical layer (PREV + EWMA)
│   ├── semantic_layer_upgrade.py      # Semantic layer (classifier + focal loss)
│   ├── temporal_fft.py                # Temporal layer (FFT + multi-scale)
│   ├── adaptive_detector.py           # Adaptive layer (variant-tolerant)
│   ├── train_semantic.py              # Contrastive + classification training
│   ├── auto_labeler.py                # Automatic weak labeler
│   ├── collect_auditd.py              # auditd event collector
│   ├── deploy_siming.py               # One-click deployment CLI
│   ├── patterns.jsonl                 # Pattern library (99 entries)
│   └── ...                            # 30+ additional utility scripts
├── models/                            # Pretrained models
│   ├── model-stat-v3/                 # Statistical layer (333-token vocab, 3.6MB)
│   ├── model-semantic-v5/             # Semantic layer (3.5MB)
│   └── model-semantic-embed/          # Semantic embeddings (532KB)
├── data/                              # Datasets
│   ├── audit_all.jsonl                # Real auditd events (7,598 records)
│   ├── synth_attacks_v4.jsonl         # Synthetic attacks (4,387 records)
│   ├── classifier_train_v5.jsonl      # Classifier training set
│   ├── contrastive_pairs_v5.jsonl     # Contrastive learning pairs
│   ├── semantic_corpus_full.json      # Semantic training corpus
│   └── auto_patterns_candidates.jsonl # Auto-extracted pattern candidates
└── README.md
```

## Tech Stack

- **TinyGPT**: 4-layer causal Transformer, 0.90M parameters, 128-dim embeddings
- **Contrastive Learning**: InfoNCE loss with 6-class behavioral intent classifier
- **Temporal Analysis**: FFT periodicity detection + coefficient of variation (CV)
- **Event Collection**: Real-time auditd execve monitoring with 8-slot tokenization
- **Runtime**: Python 3.12 / PyTorch 2.5

## Research Paper

This system is described in detail in our paper:

> **Behavioral Grammar: Detecting Adaptive Malware via Tiny Language Model Priors and Second-Order Temporal Analysis**
> Zihan Luo — submitted to arXiv (cs.AI)

## License

Apache 2.0 — Zhiyan Security Lab
