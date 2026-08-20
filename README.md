# Siming — Behavioral Grammar Detection Engine v4

**Six-network fusion pipeline**: four independent detection layers (statistical + semantic + temporal + adaptive) with cross-validation to minimize false positives.

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

## Performance

| Metric | Value |
|--------|-------|
| Benign FPR | 0.3% |
| Attack TPR | 92.8% |
| Exfiltration / Lateral Movement | 100% |
| Reconnaissance | 99.8% |
| Persistence | 87.7% |
| Privilege Escalation | 80.8% |
| FFT C2 Detection | SNR = 25.9 |
| Adaptive Adversarial Tests | 7/7 |

## Project Structure

```
siming-full/
├── docs/                          # Documentation
│   ├── siming-system-doc-v4.md    # Complete system documentation
│   ├── full-layer-upgrade-report.md    # Upgrade comparison
│   ├── anti-adaptive-upgrade-report.md # Early anti-adaptive report
│   ├── paper_*.md                 # Research paper
│   ├── figures/                   # 8 paper figures
│   └── blue-team-dialogue-notes.md
├── detector/                      # Core engine (42 Python scripts)
│   ├── fusion_pipeline.py         # Six-network fusion pipeline
│   ├── stat_layer_upgrade.py      # Statistical layer (PREV + EWMA)
│   ├── semantic_layer_upgrade.py  # Semantic layer (window classifier + focal loss)
│   ├── temporal_fft.py            # Temporal layer (FFT + multi-scale)
│   ├── adaptive_detector.py       # Adaptive layer (variant-tolerant)
│   ├── train_semantic.py          # Contrastive learning + classifier head training
│   ├── auto_labeler.py            # Automatic weak labeler
│   ├── collect_auditd.py          # auditd event collector
│   ├── deploy_siming.py           # One-click deployment CLI
│   ├── patterns.jsonl             # Pattern library (99 entries)
│   └── ...                        # Additional utility scripts
├── models/                        # Pretrained models
│   ├── model-stat-v3/             # Statistical layer (333-token vocab, 3.6MB)
│   ├── model-semantic-v5/         # Semantic layer (3.5MB)
│   └── model-semantic-embed/      # Semantic embeddings (532KB)
├── data/                          # Datasets
│   ├── audit_all.jsonl            # Real auditd events (7,598 records)
│   ├── synth_attacks_v4.jsonl     # Synthetic attacks (4,387 records)
│   ├── classifier_train_v5.jsonl  # Classifier training set
│   └── ...
└── README.md
```

## Tech Stack

- TinyGPT (4-layer Transformer, 0.90M parameters, 128-dim)
- Contrastive learning (InfoNCE) + classifier head (6 behavioral intent classes)
- FFT periodicity detection + coefficient of variation (CV) analysis
- Real-time auditd execve event collection
- Python 3.12 / PyTorch 2.5

## License

Apache 2.0 — Zhiyan Security Lab
