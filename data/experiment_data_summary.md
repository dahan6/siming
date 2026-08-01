# Siming — Complete Experiment Data Summary

**All data collected from isolated lab experiments. No live systems involved.**

---

## Experiment 1: Four-Generation Model Evolution

Cross-host generalization improvement across four iterations of model design.

| Metric | v1 (Host) | v2 (Single-VM +VMτ) | v3 (Single-VM +Localτ) | v4 (Multi-VM +p99.5) |
|--------|-----------|---------------------|------------------------|----------------------|
| Total benign FPR | 52% | 26% | 24% | 25% |
| Cross-host FPR (P2+P3) | 69% | 35% | 12% | 9.5% |
| Agent detection rate | 94% | 93% | 93% | 93% |
| P4 temporal layer | 67% | 68% | 88% | 92% |
| Onboarding FPR | 92% | 12% | 4.3% | 3.84% |
| Training data | 1 host | 1 VM (18K events) | 1 VM (18K events) | 7 VMs (116K events) |
| τ calibration | host p99 | VM p95 | local p99 | local p99.5 |

### v1→v4 Improvement Summary
- Cross-host FPR: **69% → 9.5%** (−86%)
- Onboarding FPR: **92% → 3.84%** (−96%)
- Temporal detection rate: **67% → 92%** (+37%)

---

## Experiment 2: Layer Ablation Study

Systematic removal of each detection layer to measure its contribution.

| Configuration | Agent Detection (%) | Benign FPR (%) | Discrimination Margin |
|---------------|--------------------:|---------------:|---------------------:|
| **Full (5-layer)** | **93.0** | **24.7** | **+68.3** |
| No Temporal (P4) | 73.9 | 21.5 | +52.4 |
| No Adaptive (P0/P5) | 93.0 | 24.7 | +68.3 |
| No Pattern (P1) | 90.1 | 11.7 | +78.4 |
| Prior Only (P2/P3) | 0.4 | 2.4 | −1.9 |
| Temporal Only | 89.7 | 9.3 | +80.5 |

### Key Findings
1. **Temporal (P4) is the critical layer**: removing it drops detection 93%→74% (−19pp)
2. **Adaptive layer (P0/P5) is event-dependent**: zero contribution in this window (no transformation/SUID events occurred)
3. **Prior alone is insufficient**: 0.4% detection — model learned benign distribution too well (val_ppl=1.6)
4. **Temporal alone achieves best discrimination**: +80.5% margin (89.7% detection at 9.3% FPR)

---

## Experiment 3: Second-Order Temporal CV Analysis

Per-process coefficient of variation (CV) of inter-event intervals.

### 3.1 Per-Process CV (raw timestamps, not DT buckets)

| Process | Benign CV | Agent CV | Ratio |
|---------|----------|----------|------:|
| sleep | **9.786** | **0.310** | 31.6× |
| head | 3.298 | 0.792 | 4.2× |
| systemctl | 2.446 | 1.178 | 2.1× |
| cat | 3.326 | 1.234 | 2.7× |
| ps | 2.916 | 1.304 | 2.2× |
| ls | 2.284 | 1.448 | 1.6× |
| find | 1.873 | 1.423 | 1.3× |

### 3.2 Distribution Statistics

| Statistic | Benign | Agent |
|-----------|--------|-------|
| CV p5 | 1.509 | 0.647 |
| CV p10 | 1.612 | 0.947 |
| CV median | 2.284 | 1.304 |

### 3.3 Detection Performance
- Threshold: CV < 1.5
- Detection: **15/15** agent processes flagged
- False positive: **1/15** benign processes (scripted shell, CV=1.268)
- F1 score: **0.94** (temporal dimension alone)

---

## Experiment 4: τ Percentile Sensitivity

| Percentile | Benign FPR | Assessment |
|-----------|-----------:|------------|
| p95 | 26.89% | Too aggressive |
| p97 | 16.37% | Marginal |
| p99 | 6.92% | Above 5% target |
| **p99.5** | **3.79%** | **Selected** |
| p99.9 | 0.74% | Overly conservative |
| p99.95 | 0.37% | Excessively conservative |

---

## Experiment 5: Cross-Host Leave-One-VM-Out Validation

Universal model trained on 7 VMs, tested on each independently.

| Test VM | Events | FPR |
|---------|-------:|-----:|
| VM-1 | 5,000 | 1.2% |
| VM-2 | 5,000 | 1.0% |
| VM-3 | 3,731 | 2.4% |
| VM-4 | 3,733 | 4.7% |
| VM-5 | 2,463 | 3.6% |
| VM-6 | 2,688 | 6.1% |
| VM-7 | 2,450 | 4.7% |
| **Average** | — | **3.4%** |

6/7 VMs below 5% target. Only VM-6 slightly exceeds (6.1%).

---

## Experiment 6: Synthetic Attack Sequence Detection

| Test | Result | Description |
|------|--------|-------------|
| Morphological transformation | ✅ PASS | cp→chmod→setsid→rm sequence |
| Disguised C2 exfiltration | ✅ PASS | All 9 disguise name pool entries detected |
| SUID privilege escalation | ✅ PASS | UID:0 + python3, sudo path correctly excluded |
| Memory-only exfiltration | ✅ PASS | cat passwd → python3 POST C2 chain |
| Uniform recon sweep | ✅ PASS | 8 commands, max/min ratio=1.0 |
| Sleep stepping pattern | ✅ PASS | 10/20 sleep + 10 alternations |
| Benign low FPR | ✅ PASS | Skewed distribution: 0/500 alerts |

**7/7 tests passed.**

---

## Experiment 7: Prototype Network Performance

| Metric | Value |
|--------|-------|
| Sample library | 80 sequences, 14 ATT&CK techniques |
| Prototypes per technique | K=3 |
| Leave-one-out catch rate | **96.2%** (77/80) |
| Held-out benign FPR | 0/20,000 |
| Technique-level recall | 14/14 |
| Codebook size | ~80 KB |

### Per-Technique Radii

| Technique | Catch Rate | Radius |
|-----------|-----------:|-------:|
| T1037.004 (SUID script) | 3/3 | 0.000 |
| T1041 (Exfiltration) | 1/1 | 0.000 |
| T1053.002 (At queue) | 3/3 | 0.000 |
| T1053.003 (Crontab) | 5/6 | 0.239 |
| T1053.006 (Systemd timer) | 3/3 | 0.000 |
| T1070.004 (Transformation) | 1/1 | 1.020 |
| T1082 (Discovery/Recon) | 40/41 | 2.676 |
| T1098.004 (SSH key) | 3/3 | 0.000 |
| T1136.001 (User creation) | 3/3 | 0.000 |
| T1543.002 (Systemd service) | 3/3 | 0.000 |
| T1546.004 (RC file) | 3/3 | 0.000 |
| T1547.006 (Kernel module) | 3/3 | 2.455 |
| T1548.001 (SUID escalation) | 3/4 | 1.489 |
| T1574.006 (LD preload) | 3/3 | 0.000 |

---

## Experiment 8: Self-Learning Pattern Discovery

| Metric | Value |
|--------|-------|
| Input sequences | 80 labeled |
| Discovered clusters | **22** |
| Silhouette coefficient | **0.653** |
| Candidate patterns | 15 (pending review) |
| ATT&CK techniques covered | 10 |

### Top Clusters by Sample Count

| Cluster | Samples | Primary Technique | Key Behavior |
|---------|--------:|-------------------|-------------|
| C1 | 10 | T1082 | Recon: head/systemctl/journalctl/uptime |
| C2 | 8 | T1082 | Recon: readlink batch |
| C4 | 7 | T1082 | Recon: readlink/env/sudo/grep/ps |
| C11 | 7 | T1082 | Mixed: cat/tar/tail/head/ls |
| C3 | 6 | T1053.003 | Persistence: tee → crontab |
| C5 | 6 | T1543.002 | Systemd: netplan/sshd-socket-gen |
| C8 | 5 | T1548.001 | SUID: env+sudo UID:0 |
| C9 | 5 | T1082 | Recon: readlink |
| C0 | 3 | T1098.004 | SSH key: ssh-keygen+cat |
| C6 | 3 | T1136.001 | User: useradd/userdel |

---

## Experiment 9: Cross-Host Slot τ Calibration Difference

Per-slot NLL threshold difference between host-trained model and VM data.

| Slot | Host τ | VM τ | Δ | Impact |
|------|-------:|-----:|----:|--------|
| DT | 4.346 | 10.463 | +6.117 | VM timing differs significantly |
| PARENT | 6.465 | 10.093 | +3.628 | VM parent processes differ |
| PC | 2.247 | 6.854 | +4.607 | VM path categories differ |
| DST | 2.497 | 1.733 | −0.764 | VM has fewer network targets |
| ET | 2.937 | 2.241 | −0.696 | Event types similar |
| ARGV | 4.904 | 4.493 | −0.411 | Argument patterns similar |
| PROC | 6.346 | 6.057 | −0.289 | Process names similar |
| UID | 3.987 | 4.602 | +0.615 | UID distribution slightly different |

**Root cause**: DT/PARENT/PC slots have the largest cross-host variance, confirming that timing and process lineage are host-specific features requiring per-host calibration.

---

## Experiment 10: Five-Network Per-Layer Breakdown (v4 Model)

> **Note — raw vs. fused rates:** the table below reports *raw* per-layer
> alert rates measured independently on each layer's output. The paper's
> Table 9 (Per-Network Event-Level Performance) reports *fused* rates after
> priority-ordered de-duplication, in which lower-priority signals are
> suppressed when P4 fires. The two views are consistent: the raw P4 rate of
> 92.2% matches the 92% temporal detection figure reported for Round 3, while
> the fused P4 rate is 67.4%.

| Network | Agent Active (%) | Benign Nighttime (%) | Signal Ratio |
|---------|----------------:|---------------------:|-------------:|
| P0 (adaptive-high) | 0 | 0.03 | — |
| P1 (pattern) | 4.2 | 0.6 | 7.0× |
| P2 (context) | 0.3 | 1.5 | 0.2× |
| P3 (rarity) | 0.2 | 0.9 | 0.2× |
| **P4 (temporal)** | **92.2** | **8.9** | **10.4×** |
| P5 (adaptive-low) | 3.1 | 12.8 | 0.2× |

P4 (temporal) is the single strongest discriminator with a 10.4× signal ratio.

---

## Experiment 11: Adversarial Round Summary

### Round 1: Initial Detection (Three-Layer)
| Prediction | Outcome |
|-----------|---------|
| PARENT chain = fatal flaw | **FALSIFIED** — sh -c indirect execution grants natural immunity |
| Prototype net catches known patterns | **FALSIFIED** — zero hit (benign morphology effective) |
| DT temporal signal detectable | **CONFIRMED** — +38 events (strongest signal) |
| T1053 persistence collision | **CONFIRMED** — +3 hits (auto-remediation level) |

### Round 2: Agent Adaptation (v4)
- Morphological transformation engine (concept level)
- Disguise name pool expansion
- SUID-based privilege escalation path
- Memory-only exfiltration (no disk artifacts)

### Round 3: Anti-Adaptive Upgrade
- Second-order temporal analyzer (CV/tail_ratio/eps_corr)
- Adaptive event detector (transformation/C2/SUID/recon/sleep)
- Five-network fusion with confidence tiering
- Cross-host universal prior (7-VM joint training)
- Self-learning pattern extraction pipeline

---

## Experiment 12: Model Characteristics

| Property | Value |
|--------|-------|
| Parameters | 0.88M |
| Architecture | 4-layer causal Transformer |
| Hidden dimension | 128 |
| Attention heads | 4 |
| Context window | 128 events |
| Vocabulary | 263 tokens |
| Training data | 116K events, 7 VMs |
| Validation perplexity | 1.6 |
| Training time | ~3 min (H800 GPU) / ~11 min (CPU, 1000 batches) |
| Inference | CPU real-time (<1ms per event) |
| Model file size | 3.6 MB |
| Codebook size | ~80 KB (14 techniques × 3 prototypes) |

---

## Experiment 13: Behavior Sequence Extraction Statistics

| Metric | Value |
|--------|-------|
| Total telemetry events | 181,437 |
| Candidate events (sh/bash parent, non-system) | 70,209 |
| Extracted sequences | 49 |
| Sequence length (median) | 52 events |
| Interval CV (median) | 1.126 |

### Action Primitive Distribution

| Primitive | Count | % |
|-----------|------:|----:|
| probe | 29,660 | 42.3% |
| recon | 28,391 | 40.4% |
| persist | 5,874 | 8.4% |
| unknown | 5,556 | 7.9% |
| exfil | 285 | 0.4% |
| evasion | 186 | 0.3% |
| privesc | 124 | 0.2% |
| communicate | 55 | 0.1% |
| impact | 40 | 0.1% |
| timing | 38 | 0.05% |

---

## Reproducibility

All experiments are reproducible using:
```bash
# Ablation + τ sweep + synthetic + cross-host
python3 supplementary_experiments.py

# End-to-end pipeline test
python3 test_e2e_v3.py

# Prototype training
python3 proto_head.py models/vm-universal --k 3 --epochs 200

# Self-learning pattern extraction
python3 auto_pattern.py models/vm-universal --from-patterns

# Onboard calibration
python3 onboard_v2.py models/vm-universal data/onboard_benign.jsonl
```

---

*Generated: 2026-08-01 | Zihan Luo | All experiments in isolated libvirt environment*
