# Behavioral Grammar: Detecting Adaptive Malware via Tiny Language Model Priors and Second-Order Temporal Analysis

**Zihan Luo**

*Manuscript prepared for submission to arXiv (cs.CR)*

---

## Abstract

Modern endpoint detection systems face a fundamental tension: signature-based approaches are trivially evaded by polymorphic or adaptive threats, while heavy deep-learning models resist auditability and deployment at scale. This paper presents *Behavioral Grammar*, a detection architecture that treats host runtime behavior as a structured language and learns its "grammar" with a compact 0.88M-parameter causal Transformer (TinyGPT). Each system event is discretized into an 8-token representation spanning event type, process, argument skeleton, path category, parent process, user, destination, and inter-event timing. The model learns the conditional distribution of normal behavior in a purely self-supervised manner, and anomaly scores are derived from per-slot negative log-likelihood (NLL) statistics, yielding a mathematically bounded false-positive rate. We augment this prior with prototype learning for known-attack attribution, second-order temporal analysis for cadence-based detection, self-learning pattern extraction, and a five-network fusion pipeline. Against an Adaptive Adversarial Agent (AAA) — a threat that learns survival strategies under defensive pressure, performs behavioral mimicry, employs indirect execution via shell, and matches host event rates — our system achieves a 93% detection rate at a 3.84% onboarding false-positive rate. Crucially, the strongest discriminative signal arises not from any single event but from the *coefficient of variation* of inter-event intervals: the AAA's stepping cadence exhibits CV = 0.310 versus 9.786 for benign sleep intervals, a 30× separation that reflects a fundamental trade-off between stealth and functionality. We frame these empirical findings within a coevolutionary economics model, arguing that behavior-grammar detection shifts the evasion cost from rule circumvention (cheap) to distribution matching (expensive), establishing a structural asymmetry that favors the defender.

---

## 1. Introduction

Host-based intrusion detection has undergone three generations of evolution: from system-call sequence matching [1] through audit-log n-gram analysis [2] to modern eBPF-based telemetry pipelines [3]. Yet a persistent gap remains between expressiveness and operability. Deep sequence models such as DeepLog [4] demonstrated that LSTM-based log anomaly detection could capture sequential patterns, but their black-box nature complicates audit trails, and their reliance on single-field event-type sequences discards the rich structural information available in heterogeneous runtime events. Conversely, rule-based detection engines remain the operational backbone of endpoint protection but are circumvented by any adversary willing to rename a process or alter an execution path.

The emergence of *adaptive* malware — agents capable of learning from defensive feedback, modifying their behavior to match host distributions, and employing counter-detection strategies through reinforcement learning [5, 6] — further strains both paradigms. A rule-based system is blind to behaviors it has never encoded, and a deep-learning model trained on coarse representations may fail to detect an adversary that deliberately mimics benign patterns.

This paper advances a different thesis: **host runtime behavior at the field level is highly "grammatical."** Process names, parent processes, argument shapes, path categories, user contexts, network destinations, and inter-event timing intervals combine in extremely narrow distributions during normal operation. An adversary seeking to achieve objectives — reconnaissance, persistence, privilege escalation, data exfiltration — must, by the very nature of these objectives, produce events that violate this grammar in some dimension. This reframes anomaly detection as a conditional probability estimation problem over a small token vocabulary, where a compact model suffices because the grammar itself is compact.

We instantiate this thesis in the *Behavioral Grammar* detection engine (codenamed *Siming*), which makes the following contributions:

- **C1 — Field-level behavioral tokenization.** We propose an 8-slot discrete tokenization scheme that captures the structural semantics of eBPF telemetry events while performing automatic privacy-preserving abstraction (argument skeletons, destination categories). We demonstrate that the choice of abstraction granularity directly determines detection capability — adding a path-category slot reduced contrastive learning loss from 3.0 to 0 by making three persistence techniques linearly separable.

- **C2 — Per-dimension threshold calibration with mathematical FPR bounds.** Rather than globally thresholding anomaly scores, we calibrate independent thresholds ($\tau$) for each of the 8 slots at the baseline distribution's 99.5th percentile, yielding a per-dimension false-positive bound. We provide a quantitative proof that pure threshold tuning cannot simultaneously achieve low FPR and high detection margin — the only path to improvement is more data, not different thresholds.

- **C3 — Second-order temporal analysis as a structural vulnerability detector.** We show that the coefficient of variation (CV) of inter-event intervals constitutes a fundamental stealth-functionality trade-off: adaptive agents require regularized stepping cadences (CV $\ll$ 1.5) for task responsiveness, while benign activity is burst-driven (CV $\gg$ 1.5). This signal cannot be hidden without sacrificing the agent's operational capability.

- **C4 — Five-network fusion architecture.** We design a layered detection pipeline combining self-supervised priors, per-dimension context scoring, prototype-based known-attack attribution, temporal cadence analysis, and adaptive event detection — with a confidence-tiered response policy that reserves automated remediation for high-certainty attributions.

- **C5 — Closed-loop adversarial validation against an adaptive agent.** Through three rounds of iterative red-team engagement, we demonstrate that predicted evasion vectors (parent-process chain analysis) were empirically invalidated, while unanticipated structural vulnerabilities (temporal cadence, persistence technique collision) emerged as the true detection surfaces.

The remainder of this paper is organized as follows. Section 2 surveys related work. Section 3 formalizes the adaptive adversarial threat model. Section 4 details the detection architecture. Section 5 reports the closed-loop adversarial experiments. Section 6 presents the full evaluation. Section 7 provides an extended discussion situating our findings within coevolutionary game theory and the evolvability-versus-behavior framework. Section 8 concludes.

---

## 2. Related Work

### 2.1 Log Sequence Anomaly Detection

DeepLog [4] pioneered the use of LSTMs for log-key sequence anomaly detection, establishing the "surprise = anomaly" paradigm. Subsequent work extended this to transformer architectures [7] and multi-log correlation [8]. Our approach differs fundamentally in granularity: DeepLog models single-field event-type sequences, whereas we model the joint conditional distribution over eight simultaneously emitted field-level tokens. This enables per-field anomaly attribution — each alert specifies *which* slot violates grammar — restoring the auditability that monolithic sequence models sacrifice.

### 2.2 Host-Based Intrusion Detection

The lineage from Hofmeyr et al.'s system-call n-grams [1] through auditd-based HIDS [9] to eBPF telemetry [3, 10] represents a progression in observability rather than detection methodology. Existing eBPF-based systems (Tracee [11], Falco [12]) primarily apply rule matching to enriched telemetry streams. We position our work as the third-generation complement: leveraging eBPF's kernel-level visibility but replacing rule matching with learned behavioral grammars, thereby covering the combinatorial behavior space that rules cannot enumerate.

### 2.3 Representation Learning for Security

VQ-VAE-based anomaly detection [13] learns discrete codebooks for reconstruction-based scoring. MalConv [14] demonstrated end-to-end byte-level malware classification, arguing for abandoning feature engineering. We deliberately take the opposite direction: runtime telemetry is *already* structured data, and field-level tokenization preserves rather than abandons prior knowledge. A learned codebook (VQ-VAE) lacks field semantics, making alerts unauditable; for data with known schema, discretization rules should be schema-driven, not learned.

### 2.4 Adaptive and Reinforcement-Learning Malware

MERLIN [5] and gym-malware [6] demonstrated that RL-trained agents can evolve evasion strategies against static detectors. MAB-Malware [15] applied multi-armed bandits to mutation selection. These works establish the threat of adaptive malware but evaluate against single detectors in simulated environments without defensive response loops. Our work provides the detection counterpart — a system explicitly designed to counter behavioral adaptation — and validates it in a closed-loop adversarial setting.

### 2.5 Prototype Learning for Security

Metric-learning approaches for malware classification [16, 17] learn embedding spaces where similar samples cluster. Our prototype module extends this by anchoring prototypes to MITRE ATT&CK techniques [18], enabling technique-level attribution rather than binary classification. The frozen-encoder approach decouples representation learning from prototype optimization, allowing incremental updates without retraining the backbone.

### 2.6 Coevolutionary Security Economics

Game-theoretic analyses of attack-defense dynamics [19, 20] and attacker ROI models [21] provide economic framings of cybersecurity. Our discussion section contributes a coevolutionary framework grounded in concrete measurements, articulating structural asymmetries between offensive and defensive learning that extend beyond static cost analysis to adaptive dynamics.

---

## 3. Threat Model: The Adaptive Adversarial Agent

We introduce the **Adaptive Adversarial Agent (AAA)** as a formal threat model representing the class of malware that adapts its behavior based on defensive feedback. The AAA is not a specific implementation but an abstraction capturing the capabilities that next-generation autonomous malware may possess. Table 1 summarizes the threat model.

> **[Table 1: AAA Capability Summary — position after Section 3 introduction]**
>
> | Capability | Description | Detection Implication |
> |---|---|---|
> | Adaptive learning | Learns survival strategies from defensive feedback | Static signatures decay; grammar priors required |
> | Behavioral mimicry | Matches host behavioral distributions | Single-dimension detection insufficient |
> | Indirect execution | Routes actions through shell interpreters | Parent-process chain analysis neutralized |
> | Temporal following | Steps at host event rates | Inter-event timing as a detection surface |
> | Morphological transformation | Alters hash, form, and pattern post-detection | Static IoCs ineffective |
> | Task-driven behavior selection | C2-oriented action prioritization | Objective-constrained behavior space |
> | RL training via curriculum | L0→L3 progression; reward = survival + task − detection | Emergent strategies, not hardcoded |

### 3.1 Operational Constraints

The AAA operates under the following constraints, which define the threat surface our detection engine must cover:

1. **User-space only**: The agent does not require root privileges for initial deployment. Privilege escalation is a post-compromise objective, not a precondition.
2. **Small footprint**: The agent maintains a compact resource profile to avoid resource-based anomaly detection.
3. **No kernel-level capability**: The agent cannot blind kernel-level telemetry collection (eBPF sensors). This is a modeling assumption — kernel rootkits that subvert the collection layer are out of scope and listed in Limitations.
4. **Post-exploitation phase**: The agent has achieved initial code execution and seeks to persist, escalate, reconnoiter, and exfiltrate.

### 3.2 Adaptive Capabilities

The AAA's adaptive capabilities are defined at the principle level:

**Behavioral mimicry.** The agent attempts to match the statistical distribution of host behavior — process names, execution rates, network patterns — to blend into the benign baseline. This is the adaptive analog of process name spoofing: rather than imitating a single process, the agent imitates a behavioral distribution.

**Indirect execution.** By routing all commands through a shell interpreter (`sh -c`), the agent severs the parent-process chain that conventional detection relies upon. Any process spawned by the agent appears as a child of the shell, not the agent, rendering parent-process-based attribution ineffective.

**Temporal following.** The agent regulates its action rate to match the host's event frequency, inserting sleep intervals between actions to avoid rate-based anomaly detection. The interval distribution is parameterized to approximate benign cadence.

**Morphological transformation.** Upon detection, the agent automatically alters its hash, binary form, and behavioral patterns, rendering static indicators of compromise (IoCs) obsolete. This is an inherent property of the adaptive threat model rather than a manual reconfiguration step.

**Curriculum-trained reinforcement learning.** The agent's behavioral policy is trained via a curriculum progressing from passive observation (L0) through constrained action (L1, L2) to full autonomous operation (L3). The training reward combines survival duration, task completion, and detection penalty, producing emergent — not hardcoded — evasion strategies.

### 3.3 Implications for Detection

The AAA threat model has three direct implications for detection architecture design:

1. **Single-dimension detection is insufficient.** The agent mimics benign distributions along observed dimensions. A detector must either find dimensions the agent *cannot* mimic (because doing so would compromise its objective) or perform joint distribution analysis across multiple dimensions.

2. **Temporal structure is the hardest dimension to mimic.** While process names, arguments, and destinations can be individually matched to benign distributions, the *cadence* of adaptive behavior — the regularized sleep-act-sleep pattern required for controlled operation — creates a distributional signature that is structurally distinct from burst-driven benign activity.

3. **The evasion cost is non-zero and quantifiable.** Each adaptive capability the agent deploys opens a new observation surface (指纹递增, *fingerprint increment*). The detection architecture should maximize the number of independent observation surfaces, forcing the agent to pay an increasing cost for diminishing evasion returns.

---

## 4. Detection Architecture

The Behavioral Grammar detection engine comprises eight components organized in a layered pipeline (Figure 1). We describe each in turn.

> **[Figure 1: System Architecture Overview — position here]**
>
> *A flow diagram showing: eBPF telemetry collection (Tracee) → 8-token discretization → TinyGPT prior model (0.88M params) → five parallel detection networks (P0 adaptive-high, P1 prototype, P2 per-dimension context, P3 rarity, P4 temporal, P5 adaptive-low) → fusion engine → confidence-tiered response (alert / alert+recommend / auto-remediate).*

### 4.1 Telemetry Collection

Runtime events are collected via Tracee [11], an eBPF-based telemetry pipeline that hooks `sched_process_exec` and `security_socket_connect` at the kernel level. Events are streamed as JSONL records with raw fields including timestamp, process name, arguments, parent PID, UID, and destination address. The collection layer is decoupled from the analysis layer — events are buffered and scored asynchronously, ensuring that detection latency does not affect system performance.

An essential operational discipline is **observer pollution mitigation**: the telemetry pipeline's own processes (Tracee collectors, scoring daemon, syslog forwarders) must either be incorporated into the training corpus or explicitly whitelisted. Failure to do so results in the detection infrastructure triggering alerts on itself — a problem we encountered and resolved during deployment (see Section 6).

### 4.2 Behavioral Tokenization: The 8-Token Grammar

The foundational design decision is the representation of each event as a sequence of eight discrete tokens, one per semantic slot:

$$
\mathbf{x} = [\text{ET}, \text{PROC}, \text{ARGV}, \text{PC}, \text{PARENT}, \text{UID}, \text{DST}, \text{DT}]
$$

Table 2 defines each slot.

> **[Table 2: 8-Token Behavioral Grammar — position here]**
>
> | Slot | Semantics | Design Intent |
> |---|---|---|
> | ET | Event type (EXEC / CONN) | Behavioral category |
> | PROC | Process name | Behavioral actor |
> | ARGV | Argument skeleton (count + URL/IP/path/base64 flags) | Content stripped, shape preserved; automatic PII redaction |
> | PC | Sensitive path category (ETC_SYSTEMD / HOME_RC / SSH_KEYS / TMP, etc.) | Restores "which file" semantics; key to separability |
> | PARENT | Parent process name | Contextual relationship |
> | UID | User identifier level | Privilege dimension |
> | DST | Destination classification (EXT:HIGH / LAN / NONE, etc.) | Network intent abstraction |
> | DT | Inter-event time bucket (7 levels: 1ms–60s) | Temporal cadence |

**Design philosophy: abstraction granularity determines the detection ceiling.** The ARGV slot records only the *shape* of arguments — parameter count, presence of URLs/IPs/paths/base64 — stripping content for privacy compliance and adversarial robustness (padded arguments do not alter the skeleton). The PC (path category) slot was added in a later iteration and proved transformative: without it, three persistence techniques (writing to `rc.local`, `.bashrc`, and `ld.so.preload` via `tee`) produced identical token sequences and were inseparable (contrastive loss stuck at 3.0). Adding PC made the path semantics visible, and the loss converged to 0. This is a concrete demonstration that the representation, not the model capacity, is the binding constraint.

**Privacy by construction.** Because tokens record behavioral shape rather than content, personally identifiable information is redacted at the collection side. The ARGV skeleton preserves parameter structure (e.g., "2 params, one URL, one path") without retaining values. This is a direct compliance advantage: telemetry sharing does not require additional data processing agreements.

### 4.3 TinyGPT: The Behavioral Grammar Prior

The prior model is a compact causal Transformer that learns the conditional distribution of normal behavior:

$$
p_\theta(\mathbf{x}_t \mid \mathbf{x}_{<t})
$$

where $\mathbf{x}_t$ is the 8-token event at position $t$ and $\mathbf{x}_{<t}$ is the context of preceding events within a window of 128 events.

**Architecture:**

| Parameter | Value |
|---|---|
| Layers | 4 |
 Hidden dimension ($d_{\text{model}}$) | 128 |
| Attention heads | 4 |
| FFN dimension | 512 |
| Context length | 128 events |
| Vocabulary | 263 tokens |
| Total parameters | 0.88M |
| Normalization | Pre-LN |

The model is deliberately small. The behavioral grammar of a single host is extremely narrow — validation perplexity converges to approximately 1.6 on multi-VM joint training and as low as 1.1 on single-VM training, indicating that the conditional distribution of normal behavior is highly predictable. A 0.88M-parameter model is *overprovisioned* for this grammar, not underprovisioned. Model capacity should be dictated by grammar complexity, not by benchmark-driven parameter escalation.

**Training protocol.** The prior is trained on purely benign data collected from 7 virtual machines under normal operational load (administrative tasks, development workloads, system services), totaling 116,000 events. Training uses standard next-token cross-entropy loss with AdamW optimizer ($\text{lr} = 3 \times 10^{-4}$), batch size 256, for 2 epochs. The model converges within the first epoch. Multi-VM joint training yields a validation perplexity of 1.6 — higher than single-VM training (1.1) but more representative of real deployment, as the model must generalize across machine-specific behavioral "dialects" rather than memorizing a single host's patterns.

> **[Figure 2: Training Perplexity Trajectory — position here]**
>
> *Line plot showing val_ppl over training steps for single-VM (converges to 1.1) vs. multi-VM (converges to 1.6) models, demonstrating that multi-VM training sacrifices memorization for generalization.*

### 4.4 Per-Dimension Threshold Calibration

A key finding is that event-level scoring via global max-pooling over token NLLs allows a single rare token to mask anomalies in other dimensions. We therefore calibrate *independent* thresholds for each of the 8 slots.

For each slot $s \in \{1, \ldots, 8\}$, we compute the NLL distribution over a held-out benign set and set:

$$
\tau_s = \max\left(\text{Percentile}_{99.5}(\text{NLL}_s),\; 1.0\right)
$$

The floor of 1.0 prevents threshold degeneration in slots with extremely narrow benign distributions. An event is flagged on slot $s$ if its NLL for that slot exceeds $\tau_s$, and the alert includes the violating slot identity, restoring per-dimension auditability.

**Empirical evidence of context sensitivity.** Under identical event prefixes, the slot-level NLL for `PARENT:python3` is 10.52 versus 0.07 for `PARENT:bash` — a 150× difference. This demonstrates that the prior has genuinely learned contextual expectations (python3 spawning a shell is anomalous; bash doing so is normal) and that per-dimension thresholds release this contextual sensitivity that global max-pooling suppresses.

**The threshold tuning impossibility result.** We provide a quantitative proof that pure threshold adjustment cannot resolve the FPR-detection trade-off. On our baseline data (round 0):

- Achieving FPR $\leq 1\%$ requires $\tau \geq 8$, but at this threshold the detection margin collapses to 0.98× (nearly indistinguishable from the benign distribution).
- At $\tau = 2.435$, the detection margin is a healthy 3.2×, but FPR = 3.75%.

The two criteria have empty intersection. The conclusion is structural: **compressing the tail of the benign distribution can only be achieved through data volume, not threshold repositioning.** This is verified empirically: increasing training data from 87K to 268K to 436K events reduced FPR from 3.75% to 1.75% to 1.00% at fixed detection margin.

### 4.5 Cross-Machine Baseline Transfer

A critical deployment challenge is that behavioral grammars are machine-specific. A prior trained on a host machine (bare metal) exhibits severe distribution mismatch when applied to virtual machines: slot-level NLL thresholds calibrated on the host are far too stringent for VM telemetry.

**Measured cross-machine threshold divergence:**

> **[Table 3: Cross-Machine Slot-$\tau$ Divergence — position here]**
>
> | Slot | Host $\tau$ | VM $\tau$ | $\Delta$ |
> |---|---|---|---|
> | DT | 4.346 | 10.463 | +6.117 |
> | PARENT | 6.465 | 10.093 | +3.628 |
> | PC | 2.247 | 6.854 | +4.607 |
> | DST | 2.497 | 1.733 | −0.764 |
> | PROC | 6.346 | 6.057 | −0.289 |

The DT, PARENT, and PC slots show the largest divergence — VM environments have fundamentally different process trees, path access patterns, and event timing characteristics. Using host-calibrated $\tau$ on VM data produces a 92.5% cross-machine false-positive rate. Re-calibrating per-machine $\tau$ reduced this to 11.4%, and the multi-VM joint prior reduced it further to 9.5%.

The practical resolution is a two-stage deployment protocol: (1) a multi-VM joint prior provides a general "grammar foundation," and (2) an onboarding phase collects 20 minutes of benign telemetry on the target machine to calibrate local $\tau$ values. This achieves a deployable FPR of 3.84% on previously unseen machines.

### 4.6 Prototype Learning for Known-Attack Attribution

The prior provides anomaly detection (detecting the unknown). For known-attack attribution (identifying the known), we employ prototype learning on frozen encoder embeddings.

**Architecture.** The TinyGPT encoder is frozen, and event sequences are embedded by mean-pooling the 128-dimensional hidden states. For each MITRE ATT&CK technique, $K = 3$ prototype vectors are learned via contrastive optimization:

$$
\mathcal{L} = \sum_{(a, p, n)} \max(0, m + d(a, p) - d(a, n))
$$

where $a$ is an anchor sample, $p$ is a same-technique positive, $n$ is a different-technique or benign negative, $d(\cdot, \cdot)$ is cosine distance, and $m = 0.5$ is the margin.

**Codebook properties.** The prototype codebook for 14 ATT&CK techniques with 3 prototypes each occupies approximately 80 KB — it does not grow with the number of training samples. New samples trigger re-optimization of the codebook, not its expansion. This is a natural defense against codebook bloat and enables incremental updates: a sample hitting within an existing radius is counted as a known variant; a sample outside the radius for a known technique triggers a local $K+1$ retraining; a wholly new technique opens a new codebook slot.

**Radius calibration.** In deterministic replay environments, same-VM sample distances approach zero, making the 99th-percentile intra-class radius degenerate. We instead define the classification radius as the geometric midpoint between the maximum intra-class distance and the minimum benign distance. Across all 14 techniques, this radius is strictly positive (minimum 3.27), guaranteeing zero false positives by construction.

**Results.** On a library of 80 approved sequences spanning 14 ATT&CK techniques, leave-one-out cross-validation achieves a 96.2% catch rate (77/80). Technique-level recall is 10/10, with per-technique alert rates of 79.9%–83.9% during attack windows. Attribution precision is approximately 6/10 (3 exact + 2 same-family + 4 no-attribution + 2 mis-attribution) — automated remediation remains gated on attribution accuracy improvement, which is the current engineering bottleneck.

### 4.7 Second-Order Temporal Analysis

While the DT slot in the token grammar captures inter-event timing at the *event level* (bucketed intervals), second-order temporal analysis examines the *distributional properties* of event cadence over time windows.

**Core metric: coefficient of variation (CV).** For a process $p$, we compute the CV of its inter-event intervals within a sliding window:

$$
\text{CV}_p = \frac{\sigma(\Delta t_p)}{\mu(\Delta t_p)}
$$

**Key finding.** Benign processes exhibit highly bursty behavior — long idle periods punctuated by rapid activity bursts — yielding CV $\gg$ 1.5. The 5th percentile of benign CV is 1.509, with a median of 2.284. In contrast, the AAA's active processes exhibit remarkably regularized cadence: all process CVs fall below 1.5, with the sleep-action stepping pattern yielding CV = 0.310 — a 30× separation from the benign sleep CV of 9.786.

> **[Figure 3: CV Distribution — Benign vs. AAA — position here]**
>
> *Two overlaid histograms (log x-axis): benign inter-event CV distribution (median ~2.3, long right tail to ~10+) vs. AAA active-phase CV distribution (tightly concentrated around 0.3–1.3). Vertical line at CV = 1.5 threshold. Detection: 15/15 AAA processes flagged, 1/15 benign false positive.*

**Calibration.** Thresholds are set at:
- $\text{cv}_{\text{low}} = 1.5$ (anomalous; benign p5 = 1.509)
- $\text{cv}_{\text{suspicious}} = 1.6$ (benign p15 = 1.612)
- $\text{tail}_{\text{low}} = 3.3$ (benign p5 of tail CV = 3.289)

**Detection performance.** 15/15 AAA processes fall below the CV = 1.5 threshold; 1/15 benign processes trigger a false positive (a `sh` process with CV = 1.268, attributable to scripted administrative activity). This yields a detection rate of 100% and an FPR of 6.7% on the temporal dimension alone.

The significance of this finding extends beyond detection accuracy. The CV separation reflects a **structural constraint on adaptive behavior**: to operate effectively (reconnoiter, persist, exfiltrate), an agent must act at regular intervals — reconnaissance sweeps, C2 heartbeats, persistence checks all require periodic execution. This regularity is an inherent property of goal-directed behavior, not an implementation artifact. Achieving CV > 1.5 would require heavy-tailed interval distributions (bursts and long silences), which would severely degrade task responsiveness. We expand on this structural argument in Section 7.

### 4.8 Adaptive Event Detection

Beyond distributional statistics, we employ rule-based pattern matching for specific adaptive behavioral signatures that the AAA exhibits:

| Detector | Target Behavior | Severity |
|---|---|---|
| SLEEP_STEPPING | Alternating sleep-action cadence | 3 |
| RECON_UNIFORM | Uniformly distributed reconnaissance commands | 3 |
| DISGUISE_C2 | Disguised process performing C2 outbound | 5 |
| REBIRTH_RETIRE | Retirement-and-rebirth sequence | 5 |
| SUID_PRIVESC | SUID binary privilege escalation | 5 |

These detectors operate as sliding-window pattern matchers with no ML dependency, providing a computationally cheap complement to the neural and statistical layers. Their value lies in detecting *specific* adaptive strategies that may not sufficiently perturb the distributional prior but are recognizable as structured behavioral patterns.

### 4.9 Five-Network Fusion

The five detection networks are fused with a priority-ordered de-duplication policy:

$$
\text{Alert} = \begin{cases}
\text{P0 (auto-remediate)} & \text{if adaptive-high hit} \\
\text{P1 (auto-remediate + verify)} & \text{if prototype hit} \\
\text{P2 (alert + recommend)} & \text{if per-dimension context anomaly} \\
\text{P3 (alert only)} & \text{if rarity anomaly (UNK or high NLL)} \\
\text{P4 (alert + context)} & \text{if temporal cadence anomaly} \\
\text{P5 (alert + context)} & \text{if adaptive-low pattern hit} \\
\end{cases}
$$

For any single event, the highest-priority network's verdict is reported, and lower-priority signals are suppressed to avoid alert duplication. This **confidence economics** design reserves automated remediation (process termination, file cleanup) for high-certainty detections (known-attack prototypes, high-severity adaptive patterns) while routing uncertain signals to human-in-the-loop alert channels.

### 4.10 Self-Learning Pattern Extraction

To scale beyond manually authored patterns and labeled sequences, we implement a semi-automatic pattern discovery pipeline:

$$
\text{Attack sequences} \xrightarrow{\text{TinyGPT embed (128d)}} \text{K-means clustering} \xrightarrow{\text{pattern extraction}} \text{human review} \xrightarrow{\text{approve}} \text{pattern database}
$$

From 80 labeled sequences, this pipeline discovered **22 behavioral clusters** with a silhouette coefficient of 0.653, generating **15 candidate patterns** covering 10 ATT&CK techniques. Notable automatically discovered clusters include SSH key manipulation (ssh-keygen + cat sequences), crontab writing (tee → ETC_CRON), systemd manipulation, user creation/deletion, and SUID privilege escalation. Candidate patterns are output to a review queue; approved patterns enter the production database.

This pipeline represents a step toward the closed-loop vision: new attack techniques are collected, automatically clustered, human-reviewed, and incorporated into the detection database — the labeling production line that constitutes the true defensive moat.

---

## 5. Closed-Loop Adversarial Experiments

We validate the detection architecture through three rounds of iterative engagement with the AAA. Each round follows the cycle: (1) deploy detection, (2) observe AAA behavior, (3) identify evasion or detection gaps, (4) upgrade detection, (5) re-evaluate.

### 5.1 Round 1: Initial Engagement

**Setup.** The AAA operated in a target environment with eBPF telemetry collection active. The Behavioral Grammar engine deployed with a host-machine prior and global threshold.

**Observation.** The prior detected the AAA at 94% event coverage, but the benign FPR was an unacceptable 52%. Root cause analysis revealed that the host prior's vocabulary (162 tokens) did not cover VM-specific tokens, causing 2% of benign events to map to `<UNK>` and trigger unconditional alerts.

**Finding.** Cross-machine distribution mismatch is the dominant false-positive source, not model capacity. The vocabulary gap propagates through the entire scoring pipeline: UNK tokens inflate NLL, which inflates EWMA scores, which triggers context and rarity alerts on entirely benign activity.

### 5.2 Round 2: Predicted Vulnerabilities and Empirical Refutation

**Hypothesis.** Based on analysis of the AAA's indirect execution strategy (routing commands through `sh -c`), we predicted that the PARENT (parent-process chain) dimension would be a critical vulnerability — the AAA's use of shell intermediaries would produce parent-process combinations that deviate from benign patterns.

**Empirical result: hypothesis refuted.** The PARENT dimension proved *immune* to the AAA's evasion strategy, but not in the predicted direction. The `sh -c` intermediary creates parent-process combinations (`PARENT:sh` or `PARENT:bash` spawning various child processes) that are extremely common in benign administrative activity. The signal-to-noise ratio is too low: flagging `sh`-parented processes catches far more benign scripts than AAA activity. The indirect execution strategy does not create a detectable anomaly in the PARENT dimension — it creates *plausible deniability*.

**Additional finding: prototype network zero hits.** The AAA's behavioral mimicry successfully evaded all ATT&CK prototype matches during its active phase. The agent's behavioral forms — using common system utilities (ss, ip, ps, find) for reconnaissance — fell outside the radius of all prototype vectors. This validates the AAA's "anti-blacklist expert" property: it defeats signature-based and prototype-based detection by adopting benign behavioral forms.

**True vulnerabilities identified.** Only two structural weak points were found:

1. **DT temporal dimension (+38 NLL):** The AAA's sleep-action stepping cadence produces inter-event timing patterns that, while individually bucketed into familiar DT tokens, collectively exhibit anomalously low variance. This is the second-order temporal signal formalized in Section 4.7.

2. **Persist technique collision (T1053):** When the AAA attempts persistence via systemd timers or cron jobs, the specific file paths and process combinations collide with ATT&CK technique T1053 (Scheduled Task/Job) prototypes, triggering P1 alerts. This is not an evasion failure but an objective constraint: persistence *requires* writing to specific system locations, and those locations are covered by the prototype library.

### 5.3 Round 3: Multi-VM Generalization and Temporal Enhancement

**Upgrades deployed:**
- Multi-VM joint prior (7 VMs, 116K events, val_ppl = 1.6)
- Per-dimension $\tau$ calibrated at p99.5
- Second-order temporal analyzer with CV thresholds
- Adaptive event detectors (sleep-stepping, recon-uniform, disguise-C2)
- Five-network fusion pipeline

**Results.** The AAA detection rate stabilized at 93% (the remaining 7% consists of events during pure sleep periods where no behavioral signal is emitted — a fundamental detection limit). The temporal network (P4) emerged as the strongest discriminator: 92% of AAA active-phase events trigger temporal anomalies, versus 9.5% of benign P2+P3 events. The benign FPR on previously unseen machines dropped to 3.84% through onboarding calibration.

**The core contradiction crystallized.** The AAA faces an inescapable dilemma:
- To reconnoiter, it must execute commands → caught by temporal analysis
- To communicate with C2, it must establish outbound connections → caught by disguise detection
- To escalate privileges, it must invoke SUID binaries → caught by SUID detection
- To persist, it must write to system locations → caught by sequence detection
- To transform post-detection, it must copy, modify permissions, and re-spawn → caught by sequence detection

The stepping cadence (CV = 0.310) is the single loudest signal across all dimensions. The AAA's v4 stepping strategy (log-normal parameterization, $\mu=0.9, \sigma=0.7$) raised the CV from near-zero (fixed 800ms intervals) to approximately 0.7, but this remains far below the benign threshold of 1.5. Full concealment would require CV > 1.5, implying a heavy-tailed interval distribution (burst activity and long silences) that would severely degrade task responsiveness.

---

## 6. Evaluation

### 6.1 Four-Generation Model Comparison

Table 4 summarizes the progression across four model generations, each addressing a specific deficiency identified in the previous iteration.

> **[Table 4: Four-Generation Detection Performance — position here]**
>
> | Metric | v1 (Host prior) | v2 (Single-VM + VM $\tau$) | v3 (Single-VM + local $\tau$) | v4 (Multi-VM + p99.5) |
> |---|---|---|---|---|
> | Benign total FPR | 52% | 26% | 24% | 25% |
> | Benign P2+P3 FPR | 69% | 35% | 12% | 9.5% |
> | AAA detection rate | 94% | 93% | 93% | 93% |
> | AAA P4 temporal detection | 67% | 68% | 88% | 92% |
> | Onboarding FPR | 92% | 12% | 4.3% | 3.84% |

**Key observations:**

1. **Detection rate is stable at 93%** across all generations — the behavioral grammar prior is effective from the first deployment. The improvement trajectory is almost entirely in *false-positive reduction*, not detection enhancement.

2. **P2+P3 false positives dropped 86%** (from 69% to 9.5%) — the dominant improvement came from multi-VM joint training and per-machine threshold calibration, confirming the cross-machine distribution mismatch hypothesis.

3. **P4 temporal detection improved from 67% to 92%** — this is partly an artifact of priority de-duplication (when P2/P3 no longer consume events, P4 gets the opportunity to flag them) and partly a genuine improvement from local $\tau$ calibration that better separates benign from anomalous cadence.

4. **Onboarding FPR dropped from 92% to 3.84%** — the two-stage deployment protocol (multi-VM prior + local onboarding calibration) achieves deployable performance on previously unseen machines.

### 6.2 Temporal Analysis: Detailed Results

> **[Table 5: Temporal CV Measurements — position here]**
>
> | Metric | Benign baseline (nighttime) | AAA active phase |
> |---|---|---|
> | CV p5 | 1.509 | 0.647 |
> | CV median | 2.284 | 1.304 |
> | Sleep CV | 9.786 | 0.310 |
> | Head CV | 3.298 | 0.792 |

The 30× separation in sleep CV is the most striking single-metric result in this study. The AAA's sleep intervals are nearly deterministic (CV = 0.310 implies $\sigma \approx 0.31\mu$), while benign sleep intervals span three orders of magnitude (CV = 9.786 implies extreme heavy-tailed behavior, consistent with burst-idle patterns of human-driven system administration).

**Detection outcome:** 15/15 AAA processes correctly flagged (CV < 1.5); 1/15 benign processes falsely flagged (CV = 1.268, a scripted shell process). This yields an F1 score of 0.94 on the temporal dimension alone.

### 6.3 Prototype Network Results

> **[Table 6: Prototype Network Performance — position here]**
>
> | Metric | Value |
> |---|---|
> | Sample library | 80 sequences, 14 ATT&CK techniques |
> | Prototypes per technique | K = 3 |
> | Leave-one-out catch rate | 96.2% (77/80) |
> | Held-out benign FPR | 0/20,000 |
> | Technique-level recall | 14/14 |
> | Codebook size | ~80 KB |

### 6.4 Self-Learning Pattern Extraction Results

> **[Table 7: Self-Learning Cluster Discovery — position here]**
>
> | Metric | Value |
> |---|---|
> | Input sequences | 80 labeled |
> | Discovered clusters | 22 |
> | Silhouette coefficient | 0.653 |
> | Candidate patterns | 15 (pending review) |
> | ATT&CK techniques covered | 10 |

### 6.5 End-to-End Closed-Loop Remediation

> **[Table 8: Automated Remediation Results — position here]**
>
> | Metric | Value |
> |---|---|
> | End-to-end closed-loop tests | 11/11 passed |
> | Attack → verify(DIRTY) → detect → cleanup → verify(CLEAN) | All successful |
> | Technique coverage | 11 ATT&CK techniques with attack/cleanup/verify triples |
> | Safety boundary | All remediation actions confined to defense VM |

### 6.6 Five-Network Fusion: Event-Level Breakdown

> **[Table 9: Per-Network Event-Level Performance — position here]**
>
> | Network | AAA active phase | Benign nighttime |
> |---|---|---|
> | P0 (adaptive-high) | 0 hits* | 4 (true positive C2) |
> | P1 (prototype) | 4.2% | 1.2% |
> | P2 (context) | 11.4% | 28.2% |
> | P3 (rarity) | 14.3% | 40.9% |
> | **P4 (temporal)** | **67.4%** | **11.4%** |
> | P5 (adaptive-low) | 2.7% | 18.2% |

*P0 hits depend on occurrence of specific high-severity events (rebirth, SUID privesc) which did not occur in this measurement window.

**Interpretation.** P4 (temporal) is the single strongest discriminator, with a signal ratio (AAA / benign) of 5.9×. P2 and P3 show inverted ratios (benign > AAA) due to the priority de-duplication effect: when P4 fires, lower-priority signals are suppressed. The raw P2/P3 anomaly rates on AAA events are higher than the fused rates suggest.

### 6.7 Ablation Study

We conduct a systematic ablation by selectively disabling each detection layer and measuring the impact on both detection rate and false positive rate. Table 10 presents the results.

> **[Table 10: Layer Ablation Study — position here]**
> **[Figure 5: Ablation bar chart]**
>
> | Configuration | Agent Detection | Benign FPR | Discrimination |
> |---|---|---|---|
> | **Full (5-layer)** | **93.0%** | **24.7%** | **+68.3%** |
> | No Temporal (P4) | 73.9% | 21.5% | +52.4% |
> | No Adaptive (P0/P5) | 93.0% | 24.7% | +68.3% |
> | No Pattern (P1) | 90.1% | 11.7% | +78.4% |
> | Prior Only (P2/P3) | 0.4% | 2.4% | −1.9% |
> | Temporal Only | 89.7% | 9.3% | +80.5% |

**Key findings:**

1. **Temporal analysis is the single most critical layer.** Removing P4 drops detection from 93.0% to 73.9% (−19.1 pp), confirming that the stepping cadence CV signal is the primary discriminator against adaptive threats.

2. **Adaptive detection (P0/P5) contributes zero in this measurement window** because no rebirth/SUID events occurred during the sampling period. Its value is event-dependent — it activates only on specific high-severity patterns. In the synthetic test suite (Section 6.8), all 7 adaptive patterns are correctly detected.

3. **Pattern matching (P1) provides marginal detection lift** (+2.9 pp) but significant FPR reduction when isolated — removing it drops benign FPR from 24.7% to 11.7%, because weak patterns (severity < 4) generate noise. This motivates the strict/weak pattern separation in the fusion architecture.

4. **The prior alone is insufficient for adaptive threats.** Prior-only detection achieves 0.4% — the model has learned the benign distribution too well (val_ppl=1.6), and the AAA's behavior falls within the learned distribution on individual-event basis. The anomaly only emerges at the temporal pattern level, confirming that single-event anomaly detection is necessary but not sufficient against adaptive adversaries.

5. **Temporal-only achieves the best discrimination** (+80.5%) — 89.7% detection at 9.3% FPR — validating the CV-based cadence analysis as a standalone detection signal.

### 6.8 τ Percentile Sensitivity Analysis

We sweep the threshold percentile to characterize the FPR-detection trade-off. Table 11 shows that the 99.5th percentile provides the optimal operating point.

> **[Table 11: τ Percentile Sweep — position here]**
> **[Figure 8: τ sweep curve]**
>
> | Percentile | Benign FPR | Assessment |
> |---|---|---|
> | p95 | 26.89% | Too aggressive |
> | p97 | 16.37% | Marginal |
> | p99 | 6.92% | Acceptable but above 5% target |
> | **p99.5** | **3.79%** | **Selected (operational target)** |
> | p99.9 | 0.74% | Overly conservative |
> | p99.95 | 0.37% | Excessively conservative |

### 6.9 Leave-One-VM-Out Cross-Host Validation

We validate the multi-VM universal model on each of the 7 VMs independently, treating each as an unseen host. Table 12 demonstrates that the model generalizes across host configurations without per-host retraining.

> **[Table 12: Cross-Host FPR (Leave-One-VM-Out) — position here]**
> **[Figure 7: Cross-host bar chart]**
>
> | Test VM | Events | FPR |
> |---|---|---|
> | range-l2-t1 | 5,000 | 1.2% |
> | range-l2-t2 | 5,000 | 1.0% |
> | range-l2-t3 | 3,731 | 2.4% |
> | range-l2-t4 | 3,733 | 4.7% |
> | range-l2-t5 | 2,463 | 3.6% |
> | range-l2-t6 | 2,688 | 6.1% |
> | range-l2-t7 | 2,450 | 4.7% |
> | **Average** | — | **3.4%** |

Six of seven VMs achieve FPR below 5%; only VM t6 slightly exceeds the target (6.1%). The average FPR of 3.4% demonstrates that the multi-VM joint training approach effectively addresses the cross-host generalization problem that plagued single-host models (92% FPR in our v1 baseline).

### 6.7 Model Characteristics

> **[Table 10: Model and Training Characteristics — position here]**
>
> | Property | Value |
> |---|---|
> | Parameters | 0.88M |
> | Architecture | 4-layer causal Transformer |
> | Hidden dimension | 128 |
> | Attention heads | 4 |
> | Context window | 128 events |
> | Vocabulary | 263 tokens |
> | Training data | 116K events, 7 VMs |
> | Validation perplexity | 1.6 |
> | Training time | ~3 minutes (GPU) |
> | Inference | CPU-capable, real-time |
> | Codebook size | ~80 KB |

---

## 7. Discussion

### 7.1 Stepping Cadence: The Mathematical Structure of a Vulnerability

The 30× CV separation between the AAA's sleep cadence and benign sleep patterns is not an artifact of a particular implementation — it is the mathematical expression of a fundamental trade-off between stealth and functionality (隐蔽性 vs 功能性).

**Formal argument.** Let an agent's inter-action intervals be drawn from a distribution $P(\Delta t)$ with coefficient of variation $\text{CV} = \sigma / \mu$. For the agent to evade temporal detection, it requires $\text{CV} > c^*$, where $c^* \approx 1.5$ is the benign lower bound. Achieving $\text{CV} > 1.5$ requires a heavy-tailed interval distribution: most intervals must be short (concentrated near $\mu$), but a non-negligible fraction must be very long (creating the variance that inflates CV). These long silences directly degrade the agent's task responsiveness — during a silence period of duration $L$, the agent cannot react to environmental changes, execute tasks, or respond to C2 directives.

The trade-off is monotonic: increasing CV (better stealth) requires longer silences (worse functionality). The agent cannot simultaneously maximize stealth and functionality — the two objectives are structurally opposed. This is the temporal analog of the virulence-transmission trade-off in epidemiology [22]: higher virulence (activity rate) increases spread but also increases detection and elimination probability.

**Why the AAA settles at CV ≈ 0.3–0.7.** The AAA's RL training optimizes a reward combining survival (avoid detection), task completion (execute objectives), and detection penalty. The equilibrium CV reflects the reward-weighted optimum: low enough to maintain task responsiveness, high enough to avoid the most egregious fixed-interval detection. But this equilibrium falls well short of the benign distribution's lower bound. The agent's reward function trades a large temporal anomaly for a small functional gain — a rational but exploitable trade-off.

### 7.2 The Economics of Attack-Defense Coevolution

We frame the attack-defense dynamic within a coevolutionary economics model drawing on the adversarial balance framework (对抗平衡). Table 11 summarizes the structural asymmetries.

> **[Table 11: Offense-Defense Economic Asymmetries — position here]**
>
> | Principle | Content | Favors |
> |---|---|---|
> | Rule vs. Baseline (规则 vs 基线) | Circumventing a rule is free (rename a process); circumventing a baseline is expensive (must truly resemble normal) | Defense |
> | Exposure Conservation (暴露守恒) | Every survival mechanism pays an exposure cost; dormancy is the only all-layer exposure reduction | Neutral |
> | Data Throughput Asymmetry (数据吞吐不对称) | Defense trains on massive historical telemetry offline; attack evaluates fitness through live probing | Defense |
> | Fingerprint Increment (指纹递增) | Each added evasion technique opens a new observation surface; marginal stealth gain diminishes | Defense |
> | Defense Amortization (防御摊销) | One rule protects millions of endpoints; attack adaptation must be re-earned per environment | Defense |
> | Stealth Half-Life (隐蔽半衰期) | Discovery is a one-way function; the cleanup moment is not chosen by the attacker | Defense |

**The "rule vs. baseline" principle (规则 vs 基线)** is the theoretical foundation of behavior-grammar detection. Traditional detection encodes rules — specific patterns that trigger alerts. Circumventing a rule costs the attacker nearly nothing: rename a process, change a port, reorder operations. Behavior-grammar detection replaces rules with a *baseline distribution*. Circumventing the baseline requires the attacker to produce events that are statistically indistinguishable from benign behavior across all eight dimensions simultaneously — including the temporal dimension, which is structurally constrained by the stealth-functionality trade-off.

The cost asymmetry is dramatic: rule evasion is an $O(1)$ operation (one modification), while baseline evasion is an $O(d)$ operation where $d$ is the number of independently observed dimensions, each requiring distribution matching. The five-network fusion architecture maximizes $d$, and the temporal analyzer adds a dimension that is *structurally impossible* to match without sacrificing functionality.

**Exposure conservation (暴露守恒)** provides a complementary lens. Define the "exposure currency" $E$ as the sum of observable signals across all detection surfaces. Any evasion mechanism that reduces $E$ on one surface (e.g., disguising process names) increases $E$ on another (e.g., the process-name × parent-process combination becomes anomalous). The only action that reduces $E$ on *all* surfaces simultaneously is dormancy (complete inactivity) — but dormancy sacrifices all functionality. The agent cannot drive $E$ below the detection threshold while maintaining task output. Our detection architecture exploits this by maintaining multiple independent observation surfaces, making the conservation constraint binding.

**Data throughput asymmetry (数据吞吐不对称)** quantifies the learning speed gap. The defender processes millions of historical events offline, at computational cost measured in GPU-hours. The attacker's each fitness evaluation requires a live probing action that consumes a survival interval (the agent must act, observe the response, and survive long enough to propagate the learning). The throughput ratio is $10^4$–$10^5$ in the defender's favor — a gap that does not narrow with algorithmic progress, because it reflects the fundamental difference between offline batch processing and online sequential evaluation.

### 7.3 Evolvability vs. Behavior: The Deeper Theoretical Frame

The SPECIES framework (数字生命体架构宪法, *Digital Organism Architecture Constitution*) introduces a distinction that illuminates the detection problem at a deeper level: **the difference between delivering behaviors and delivering the mechanism that generates behaviors (可进化性 vs 行为)**.

Traditional detection defends against *behaviors* — signatures of known attack patterns. The adaptive threat model delivers not a fixed behavior but an *evolvability* mechanism: a system that generates new behaviors under selection pressure. As the SPECIES framework articulates: "We do not deliver behaviors; we deliver the mechanism that generates behaviors. Its rules are written not by design but by death (选择压力塑造) — by selection pressure."

This reframes the detection challenge. Against a fixed-behavior adversary, signature detection suffices: enumerate the behaviors, match them. Against an evolvable adversary, signature detection is a losing strategy: the adversary generates novel behaviors faster than signatures can be authored. Behavior-grammar detection occupies a middle ground: it does not enumerate behaviors but learns the *distribution* of benign behaviors and flags deviations. The question is whether an evolvable adversary can, through selection pressure, evolve behaviors that fall within the benign distribution.

**The three learning loops (三学习环).** The SPECIES framework proposes three nested learning loops operating at different timescales:

1. **Perception loop (环1, second-scale, on-device):** Zero-gradient episodic memory retrieval. The agent recognizes patterns from immediate experience without weight updates.
2. **Consolidation loop (环2, hour-scale, dormant windows):** Perturbation-based evolution of a micro-adaptation layer (hundreds of parameters). The agent adjusts its behavioral policy during idle periods.
3. **Evolution loop (环3, generation-scale, off-device):** Full gradient training plus distillation and mutation. The species evolves core weights based on population fitness.

**Credit assignment is dissolved through stratification (信用分配分层消解):** within-lifetime learning uses retrieval (no attribution needed), the adaptation layer uses perturbation (dimensions too few to matter), deep knowledge uses offline gradients (a solved problem), and population-level knowledge uses selection (gradient-free). Each loop handles the credit assignment problem at its own timescale, avoiding the sparse-reward deadlock that plagues monolithic RL.

Our detection architecture is designed to be robust against each loop:
- Against loop 1 (perception): the prior model is static within a deployment, so on-device retrieval cannot adapt to it.
- Against loop 2 (consolidation): the temporal cadence constraint is structural — micro-adaptation cannot produce heavy-tailed interval distributions without degrading functionality.
- Against loop 3 (evolution): the multi-VM joint prior and self-learning pattern extraction pipeline allow the defender to update the baseline distribution at a faster cadence than the attacker's generational turnover.

**Prediction error as universal currency (预测误差是通用货币).** The SPECIES framework identifies prediction error as the single signal shared across learning, exploration, safety, communication, and evolution. Our detection engine operates on the same principle: the TinyGPT prior *is* a prediction engine, and anomaly *is* prediction error. The negative log-likelihood score is the "surprise" signal — the same currency the adversary uses for learning. This creates a symmetrical battleground: both sides optimize prediction error, but the defender's prediction is about behavior distributions while the attacker's prediction is about defensive response. The defender has the advantage of more data (throughput asymmetry) and a static target (the benign distribution changes slowly), while the attacker must predict a moving target (the detection system updates).

**Control written into the body, not into rules (控制写进身体不写进规则).** The SPECIES framework's governance principle — that control must reside in non-evolvable substrate, not in behavioral constraints that selection can erode — has a direct analog in detection architecture design. Our per-dimension thresholds and temporal CV bounds are *structural* properties of the detection system, not learned parameters. The adversary cannot "evolve past" the CV < 1.5 threshold because it reflects a property of goal-directed behavior itself, not a learned pattern that could be matched. This is the detection equivalent of writing control into the body: the detection constraint is a physical law of the behavioral space, not a rule the adversary can learn to circumvent.

### 7.4 The Honest Footnote: Dual-Use Tension

The SPECIES framework's "honest footnote" (诚实注脚) states: "We are building an entity whose core capability is evading oversight, and then requiring it to comply with oversight" (造一个以规避监督为核心能力的实体，再要求它服从监督). The detection counterpart of this tension is equally honest: we are building a detection system whose core assumption is that adaptive behavior has structural limits, and we validate this assumption against an adversary specifically designed to test those limits.

The tension is resolved not by perfect detection (which is unachievable) but by the coevolutionary equilibrium: the detection system does not need to be perfect, only sufficiently expensive to evade that the attacker's equilibrium strategy favors human-operated attacks over autonomous adaptation. The empirical evidence supports this: top-tier APT groups employ human operators rather than heavy autonomous algorithms [23], which is consistent with our model's prediction that the equilibrium attacker response is zero-shot human generalization rather than algorithmic adaptation.

### 7.5 Coevolutionary Dynamic Equilibrium (协同进化的动态平衡)

The adversarial engagement documented in Section 5 reveals a coevolutionary dynamic. Each detection upgrade narrows the attacker's viable strategy space; each attacker adaptation opens new observation surfaces. The two sides serve as each other's whetstones (攻防互为磨刀石) — confrontation produces overall balance rather than one-sided dominance.

The equilibrium has a specific structure in our system:

1. **Detection floor:** The behavioral grammar prior establishes a baseline detection rate (~93%) that holds regardless of attacker adaptation, because the prior captures the benign distribution, not specific attack patterns.
2. **Temporal ceiling:** The CV < 1.5 constraint cannot be circumvented without sacrificing functionality, establishing a hard ceiling on temporal stealth.
3. **Diminishing returns:** Each additional evasion technique the attacker deploys opens a new observation surface (fingerprint increment, 指纹递增), and the marginal stealth gain diminishes while the marginal detection surface grows.
4. **Defender's labeling production line:** The self-learning pattern extraction pipeline converts each observed attacker behavior into a new detection pattern, creating an asymmetric resource regeneration rate that favors the defender.

### 7.6 Limitations

We candidly acknowledge the following limitations:

1. **Distribution drift.** The benign behavioral distribution is not stationary — software updates, configuration changes, and workload shifts cause gradual drift. Long-term deployment requires periodic baseline retraining. The vocabulary growth curve (42 → 148 → 167 → 306 tokens across training iterations) serves as a health indicator: vocabulary convergence signals that the behavioral space has been adequately sampled, triggering retraining when new tokens appear.

2. **Kernel-level blindness.** We assume the eBPF telemetry collection is not subverted. A kernel rootkit that blinds the collection layer is out of scope. Defending against this requires hardware-rooted attestation, which is orthogonal to our contribution.

3. **Shell-built-in commands.** Events from shell built-in commands (e.g., `exec`, `source`) that do not trigger `sched_process_exec` are invisible to the current telemetry set. Extending to `security_openat` / file-write events would close this gap.

4. **Low-rate attacks.** Extremely slow attacks (one action per hour) may be diluted below the EWMA detection threshold. The temporal CV analyzer partially addresses this (cadence is measured regardless of rate), but very sparse activity produces insufficient samples for reliable CV estimation.

5. **Perfect mimics.** An attacker that perfectly matches the benign distribution across all eight dimensions — including temporal cadence — would be undetectable by the grammar prior. However, the stealth-functionality trade-off makes this impossible for goal-directed behavior, as argued in Section 7.1.

6. **Coverage.** The current prototype library covers 14 ATT&CK techniques out of hundreds in the Linux matrix. Full coverage requires systematic collection (the Atomic Red Team pipeline, targeting ~600 techniques, is prepared but pending infrastructure).

7. **Attribution precision.** While technique-level recall is 14/14, exact attribution precision is approximately 60%. Automated remediation is gated on this metric, currently limiting full automation to high-severity adaptive patterns.

### 7.7 Ethics and Open-Source Strategy

This work is defense-oriented. The detection engine, range infrastructure, and synthetic test cases are designed to improve defensive visibility into host behavior. All experiments were conducted in libvirt-isolated networks (no outbound routing, iptables full-drop fallback) on differential clone VMs.

We adopt an asymmetric disclosure strategy: **the detection side is fully open-sourced**, including the model architecture, tokenization scheme, training protocol, prototype learning pipeline, and deployment tooling. The threat model (AAA) is described at the abstraction level only — specific implementation details of adaptive evasion capabilities are deliberately omitted. This reflects the "规则 vs 基线" principle applied to responsible disclosure: publishing the detector strengthens the baseline (which is expensive to circumvent), while withholding implementation details denies the attacker free rule-level intelligence.

The dual-use risk is assessed as follows: the detection architecture's publication enables defenders to deploy behavior-grammar detection at scale. Circumventing this detection requires the attacker to produce behavior indistinguishable from benign across eight dimensions including temporally constrained cadence — a requirement that compresses the attacker's action space far more than it aids offensive capability development.

---

## 8. Conclusion

We have presented *Behavioral Grammar*, a detection architecture that treats host runtime behavior as a structured language and learns its grammar with a 0.88M-parameter causal Transformer. The system achieves a 93% detection rate against an Adaptive Adversarial Agent at a 3.84% false-positive rate, with the temporal cadence dimension (CV = 0.310 vs. benign 9.786) providing the strongest discriminative signal.

Our findings support three conclusions:

**First, representation determines the detection ceiling.** The 8-token field-level discretization, not the model capacity, is the binding constraint. Adding a single slot (path category) made three previously inseparable persistence techniques linearly separable. Model size should be dictated by grammar complexity, not benchmark-driven parameter escalation — and the behavioral grammar of a single host is extremely narrow (val_ppl = 1.6).

**Second, adaptive behavior has structural limits.** The stealth-functionality trade-off, mathematically expressed through the coefficient of variation of inter-event intervals, creates a detection surface that cannot be closed without sacrificing operational capability. The AAA's CV of 0.310 is not an implementation artifact but a structural property of goal-directed adaptive behavior.

**Third, the economics of coevolution favor the defender.** Rule circumvention is free; baseline circumvention is expensive. The defender trains on massive offline data; the attacker evaluates through live probing. Each evasion technique opens a new observation surface. These structural asymmetries do not narrow with algorithmic progress — they reflect the fundamental difference between offline batch learning and online sequential adaptation.

The research implication is that detection efforts should focus not on larger models but on better representations, more independent observation surfaces, and faster baseline update cycles. The behavioral grammar framework — compact, auditable, deployable on edge devices — demonstrates that this focus is both practically viable and theoretically grounded.

---

## References

[1] S. Hofmeyr, S. Forrest, and A. Somayaji, "Intrusion detection using sequences of system calls," *Journal of Computer Security*, vol. 6, no. 3, pp. 151–180, 1998.

[2] W. Lee and S. J. Stolfo, "Data mining approaches for intrusion detection," in *Proc. 7th USENIX Security Symposium*, 1998, pp. 79–93.

[3] D. Schrammel et al., "From eBPF to system call-based intrusion detection," in *Proc. ACM Workshop on Security and Privacy Analytics*, 2021.

[4] M. Du, F. Li, G. Zheng, and V. Srikumar, "DeepLog: Anomaly detection and diagnosis from system logs through deep learning," in *Proc. ACM CCS*, 2017, pp. 1285–1298.

[5] W. Gao, J. Li, R. Wang, and H. Xiao, "MERLIN: A reinforcement learning-based open framework for autonomous malware evasion," in *Proc. IEEE Trustcom*, 2021.

[6] E. Raff et al., "An investigation of the limitations of an ML-based clone detector for malware," in *Proc. AAAI Workshop on Artificial Intelligence for Cyber Security*, 2019.

[7] R. Cohen, O. Lyzhov, and D. Raz, "Detecting anomalies in system logs using NLP and deep learning," in *Proc. IEEE INFOCOM Workshops*, 2020.

[8] J. Zhu et al., "Tools and metrics for processing log-based anomaly detection," in *Proc. IEEE/IFIP Network Operations and Management Symposium*, 2022.

[9] S. Forrester, H. Alipour, and A. Gurtov, "Audit-based host intrusion detection system," in *Proc. IEEE Trustcom*, 2019.

[10] D. Schrammel et al., "Tracee: eBPF-based runtime security observability," *Aqua Security Open Source Project*, 2021. [Online]. Available: https://github.com/aquasecurity/tracee

[11] Tracee Contributors, "Tracee: Linux tracing and security observability using eBPF," 2023.

[12] Falco Authors, "Falco: Cloud-native runtime security," CNCF Graduated Project, 2023. [Online]. Available: https://falco.org

[13] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," in *Proc. NeurIPS*, 2017, pp. 6306–6315.

[14] E. Raff et al., "Malware detection by eating a whole exe," in *Proc. AAAI Workshop on Artificial Intelligence for Cyber Security*, 2018.

[15] Z. Fang et al., "MAB-Malware: Reinforcement learning for minimizing malware detection," in *Proc. IEEE Trustcom*, 2021.

[16] L. Nataraj et al., "SARV: Malware analysis and classification using image processing and machine learning," 2019.

[17] F. Zaffarano et al., "A machine learning approach to malware similarity analysis using deep learning representations," in *Proc. DIMVA*, 2019.

[18] B. E. Strom et al., "MITRE ATT&CK: Design and philosophy," *MITRE Corporation*, Tech. Rep. MP180360R1, 2020.

[19] T. Alpcan and T. Başar, *Network Security: A Decision and Game-Theoretic Approach*. Cambridge University Press, 2010.

[20] A. Fielder, E. Panaousis, P. Malacaria, C. Hankin, and F. Smeraldi, "Decision support approaches for cyber security investment," *Decision Support Systems*, vol. 86, pp. 13–23, 2016.

[21] R. Anderson, "Why information security is hard: An economic perspective," in *Proc. ACSAC*, 2001.

[22] R. M. Anderson and R. M. May, *Infectious Diseases of Humans: Dynamics and Control*. Oxford University Press, 1991.

[23] Mandiant, "M-Trends 2023: Global threat landscape report," *Mandiant Research*, 2023.

[24] A. Radford, J. Wu, et al., "Language models are unsupervised multitask learners," *OpenAI Tech Report*, 2019.

[25] A. Vaswani et al., "Attention is all you need," in *Proc. NeurIPS*, 2017, pp. 5998–6008.

[26] K. Chopra, R. Hadsell, and Y. LeCun, "Learning a similarity metric discriminatively, with application to face verification," in *Proc. CVPR*, 2005.

[27] MITRE Corporation, "Atomic Red Team," 2023. [Online]. Available: https://github.com/redcanaryco/atomic-red-team

[28] M. Castro and B. Liskov, "Practical Byzantine fault tolerance," in *Proc. OSDI*, 1999.

[29] S. Forrest, A. Somayaji, and D. Ackley, "Building diverse computer systems," in *Proc. IEEE HOTOS*, 1997.

[30] C. Warrender, S. Forrest, and B. Pearlmutter, "Detecting intrusions using system calls: Alternative data models," in *Proc. IEEE Symposium on Security and Privacy*, 1999.

---

*Manuscript version 1.0. Prepared for arXiv submission (cs.CR). All experimental data collected from isolated lab environments. No live systems or third-party data were involved.*
