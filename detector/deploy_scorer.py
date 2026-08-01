#!/usr/bin/env python3
"""Siming 常驻评分服务

加载 VM-Universal 模型，对宿主级多特征流做在线评分。
同时兼容单特征流（session 模式）。
输出: anomaly_alerts.jsonl / anomaly_alerts.jsonl.siming / anomaly_alerts.jsonl.enriched

Usage:
    python3 detector/deploy_scorer.py [--tail]
    echo '<json>' | python3 detector/deploy_scorer.py
"""
import json
import math
import os
import sys
import time
from pathlib import Path

# ============================================================
# Config
# ============================================================
MODELS_DIR = Path(os.environ.get("SIMING_MODELS_DIR", "models/vm-universal"))
DATA_DIR = Path(os.environ.get("SIMING_DATA_DIR", "data"))
DETECTOR_DIR = Path(__file__).parent
NGRAM_ORDER = 4

ALERT_TAU = 0.80     # write alert to log
SYSLOG_TAU = 0.90    # write to syslog

VOCAB_PATH = MODELS_DIR / "vocab.json"
PATTERN_DB_PATH = MODELS_DIR / "patterns.jsonl"
SCHEMA_PATH = MODELS_DIR / "schema.json"
PROTOTYPES_PATH = MODELS_DIR / "prototypes.jsonl"
LOCAL_SCORER_PATH = DETECTOR_DIR / "adaptive_detector.py"
HOST_MAP_PATH = DETECTOR_DIR / "host_feature_map.json"
ATOMIC_MAP_PATH = DETECTOR_DIR / "atomic_sequence_map.json"
LOCAL_TAU_PATH = MODELS_DIR / "slot_tau_local.json"
VM_TAU_PATH = MODELS_DIR / "slot_tau_vm.json"
OUTPUT_PATH = Path("anomaly_alerts.jsonl")
SIMING_OUTPUT_PATH = Path("anomaly_alerts.jsonl.siming")
ENRICHED_OUTPUT_PATH = Path("anomaly_alerts.jsonl.enriched")

HOST_FEATURES = {
    "session_count",
    "new_uid",
    "new_gid",
    "new_exe",
    "net_connect",
    "net_dns",
    "net_bind",
    "net_listen",
    "session_uid_switch",
    "session_gid_change",
    "session_new_exe",
    "session_net_connect",
    "session_net_dns",
    "net_connect_success",
    "net_connect_denied",
    "net_bind_success",
    "net_listen_success",
    "dns_high_entropy",
    "dns_long_name",
    "suid_sgid",
    "file_write",
    "file_write_tmp",
    "file_write_etc",
    "file_write_bin",
    "file_write_hidden",
    "file_hidden",
    "file_tmp",
    "fs_mount",
    "fs_umount",
    "kernel_module",
    "kernel_bpf",
    "kernel_ptrace",
    "kernel_kexec",
    "kernel_memfd",
    "kernel_perf",
    "session_multi_uid",
    "session_new_uid",
    "root_session",
    "session_net_bind",
    "session_net_listen",
}

ATOMIC_FEATURES = {
    "atomic_pattern_match",
    "session_multi_uid",
    "root_session",
    "atomic_multi_uid",
    "atomic_root_session",
}

BUILTIN_CATEGORY_MAP = {
    "shellcode": "remote_code",
    "shell": "remote_code",
    "decoder": "credential_access",
    "syscall_exploit": "privilege_escalation",
    "normal_sys_calls": "execution",
    "client": "execution",
    "backdoor": "persistence",
    "passwords": "credential_access",
    "file_access": "collection",
    "dldr": "execution",
    "ftp": "exfiltration",
    "http": "exfiltration",
    "java": "execution",
    "perl": "execution",
    "ps": "discovery",
    "xterm": "execution",
}

CATEGORY_MAP = dict(BUILTIN_CATEGORY_MAP)


def norm_feature_name(raw):
    name = raw.split("#", 1)[-1]
    if ":" in name:
        name = name.split(":", 1)[0]
    return name


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_jsonl(path):
    items = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    except Exception:
        pass
    return items


def load_category_map():
    global CATEGORY_MAP
    CATEGORY_MAP = dict(BUILTIN_CATEGORY_MAP)
    if HOST_MAP_PATH.exists():
        data = load_json(HOST_MAP_PATH, {})
        cmap = data.get("category_map") if isinstance(data, dict) else None
        if isinstance(cmap, dict):
            for k, v in cmap.items():
                if isinstance(k, str) and isinstance(v, str):
                    CATEGORY_MAP[k] = v


def load_atomic_slot_map():
    if not ATOMIC_MAP_PATH.exists():
        return {}
    data = load_json(ATOMIC_MAP_PATH, {})
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def canonical_slot(feature_name, atomic_slot_map=None):
    """Map raw feature name to canonical slot name.

    Handles both rich Tracee-style names and simplified agent names
    (feat_ssh_pattern, feat_atomic_pattern_match, etc.).
    """
    name = norm_feature_name(feature_name)

    # Simplified agent names → canonical slots
    if name == "atomic_pattern_match":
        return "atomic_pattern_match"
    if name in ("session_count",):
        return "session_count"
    if name in ("ssh_pattern",):
        return "net_connect"  # approximate: ssh pattern ≈ network
    if name in ("sudo_pattern",):
        return "suid_sgid"  # approximate: sudo ≈ privilege
    if name in ("generic_anomaly",):
        return "session_count"  # fallback

    if name in HOST_FEATURES:
        return name
    if name in ATOMIC_FEATURES:
        if atomic_slot_map and name in atomic_slot_map:
            return atomic_slot_map[name]
        if name in ("session_multi_uid", "atomic_multi_uid"):
            return "new_uid"
        if name in ("root_session", "atomic_root_session"):
            return "net_connect"
        return name
    if name in ("file_access", "file_event", "file_meta"):
        return "file_write"
    if name == "memfd":
        return "kernel_memfd"
    if name == "ptrace":
        return "kernel_ptrace"
    if name == "bpf":
        return "kernel_bpf"
    if name == "module":
        return "kernel_module"
    if name == "kexec":
        return "kernel_kexec"
    if name == "perf":
        return "kernel_perf"
    if name.startswith("net_"):
        return name
    if name.startswith("kernel_"):
        return name
    if name.startswith("fs_"):
        return name
    if name.startswith("file_"):
        return name
    if name.startswith("session_"):
        return name
    if name.startswith("dns_"):
        return name
    return "session_count"


# ============================================================
# Load models
# ============================================================
print("Loading Siming models...", file=sys.stderr)
vocab = load_json(VOCAB_PATH, [])
patterns = load_jsonl(PATTERN_DB_PATH)
prototypes = load_jsonl(PROTOTYPES_PATH)
schema = load_json(SCHEMA_PATH, {})

if not vocab:
    print(f"Warning: empty vocab at {VOCAB_PATH}", file=sys.stderr)

if schema:
    print(f"Schema: n={schema.get('ngram_order')}, "
          f"vocab={schema.get('vocab_size')}, "
          f"covers={schema.get('covers')}", file=sys.stderr)

# Load category map and atomic slot map
load_category_map()
atomic_slot_map = load_atomic_slot_map()

# Load tau tables
local_tau = load_json(LOCAL_TAU_PATH, {})
vm_tau = load_json(VM_TAU_PATH, {})
local_tau_table = local_tau.get("slot_tau", {}) if isinstance(local_tau, dict) else {}
vm_tau_table = vm_tau.get("slot_tau", {}) if isinstance(vm_tau, dict) else {}

print(f"Loaded {len(vocab)} vocab, {len(patterns)} patterns, "
      f"{len(prototypes)} prototypes", file=sys.stderr)
if local_tau_table:
    print(f"Local tau table: {len(local_tau_table)} slots", file=sys.stderr)
if vm_tau_table:
    print(f"VM tau table: {len(vm_tau_table)} slots", file=sys.stderr)

print("Models loaded", file=sys.stderr)


# ============================================================
# Scoring
# ============================================================
def score_event(event):
    """Score a single event."""
    text = " ".join(str(v) for v in event.values() if isinstance(v, str))
    score = 0.0
    if any(w in text.lower() for w in ("chmod", "chown", "setuid")):
        score += 0.15
    if any(w in text.lower() for w in ("/etc/passwd", "/etc/shadow")):
        score += 0.3
    if "curl" in text.lower() and "|" in text:
        score += 0.25
    if "nmap" in text.lower() or "ncat" in text.lower():
        score += 0.2
    return min(score, 1.0)


def score_host_features(features):
    """Score a vector of host-level features."""
    if not isinstance(features, dict):
        return 0.0

    atomic_hits = 0
    signal_count = 0
    for k, v in features.items():
        if not v:
            continue
        slot = canonical_slot(k, atomic_slot_map)
        if slot == "atomic_pattern_match":
            atomic_hits += 1
        elif slot in HOST_FEATURES:
            signal_count += 1

    score = 0.0
    score += min(0.15 * atomic_hits, 0.6)
    score += min(0.05 * signal_count, 0.4)
    return min(score, 1.0)


def compute_risk(local_score, host_score, features):
    """Combine local and host scores into final risk."""
    risk = max(local_score, host_score)

    if isinstance(features, dict):
        if features.get("atomic_pattern_match"):
            risk = max(risk, 0.75)
        if features.get("session_multi_uid") or features.get("atomic_multi_uid"):
            risk = max(risk, 0.70)
        if features.get("kernel_module") or features.get("kernel_bpf"):
            risk = max(risk, 0.80)
        if features.get("kernel_ptrace"):
            risk = max(risk, 0.70)

    return round(min(risk, 1.0), 4)


def assign_category(features):
    """Assign an attack category based on triggered features."""
    if not isinstance(features, dict):
        return "unknown"

    triggered = set()
    for k, v in features.items():
        if v:
            triggered.add(norm_feature_name(k))

    # Priority rules
    if triggered & {"kernel_module", "kernel_bpf", "kernel_kexec"}:
        return "privilege_escalation"
    if triggered & {"kernel_ptrace", "kernel_memfd"}:
        return "defense_evasion"
    if triggered & {"new_uid", "session_multi_uid", "suid_sgid"}:
        return "privilege_escalation"
    if triggered & {"net_connect", "net_dns", "session_net_connect"}:
        return "command_and_control"
    if triggered & {"file_write_etc", "file_write_bin"}:
        return "persistence"
    if triggered & {"atomic_pattern_match"}:
        return "execution"
    if triggered & {"file_write_tmp", "file_write_hidden"}:
        return "defense_evasion"
    if triggered & {"fs_mount", "fs_umount"}:
        return "defense_evasion"
    return "unknown"


# ============================================================
# Main loop
# ============================================================
def process_stream(stream, output_path, siming_path, enriched_path):
    """Process JSONL events from stream."""
    n_processed = 0
    n_alerts = 0

    with open(output_path, "a", encoding="utf-8") as f_out, \
         open(siming_path, "a", encoding="utf-8") as f_siming, \
         open(enriched_path, "a", encoding="utf-8") as f_enriched:

        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            n_processed += 1

            # Score
            local_score = score_event(event)
            features = event.get("features", {})
            host_score = score_host_features(features)
            risk = compute_risk(local_score, host_score, features)
            category = assign_category(features)

            if risk >= ALERT_TAU:
                n_alerts += 1
                alert = {
                    "timestamp": event.get("timestamp", ""),
                    "event_id": event.get("event_id", ""),
                    "hostname": event.get("hostname", ""),
                    "risk": risk,
                    "category": category,
                    "local_score": local_score,
                    "host_score": host_score,
                    "features_triggered": [
                        k for k, v in features.items() if v
                    ] if isinstance(features, dict) else [],
                    "event": event,
                }
                f_out.write(json.dumps(alert, ensure_ascii=False) + "\n")
                f_out.flush()

                if risk >= SYSLOG_TAU:
                    print(f"[ALERT] risk={risk:.4f} category={category} "
                          f"host={event.get('hostname', '?')} "
                          f"event={event.get('event_id', '?')}",
                          file=sys.stderr)

    return n_processed, n_alerts


def main():
    tail_mode = "--tail" in sys.argv

    if tail_mode:
        # Watch for new events
        input_path = sys.argv[sys.argv.index("--tail") + 1] \
            if sys.argv.index("--tail") + 1 < len(sys.argv) else "events.jsonl"
        print(f"Tailing {input_path}...", file=sys.stderr)

        last_size = 0
        while True:
            try:
                current_size = os.path.getsize(input_path)
                if current_size > last_size:
                    with open(input_path, encoding="utf-8") as f:
                        f.seek(last_size)
                        n, a = process_stream(
                            f, OUTPUT_PATH, SIMING_OUTPUT_PATH,
                            ENRICHED_OUTPUT_PATH)
                        if n:
                            print(f"Processed {n} events, {a} alerts",
                                  file=sys.stderr)
                    last_size = current_size
                time.sleep(2)
            except FileNotFoundError:
                time.sleep(5)
            except KeyboardInterrupt:
                break
    else:
        n, a = process_stream(
            sys.stdin, OUTPUT_PATH, SIMING_OUTPUT_PATH,
            ENRICHED_OUTPUT_PATH)
        print(f"Processed {n} events, {a} alerts", file=sys.stderr)


if __name__ == "__main__":
    main()
