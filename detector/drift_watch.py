#!/usr/bin/env python3
"""Siming Drift Watch — periodic model freshness check.

Compares current model hashes against a baseline and alerts if drift detected.
Usage:
    python3 detector/drift_watch.py [--baseline BASELINE_JSON]
"""
import hashlib
import json
import os
import sys
from pathlib import Path
from datetime import datetime

MODELS_DIR = Path(os.environ.get("SIMING_MODELS_DIR", "models/vm-universal"))
BASELINE_PATH = Path(os.environ.get(
    "SIMING_DRIFT_BASELINE", "models/vm-universal/drift_baseline.json"))

WATCHED_FILES = [
    "vocab.json",
    "patterns.jsonl",
    "prototypes.jsonl",
    "schema.json",
]


def file_hash(path):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def main():
    current = {}
    for name in WATCHED_FILES:
        path = MODELS_DIR / name
        h = file_hash(path)
        if h:
            current[name] = h

    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        baseline_hashes = baseline.get("hashes", {})
    else:
        baseline_hashes = {}

    drifted = []
    for name, current_hash in current.items():
        old_hash = baseline_hashes.get(name)
        if old_hash and old_hash != current_hash:
            drifted.append({
                "file": name,
                "old_hash": old_hash,
                "new_hash": current_hash,
            })

    report = {
        "timestamp": datetime.now().isoformat(),
        "checked_files": len(current),
        "drifted_files": len(drifted),
        "drifted": drifted,
        "status": "DRIFT" if drifted else "OK",
    }

    # Update baseline if no drift or first run
    if not drifted or not baseline_hashes:
        baseline = {
            "timestamp": datetime.now().isoformat(),
            "hashes": current,
        }
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(
            json.dumps(baseline, indent=2, ensure_ascii=False),
            encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(1 if drifted else 0)


if __name__ == "__main__":
    main()
