#!/usr/bin/env python3
"""Detect anomaly patterns in a syscall trace.

Usage:
    python3 detector/detect_in_trace.py <trace_file> [--vocab VOCAB] [--ngram N]
"""
import json
import sys
from pathlib import Path
from collections import Counter

DEFAULT_VOCAB = "models/vm-universal/vocab.json"
DEFAULT_NGRAM = 4

# Minimum count for a pattern to be considered anomalous
MIN_ANOMALY_COUNT = 3
# Maximum frequency ratio (pattern_count / total_ngrams) to be suspicious
MAX_NORMAL_RATIO = 0.30


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ngrams(seq, n):
    return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    trace_file = sys.argv[1]
    vocab_path = DEFAULT_VOCAB
    n = DEFAULT_NGRAM

    if "--vocab" in sys.argv:
        vocab_path = sys.argv[sys.argv.index("--vocab") + 1]
    if "--ngram" in sys.argv:
        n = int(sys.argv[sys.argv.index("--ngram") + 1])

    vocab = set(load_vocab(vocab_path))

    # Read trace
    trace = []
    with open(trace_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "syscall" in rec:
                    trace.append(rec["syscall"])
                elif "name" in rec:
                    trace.append(rec["name"])
            except json.JSONDecodeError:
                # Plain text: one syscall per line
                trace.append(line)

    if not trace:
        print(f"Empty trace: {trace_file}")
        sys.exit(0)

    # Compute n-gram statistics
    grams = ngrams(trace, n)
    counts = Counter(grams)
    total = len(grams)

    anomalies = []
    for gram, count in counts.most_common():
        ratio = count / total
        in_vocab = any(
            " ".join(gram[i:i+n]) in vocab for i in range(len(gram) - n + 1)
        ) if len(gram) > n else " ".join(gram) in vocab

        if count >= MIN_ANOMALY_COUNT and ratio > MAX_NORMAL_RATIO:
            anomalies.append({
                "pattern": list(gram),
                "count": count,
                "ratio": round(ratio, 4),
                "in_vocab": in_vocab,
            })

    result = {
        "trace_file": trace_file,
        "trace_length": len(trace),
        "ngram_order": n,
        "total_ngrams": total,
        "unique_ngrams": len(counts),
        "anomalies": anomalies,
        "status": "ANOMALOUS" if anomalies else "CLEAN",
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
