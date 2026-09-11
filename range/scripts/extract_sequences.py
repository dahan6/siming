#!/usr/bin/env python3
"""样本提取：采集 staging → 候选 sequence 条目（待人工过审）
差分原理：每份采集的前 N 行是基线窗（无攻击动作），之后的窗口与基线比对，
不属于基线行为模式的事件序列即为攻击片段。

用法: extract_sequences.py <staging_dir> [out.jsonl]
"""
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.expanduser("~/defense-lab/detector"))
from parse_raw_tracee import event_to_tokens


def load_events(path):
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def main():
    staging = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(staging, "candidates.jsonl")
    n_out = 0
    with open(out_path, "w") as out:
        for meta_path in sorted(glob.glob(os.path.join(staging, "**", "*.meta.json"),
                                          recursive=True)):
            meta = json.load(open(meta_path))
            round_tag = os.path.basename(os.path.dirname(meta_path))
            if round_tag == os.path.basename(staging.rstrip("/")):
                round_tag = "r0"
            tid = meta["id"]
            raw_path = meta_path.replace(".meta.json", ".jsonl")
            if not os.path.exists(raw_path):
                continue
            events = list(load_events(raw_path))
            if not events:
                continue
            base_n = meta["baseline_lines"]
            base, window = events[:base_n], events[base_n:]

            # 基线事件签名（PROC+PARENT+ET 三元组）
            def sig(ev_tokens):
                d = {t.split(":")[0]: t for t in ev_tokens if ":" in t}
                return (d.get("ET"), d.get("PROC"), d.get("PARENT"))

            base_sigs = Counter()
            prev_ts = None
            for ev in base:
                ts = ev.get("timestamp", 0)
                delta = 0 if prev_ts is None else max(0, (ts - prev_ts) // 1_000_000)
                prev_ts = ts
                base_sigs[sig(event_to_tokens(ev, delta))] += 1

            prev_ts = None
            attack_seqs = []
            for ev in window:
                ts = ev.get("timestamp", 0)
                delta = 0 if prev_ts is None else max(0, (ts - prev_ts) // 1_000_000)
                prev_ts = ts
                toks = event_to_tokens(ev, delta)
                if base_sigs.get(sig(toks), 0) == 0:
                    attack_seqs.append(toks)

            # 过滤采集器自身的噪声（ssh/scp/sudo 来自执行器本身）
            noise = {"PROC:ssh", "PROC:scp", "PROC:sudo"}
            attack_seqs = [s for s in attack_seqs
                           if not any(t in noise for t in s)]

            if attack_seqs:
                entry = {
                    "id": f"S-{tid}-{round_tag}", "type": "sequence",
                    "technique": tid.rstrip("b"), "name": f"实采样本 {tid} {round_tag}",
                    "severity": 4,
                    "sequence": [t for s in attack_seqs for t in s],
                    "source": f"atomic:{tid}:{round_tag}", "n_events": len(attack_seqs),
                    "review": "pending",
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_out += 1
                print(f"{tid}: {len(events)} 事件 -> 攻击片段 {len(attack_seqs)} 条")
            else:
                print(f"{tid}: {len(events)} 事件 -> 未提取到攻击片段（需人工检查）")
    print(f"\n候选样本 {n_out} 条 -> {out_path}（review=pending，过审后改 approved 并入库）")


if __name__ == "__main__":
    main()
