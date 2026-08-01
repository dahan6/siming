#!/usr/bin/env python3
"""漂移监测守护：词表增长 / 告警率突变 / 基线分布漂移 → 触发重训信号
周期性检查最近窗口的 token 流统计，与基线快照对比，超阈则写触发文件。

用法: drift_watch.py <model_dir> <tokens.jsonl> [--window 20000] [--once]
触发文件: <model_dir>/RETRAIN_TRIGGER（含原因）
"""
import json
import os
import sys
import time
from collections import Counter


def window_stats(path, n):
    """最近 n 事件的统计快照：词表、槽位分布"""
    events = [json.loads(l) for l in open(path)][-n:]
    vocab = set()
    slot_tok = Counter()
    for ev in events:
        for t in ev["tokens"]:
            vocab.add(t)
            s = t.split(":")[0] if ":" in t else ("ARGV" if t.startswith("ARGV") else "DT")
            slot_tok[s] += 1
    return {"n": len(events), "vocab": len(vocab),
            "slot_dist": {s: c / max(1, len(events)) for s, c in slot_tok.items()}}


def check(model_dir, tokens_path, window):
    cur = window_stats(tokens_path, window)
    snap_path = os.path.join(model_dir, "drift_baseline.json")
    if not os.path.exists(snap_path):
        json.dump(cur, open(snap_path, "w"))
        return None, "基线快照已建"
    base = json.load(open(snap_path))
    reasons = []
    dv = cur["vocab"] - base["vocab"]
    if dv > base["vocab"] * 0.15:
        reasons.append(f"词表增长 {base['vocab']}→{cur['vocab']} (+{dv})")
    for s in ("DST", "UID", "PARENT"):
        d = abs(cur["slot_dist"].get(s, 0) - base["slot_dist"].get(s, 0))
        if d > 0.05:
            reasons.append(f"{s} 分布漂移 {base['slot_dist'].get(s,0):.3f}→{cur['slot_dist'].get(s,0):.3f}")
    return (reasons or None), f"vocab {base['vocab']}→{cur['vocab']}"


def main():
    model_dir, tokens_path = sys.argv[1], sys.argv[2]
    win = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 20000
    once = "--once" in sys.argv
    while True:
        reasons, info = check(model_dir, tokens_path, win)
        ts = time.strftime("%F %T")
        if reasons:
            trig = os.path.join(model_dir, "RETRAIN_TRIGGER")
            json.dump({"ts": ts, "reasons": reasons}, open(trig, "w"), ensure_ascii=False)
            print(f"[{ts}] !! 触发重训: {'; '.join(reasons)} -> {trig}")
        else:
            print(f"[{ts}] 正常 ({info})")
        if once:
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
