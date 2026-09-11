#!/usr/bin/env python3
"""墨城防火墙: 网络流量漂移守护

周期性检查最近窗口的 token 流统计，与基线快照对比，超阈则写触发文件。
触发重训信号，配合 fw_daemon 的 shadow_release 做无缝升级。

检测维度:
  1. 词表增长 — 新协议/新端口/新网络出现（>15% 新 token）
  2. 槽位分布漂移 — DSTNET/PORTCLS/PROTO 分布变化
  3. 流量速率漂移 — 事件频率突变
  4. 网络拓扑变化 — 新 IP 段出现

用法: fw_drift.py <model_dir> <tokens.jsonl> [--window 20000] [--once]
触发文件: <model_dir>/RETRAIN_TRIGGER（含原因）
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fw_calibrate import slot_of


def window_stats(path, n):
    """最近 n 事件的统计快照"""
    events = [json.loads(l) for l in open(path)][-n:]
    vocab = set()
    slot_tok = Counter()
    raw_ips = set()
    raw_ports = Counter()
    ts_first = events[0].get("ts_epoch", 0) if events else 0
    ts_last = events[-1].get("ts_epoch", 0) if events else 0
    duration = max(1, ts_last - ts_first)

    for ev in events:
        for t in ev["tokens"]:
            vocab.add(t)
            s = slot_of(t)
            slot_tok[s] += 1
        # 提取原始 IP 和端口
        raw = ev.get("raw", {})
        if raw.get("dst_ip"):
            # 取 /24 网段
            parts = raw["dst_ip"].split(".")
            if len(parts) == 4:
                raw_ips.add(".".join(parts[:3]) + ".0/24")
        if raw.get("dst_port"):
            raw_ports[raw["dst_port"]] += 1

    n_ev = max(1, len(events))
    return {
        "n": n_ev,
        "vocab": len(vocab),
        "vocab_set": vocab,
        "slot_dist": {s: c / n_ev for s, c in slot_tok.items()},
        "rate": n_ev / duration,  # events per second
        "n_subnets": len(raw_ips),
        "top_ports": dict(raw_ports.most_common(10)),
    }


def check(model_dir, tokens_path, window):
    """检查漂移，返回 (原因列表 or None, 信息字符串)"""
    cur = window_stats(tokens_path, window)
    snap_path = os.path.join(model_dir, "drift_baseline.json")

    if not os.path.exists(snap_path):
        json.dump({k: v for k, v in cur.items() if k != "vocab_set"},
                  open(snap_path, "w"), ensure_ascii=False, indent=2)
        return None, f"基线快照已建 (vocab={cur['vocab']}, rate={cur['rate']:.1f}/s)"

    base = json.load(open(snap_path))
    reasons = []

    # 1. 词表增长
    dv = cur["vocab"] - base["vocab"]
    if dv > base["vocab"] * 0.15:
        reasons.append(f"词表增长 {base['vocab']}→{cur['vocab']} (+{dv})")

    # 2. 槽位分布漂移
    for s in ("DSTNET", "PORTCLS", "PROTO", "DIR"):
        d = abs(cur["slot_dist"].get(s, 0) - base["slot_dist"].get(s, 0))
        if d > 0.05:
            reasons.append(f"{s} 分布漂移 {base['slot_dist'].get(s,0):.3f}→{cur['slot_dist'].get(s,0):.3f}")

    # 3. 流量速率漂移（>2x 或 <0.5x）
    base_rate = base.get("rate", cur["rate"])
    if base_rate > 0:
        ratio = cur["rate"] / base_rate
        if ratio > 2.0:
            reasons.append(f"流量速率激增 {base_rate:.1f}→{cur['rate']:.1f}/s ({ratio:.1f}x)")
        elif ratio < 0.5:
            reasons.append(f"流量速率骤降 {base_rate:.1f}→{cur['rate']:.1f}/s ({ratio:.1f}x)")

    # 4. 子网数变化
    base_subnets = base.get("n_subnets", cur["n_subnets"])
    if cur["n_subnets"] > base_subnets * 1.5:
        reasons.append(f"新子网出现 {base_subnets}→{cur['n_subnets']}")

    return (reasons or None), \
        f"vocab {base['vocab']}→{cur['vocab']}, rate {base_rate:.1f}→{cur['rate']:.1f}/s"


def main():
    model_dir = sys.argv[1]
    tokens_path = sys.argv[2]
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
