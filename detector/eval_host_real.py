#!/usr/bin/env python3
"""在宿主机真实数据上评估 FPR 和攻击检测率

用法:
  eval_host_real.py <model_dir> <benign_jsonl> <attack_jsonl>
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

from train_prior import TinyGPT, CTX

DET = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def slot_of(tok):
    if ":" in tok:
        return tok.split(":")[0]
    if tok.startswith("ARGV"):
        return "ARGV"
    if tok.startswith("DT"):
        return "DT"
    return tok


def convert_7to8(tokens_7):
    out = list(tokens_7)
    has_pc = any(t.startswith("PC:") for t in out)
    if not has_pc:
        dst_idx = next((i for i, t in enumerate(out) if t.startswith("DST:")), 5)
        out.insert(dst_idx + 1, "PC:NONE")
    return out


def load_events(path):
    events = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        tokens = e.get("tokens", e.get("event", []))
        tokens = convert_7to8(tokens)
        events.append(tokens)
    return events


def score_events(model, stoi, events):
    """逐事件打分：返回每事件的 (per_slot_nll dict, max_nll, mean_nll)。"""
    results = []
    window = []

    for tokens in events:
        ids = [stoi.get(t, 0) for t in tokens]
        window = (window + ids)[-CTX:]
        n_toks, L = len(ids), len(window)
        start = max(L - n_toks, 1)

        with torch.no_grad():
            x = torch.tensor(window, device=DEVICE).unsqueeze(0)
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(window[start:L], device=DEVICE)
            nll = -lp[start-1:L-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

        per_slot = defaultdict(list)
        for t, v in zip(tokens, nll.tolist()):
            per_slot[slot_of(t)].append(v)

        max_nll = max(nll.tolist()) if nll.tolist() else 0.0
        mean_nll = sum(nll.tolist()) / max(1, len(nll.tolist()))
        slot_max = {s: max(vs) for s, vs in per_slot.items()}

        results.append({
            "per_slot": slot_max,
            "max_nll": max_nll,
            "mean_nll": mean_nll,
        })

    return results


def eval_fpr(model, stoi, slot_tau, benign_events, min_tau=2.0):
    """用 per-slot τ 评估 FPR：事件任一 slot 超过 τ 即判为异常。"""
    results = score_events(model, stoi, benign_events)
    fp = 0
    for r in results:
        is_alert = any(r["per_slot"].get(s, 0) > max(tau, min_tau)
                       for s, tau in slot_tau.items())
        if is_alert:
            fp += 1
    fpr = fp / max(1, len(results)) * 100
    return fpr, results


def eval_detection(model, stoi, slot_tau, attack_events, min_tau=2.0):
    """评估攻击检测率。"""
    results = score_events(model, stoi, attack_events)
    tp = 0
    for r in results:
        is_alert = any(r["per_slot"].get(s, 0) > max(tau, min_tau)
                       for s, tau in slot_tau.items())
        if is_alert:
            tp += 1
    detection_rate = tp / max(1, len(results)) * 100
    return detection_rate, results


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        DET, "model-host-real-v2")
    benign_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        DET, "data/host_real_benign.jsonl")
    attack_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        DET, "data/host_real_attack.jsonl")

    # 加载模型
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"),
                      map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"模型: {model_dir} (词表 {len(stoi)})")

    # 加载 slot_tau
    tau_path = os.path.join(model_dir, "slot_tau_local.json")
    if os.path.exists(tau_path):
        slot_tau = json.load(open(tau_path))["slot_tau"]
        print(f"τ: {tau_path}")
    else:
        # 用 baseline_nll p99.5 作为默认 τ
        baseline = ckpt.get("baseline_nll", {})
        slot_tau = {
            "ET": baseline.get("p995", 5.0),
            "PROC": baseline.get("p995", 5.0),
            "ARGV": baseline.get("p995", 5.0),
            "PARENT": baseline.get("p995", 5.0),
            "UID": baseline.get("p995", 5.0),
            "DST": baseline.get("p995", 5.0),
            "PC": baseline.get("p995", 5.0),
            "DT": baseline.get("p995", 5.0),
        }
        print(f"τ: baseline p99.5 fallback")

    print(f"slot_tau: {json.dumps(slot_tau, indent=2)}\n")

    # 评估 FPR
    benign_events = load_events(benign_path)
    print(f"良性事件: {len(benign_events)}")
    fpr, benign_results = eval_fpr(model, stoi, slot_tau, benign_events)
    print(f"\n=== FPR: {fpr:.1f}% ({int(fpr*len(benign_events)/100)}/{len(benign_events)}) ===")

    # 打印良性事件 NLL 分布
    if benign_results:
        all_max = [r["max_nll"] for r in benign_results]
        print(f"  良性 max_NLL: mean={np.mean(all_max):.2f} "
              f"p50={np.percentile(all_max, 50):.2f} "
              f"p95={np.percentile(all_max, 95):.2f} "
              f"p99={np.percentile(all_max, 99):.2f}")

    # 评估攻击检测率
    if os.path.exists(attack_path):
        attack_events = load_events(attack_path)
        print(f"\n攻击事件: {len(attack_events)}")
        detection_rate, attack_results = eval_detection(
            model, stoi, slot_tau, attack_events)
        print(f"\n=== 检测率: {detection_rate:.1f}% "
              f"({int(detection_rate*len(attack_events)/100)}/{len(attack_events)}) ===")

        if attack_results:
            all_max = [r["max_nll"] for r in attack_results]
            print(f"  攻击 max_NLL: mean={np.mean(all_max):.2f} "
                  f"p50={np.percentile(all_max, 50):.2f} "
                  f"p95={np.percentile(all_max, 95):.2f}")

        # 逐事件展示前 10 个攻击事件的详情
        print(f"\n--- 前 10 条攻击事件详情 ---")
        with open(attack_path) as f:
            attack_raw = [json.loads(l) for l in f if l.strip()]
        for i, (ev, r) in enumerate(zip(attack_raw[:10], attack_results[:10])):
            flagged = any(r["per_slot"].get(s, 0) > max(tau, 2.0)
                          for s, tau in slot_tau.items())
            cmd = ev.get("attack_cmd", "?")
            print(f"  [{i}] {'⚠' if flagged else '✓'} "
                  f"max_nll={r['max_nll']:.2f} "
                  f"{' '.join(ev.get('tokens', [])[:3])} "
                  f"(cmd={cmd})")

    return fpr


if __name__ == "__main__":
    main()
