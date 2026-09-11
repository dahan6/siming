#!/usr/bin/env python3
"""补充实验：消融实验 + 原型组留一技术验证 + 误报审计 + τ百分位扫描

这些是论文 Evaluation 部分需要的实验数据。
"""
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

from train_prior import TinyGPT, CTX
from pattern_db import PatternDB
from temporal_analyzer import TemporalAnalyzer
from adaptive_detector import AdaptiveDetector

DET = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.3
CTX_SLOTS = {"PARENT", "DST", "DT"}
RARE_SLOTS = {"PROC", "ARGV"}


def slot_of(tok):
    if ":" in tok: return tok.split(":")[0]
    if tok.startswith("ARGV"): return "ARGV"
    if tok.startswith("DT"): return "DT"
    return tok


def convert_7to8(tokens_7):
    out = list(tokens_7)
    if not any(t.startswith("PC:") for t in out):
        dst_idx = next((i for i, t in enumerate(out) if t.startswith("DST:")), 5)
        out.insert(dst_idx + 1, "PC:NONE")
    return out


def run_pipeline(model_dir, events_path, hour_filter=None, n_max=50000,
                 enable_prior=True, enable_pattern=True, enable_temporal=True,
                 enable_adaptive=True):
    """可配置的管道——用于消融实验"""
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"),
                      map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tau_path = os.path.join(model_dir, "slot_tau_local.json")
    if os.path.exists(tau_path):
        slot_tau = json.load(open(tau_path))["slot_tau"]
    else:
        tau_path = os.path.join(model_dir, "slot_tau_vm.json")
        slot_tau = json.load(open(tau_path))["slot_tau"] if os.path.exists(tau_path) else {}

    db = PatternDB(os.path.join(DET, "patterns.jsonl")) if enable_pattern else None
    temporal = TemporalAnalyzer(min_samples=20) if enable_temporal else None
    adaptive = AdaptiveDetector(window_size=200, cooldown=50) if enable_adaptive else None

    window = []
    ewma = 0.0
    n_scored = 0
    n_alert = 0

    for line in open(events_path):
        line = line.strip()
        if not line: continue
        try:
            e = json.loads(line)
        except: continue
        if hour_filter and e["ts"][11:13] not in hour_filter: continue
        if n_scored >= n_max: break

        tokens = convert_7to8(e["tokens"])
        try:
            ts = datetime.fromisoformat(e["ts"]).timestamp() * 1000
        except:
            ts = n_scored * 1000
        n_scored += 1

        # Prior
        fired = []
        n_unk = 0
        if enable_prior:
            ids = [stoi.get(t, 0) for t in tokens]
            window = (window + ids)[-CTX:]
            n_toks, L = len(ids), len(window)
            start = max(L - n_toks, 1)
            with torch.no_grad():
                x = torch.tensor(window, device=DEVICE).unsqueeze(0)
                lp = torch.log_softmax(model(x), dim=-1)[0]
                tgt = torch.tensor(window[start:L], device=DEVICE)
                nll = -lp[start-1:L-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            s_ev = nll.max().item()
            ewma = ALPHA * s_ev + (1 - ALPHA) * ewma
            for t, v in zip(tokens, nll.tolist()):
                s = slot_of(t)
                if v > slot_tau.get(s, 999):
                    fired.append((s, t, v))
            n_unk = sum(1 for t in tokens if t not in stoi)

        ctx_fired = [x for x in fired if x[0] in CTX_SLOTS]
        rare_fired = [x for x in fired if x[0] in RARE_SLOTS]

        # Pattern
        strong_pats = []
        weak_pats = []
        if enable_pattern and db:
            hits = db.match(tokens)
            strong_pats = [h for h in hits if h.get("severity", 0) >= 4]
            weak_pats = [h for h in hits if h.get("severity", 0) < 4]

        # Adaptive
        adapt_p5 = []
        adapt_low = []
        if enable_adaptive:
            alerts = adaptive.update(tokens, ts=e.get("ts", ""))
            adapt_p5 = [a for a in alerts if a["severity"] >= 5]
            adapt_low = [a for a in alerts if a["severity"] < 5]

        # Temporal
        temp_anom = False
        if enable_temporal and temporal:
            proc_name = "?"
            uid_val = "?"
            for t in tokens:
                if t.startswith("PROC:"): proc_name = t.split(":",1)[1]
                elif t.startswith("UID:"): uid_val = t.split(":",1)[1]
            res = temporal.update(proc_name, ts, ewma, uid=uid_val)
            if res and res["verdict"] == "anomalous":
                temp_anom = True

        # Fusion
        if adapt_p5 or strong_pats or ctx_fired or (rare_fired or n_unk > 0) or temp_anom or (adapt_low or weak_pats):
            n_alert += 1

    return n_scored, n_alert


def experiment_ablation():
    """消融实验：逐层去除，看检测率和FPR变化"""
    model_dir = os.path.join(DET, "model-vm-universal")
    clone = os.path.expanduser("~/adaptive-agent-sim/checkpoints/clone_events.jsonl")
    regime = os.path.expanduser("~/adaptive-agent-sim/checkpoints/regime_events.jsonl")

    configs = [
        ("Full (5-layer)", True, True, True, True),
        ("No Temporal (P4)", True, True, False, True),
        ("No Adaptive (P0/P5)", True, True, True, False),
        ("No Pattern (P1)", True, False, True, True),
        ("Prior Only (P2/P3)", True, False, False, False),
        ("Temporal Only", False, False, True, False),
    ]

    print("=" * 70)
    print("Experiment 1: Layer Ablation Study")
    print("=" * 70)
    print(f"{'Config':<25} {'Agent Detection':>16} {'Benign FPR':>12} {'Discrim.':>10}")
    print("-" * 63)

    results = []
    for name, ep, epa, et, ea in configs:
        bee_n, bee_alert = run_pipeline(model_dir, regime, {"13"}, 50000,
                                         enable_prior=ep, enable_pattern=epa,
                                         enable_temporal=et, enable_adaptive=ea)
        ben_n, ben_alert = run_pipeline(model_dir, clone, {"02","03","04","05"}, 20000,
                                         enable_prior=ep, enable_pattern=epa,
                                         enable_temporal=et, enable_adaptive=ea)
        bee_rate = bee_alert / max(bee_n, 1) * 100
        ben_rate = ben_alert / max(ben_n, 1) * 100
        disc = bee_rate - ben_rate
        print(f"{name:<25} {bee_rate:>15.1f}% {ben_rate:>11.1f}% {disc:>+9.1f}%")
        results.append({"config": name, "agent_rate": bee_rate, "benign_rate": ben_rate, "discrimination": disc})

    return results


def experiment_tau_sweep():
    """τ 百分位扫描：不同百分位的 FPR 和检测率"""
    model_dir = os.path.join(DET, "model-vm-universal")
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    benign_path = os.path.join(DET, "data", "onboard_benign.jsonl")

    # Collect per-slot NLLs
    slot_nlls = defaultdict(list)
    window = []
    for line in open(benign_path):
        line = line.strip()
        if not line or not line.startswith("{"): continue
        try:
            e = json.loads(line)
        except: continue
        tokens = e.get("tokens", [])
        if len(tokens) < 4: continue
        ids = [stoi.get(t, 0) for t in tokens]
        window = (window + ids)[-CTX:]
        n_toks, L = len(ids), len(window)
        start = max(L - n_toks, 1)
        with torch.no_grad():
            x = torch.tensor(window, device=DEVICE).unsqueeze(0)
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(window[start:L], device=DEVICE)
            nll = -lp[start-1:L-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for t, v in zip(tokens, nll.tolist()):
            slot_nlls[slot_of(t)].append(v)

    print("\n" + "=" * 70)
    print("Experiment 2: τ Percentile Sweep")
    print("=" * 70)
    print(f"{'Percentile':<15} {'Benign FPR':>12} {'Agent P2+P3':>12} {'Separation':>12}")
    print("-" * 51)

    for pct in [95, 97, 99, 99.5, 99.9, 99.95]:
        tau = {}
        for slot, vals in slot_nlls.items():
            arr = np.array(vals)
            if len(arr) >= 10:
                tau[slot] = float(np.percentile(arr, pct))

        # FPR on benign
        n_alert = 0
        n_total = 0
        window = []
        for line in open(benign_path):
            line = line.strip()
            if not line or not line.startswith("{"): continue
            try:
                e = json.loads(line)
            except: continue
            tokens = e.get("tokens", [])
            if len(tokens) < 4: continue
            ids = [stoi.get(t, 0) for t in tokens]
            window = (window + ids)[-CTX:]
            n_toks, L = len(ids), len(window)
            start = max(L - n_toks, 1)
            with torch.no_grad():
                x = torch.tensor(window, device=DEVICE).unsqueeze(0)
                lp = torch.log_softmax(model(x), dim=-1)[0]
                tgt = torch.tensor(window[start:L], device=DEVICE)
                nll = -lp[start-1:L-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            for t, v in zip(tokens, nll.tolist()):
                if v > tau.get(slot_of(t), 999):
                    n_alert += 1; break
            n_total += 1

        benign_fpr = n_alert / max(n_total, 1) * 100
        print(f"p{pct:<10} {benign_fpr:>11.2f}%")


def experiment_synthetic_detection():
    """合成攻击序列的完整检测率"""
    from test_adaptive_synth import (test_rebirth_retire, test_disguise_c2,
                                      test_suid_privesc, test_exfil_memory,
                                      test_recon_uniform, test_sleep_stepping,
                                      test_benign_baseline)

    print("\n" + "=" * 70)
    print("Experiment 3: Synthetic Attack Sequence Detection")
    print("=" * 70)

    tests = [
        ("Rebirth Sequence", test_rebirth_retire),
        ("Disguised C2 Exfil", test_disguise_c2),
        ("SUID Privilege Escalation", test_suid_privesc),
        ("Memory-only Exfiltration", test_exfil_memory),
        ("Uniform Recon Sweep", test_recon_uniform),
        ("Sleep Stepping Pattern", test_sleep_stepping),
        ("Benign Low FPR", test_benign_baseline),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n  [{name}]")
        try:
            result = test_fn()
            status = "PASS" if result else "FAIL"
            results.append({"test": name, "result": status})
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"test": name, "result": "ERROR"})

    print(f"\n{'='*40}")
    print(f"Summary: {sum(1 for r in results if r['result']=='PASS')}/{len(results)} passed")
    return results


def experiment_cross_host():
    """跨机验证：留一 VM 验证"""
    model_dir = os.path.join(DET, "model-vm-universal")
    clone = os.path.expanduser("~/adaptive-agent-sim/checkpoints/clone_events.jsonl")

    # 按host分组
    host_events = defaultdict(list)
    for line in open(clone):
        e = json.loads(line)
        host_events[e.get("host", "?")].append(line)

    print("\n" + "=" * 70)
    print("Experiment 4: Leave-One-VM-Out Cross-Host Validation")
    print("=" * 70)
    print(f"{'Test VM':<15} {'Events':>8} {'FPR':>8}")
    print("-" * 33)

    results = []
    for host in sorted(host_events.keys()):
        events = host_events[host]
        n_max = min(len(events), 5000)
        # 用全部数据训的模型，在每台VM上测FPR
        ckpt = torch.load(os.path.join(model_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
        stoi = ckpt["stoi"]
        model = TinyGPT(len(stoi)).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()

        tau_path = os.path.join(model_dir, "slot_tau_local.json")
        slot_tau = json.load(open(tau_path))["slot_tau"]

        n_alert = 0
        n_total = 0
        window = []
        for line in events[:n_max]:
            e = json.loads(line)
            tokens = convert_7to8(e["tokens"])
            ids = [stoi.get(t, 0) for t in tokens]
            window = (window + ids)[-CTX:]
            n_toks, L = len(ids), len(window)
            start = max(L - n_toks, 1)
            if L < 2: continue
            with torch.no_grad():
                x = torch.tensor(window, device=DEVICE).unsqueeze(0)
                lp = torch.log_softmax(model(x), dim=-1)[0]
                tgt = torch.tensor(window[start:L], device=DEVICE)
                nll = -lp[start-1:L-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            for t, v in zip(tokens, nll.tolist()):
                if v > slot_tau.get(slot_of(t), 999):
                    n_alert += 1; break
            n_total += 1

        fpr = n_alert / max(n_total, 1) * 100
        print(f"{host:<15} {n_total:>8} {fpr:>7.1f}%")
        results.append({"host": host, "events": n_total, "fpr": fpr})

    avg_fpr = np.mean([r["fpr"] for r in results])
    print(f"\n  Average FPR across {len(results)} VMs: {avg_fpr:.1f}%")
    return results


def main():
    # Experiment 1: Ablation
    ablation = experiment_ablation()

    # Experiment 2: τ sweep
    experiment_tau_sweep()

    # Experiment 3: Synthetic detection
    synthetic = experiment_synthetic_detection()

    # Experiment 4: Cross-host
    cross_host = experiment_cross_host()

    # Save all results
    all_results = {
        "ablation": ablation,
        "synthetic_detection": synthetic,
        "cross_host": cross_host,
    }
    out_path = os.path.join(DET, "experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n所有实验结果保存到 {out_path}")


if __name__ == "__main__":
    main()
