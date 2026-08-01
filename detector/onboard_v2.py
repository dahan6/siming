#!/usr/bin/env python3
"""一键自适应上线脚本（onboard）

新机器部署流程：
1. 采集 20 分钟良性遥测 → data/onboard_benign.jsonl
2. 在通用 prior 上标定本机 slot_tau → slot_tau_local.json
3. 验证 FPR < 5%
4. 激活 deploy_scorer

用法: onboard.py <model_dir> [--collect-min 20] [--max-fpr 0.05]
"""
import json
import os
import sys
import subprocess
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

from train_prior import TinyGPT, CTX

DET = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.3


def slot_of(tok):
    if ":" in tok:
        return tok.split(":")[0]
    if tok.startswith("ARGV"):
        return "ARGV"
    if tok.startswith("DT"):
        return "DT"
    return tok


def calibrate(model_dir, benign_path, n_max=20000):
    """在良性数据上标定 per-slot τ"""
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"),
                      map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    slot_nlls = defaultdict(list)
    window = []
    n = 0

    for line in open(benign_path):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if n >= n_max:
            break

        tokens = e.get("tokens", [])
        if len(tokens) < 4:
            continue

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
            s = slot_of(t)
            slot_nlls[s].append(v)
        n += 1

    # τ = p95 of benign NLL，但加最小阈值保底
    MIN_TAU = {"ARGV": 0.5, "DST": 0.3, "DT": 2.0, "ET": 0.5,
               "PARENT": 0.3, "PC": 0.3, "PROC": 1.0, "UID": 0.3}
    tau = {}
    for slot, vals in slot_nlls.items():
        arr = np.array(vals)
        if len(arr) >= 10:
            p95 = float(np.percentile(arr, 95))
            min_t = MIN_TAU.get(slot, 0.5)
            tau[slot] = round(max(p95, min_t), 3)

    # 验证 FPR
    n_alert = 0
    n_total = 0
    window = []
    for line in open(benign_path):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue

        tokens = e.get("tokens", [])
        if len(tokens) < 4:
            continue

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
            s = slot_of(t)
            if v > tau.get(s, 1.0):
                n_alert += 1
                break
        n_total += 1

    fpr = n_alert / max(n_total, 1)
    return tau, fpr, n


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DET, "model-vm-universal")

    # 如果 model_dir 里没有 prior.pt，尝试 model-current
    if not os.path.exists(os.path.join(model_dir, "prior.pt")):
        model_dir = os.path.join(DET, "model-current")
    if not os.path.exists(os.path.join(model_dir, "prior.pt")):
        print("ERROR: 找不到 prior.pt，请指定 model_dir")
        sys.exit(1)

    benign_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DET, "data", "onboard_benign.jsonl")
    max_fpr = 0.05

    print(f"模型: {model_dir}")
    print(f"良性数据: {benign_path}")

    if not os.path.exists(benign_path):
        print(f"\n良性数据不存在。请先采集：")
        print(f"  tracee 或 falco 采集 ≥20 分钟正常活动 → {benign_path}")
        print(f"  或用 VM 遥测：")
        print(f"  python3 -c \"...\" > {benign_path}")
        sys.exit(1)

    print(f"\n=== 标定 slot_tau ===")
    tau, fpr, n = calibrate(model_dir, benign_path)

    print(f"样本: {n}")
    print(f"FPR: {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"\n{'Slot':<10}{'τ':>8}")
    print("-" * 18)
    for s in sorted(tau.keys()):
        print(f"{s:<10}{tau[s]:>8.3f}")

    if fpr > max_fpr:
        print(f"\n⚠️  FPR {fpr:.2%} > {max_fpr:.0%} 阈值")
        print("建议：增加采集时间，或手动调高 τ")
        # 自动调高：用 p99 代替 p95
        print("\n自动修复：尝试 p99 阈值...")
        tau2, fpr2, _ = calibrate_p99(model_dir, benign_path)
        if fpr2 <= max_fpr:
            print(f"p99 标定成功: FPR={fpr2:.2%}")
            tau = tau2
            fpr = fpr2
        else:
            print(f"p99 仍然 {fpr2:.2%} > {max_fpr:.0%}")
    else:
        print(f"\n✅ FPR {fpr:.2%} ≤ {max_fpr:.0%} — 可用")

    # 保存
    tau_path = os.path.join(model_dir, "slot_tau_local.json")
    with open(tau_path, "w") as f:
        json.dump({"slot_tau": tau, "fpr": fpr, "n_calib": n,
                   "source": os.path.basename(benign_path),
                   "calibrated_at": time.strftime("%F %T")}, f, indent=2)
    print(f"\nτ 已保存 → {tau_path}")
    print(f"\n下一步: deploy_scorer.py {model_dir} --state {model_dir}/scorer_state.json")


def calibrate_p99(model_dir, benign_path, n_max=20000):
    """p99 标定（更保守）"""
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"),
                      map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    slot_nlls = defaultdict(list)
    window = []
    n = 0

    for line in open(benign_path):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if n >= n_max:
            break
        tokens = e.get("tokens", [])
        if len(tokens) < 4:
            continue
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
            s = slot_of(t)
            slot_nlls[s].append(v)
        n += 1

    tau = {}
    MIN_TAU = {"ARGV": 0.8, "DST": 0.5, "DT": 3.0, "ET": 0.8,
               "PARENT": 0.5, "PC": 0.5, "PROC": 1.5, "UID": 0.5}
    for slot, vals in slot_nlls.items():
        arr = np.array(vals)
        if len(arr) >= 10:
            p99 = float(np.percentile(arr, 99))
            min_t = MIN_TAU.get(slot, 0.8)
            tau[slot] = round(max(p99, min_t), 3)

    # FPR
    n_alert = 0
    n_total = 0
    window = []
    for line in open(benign_path):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except:
            continue
        tokens = e.get("tokens", [])
        if len(tokens) < 4:
            continue
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
            s = slot_of(t)
            if v > tau.get(s, 1.0):
                n_alert += 1
                break
        n_total += 1
    return tau, n_alert / max(n_total, 1), n


if __name__ == "__main__":
    main()
