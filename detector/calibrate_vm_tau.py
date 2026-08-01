#!/usr/bin/env python3
"""跨机基线标定：在 VM 遥测（良性夜间）上重新标定 slot_tau

解决 §16 实证的 92.5% 跨机误报：
宿主机 slot_tau 不适用 VM，需在 VM 数据上重标定。
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

from train_prior import TinyGPT, CTX


def slot_of(tok):
    if ":" in tok:
        return tok.split(":")[0]
    if tok.startswith("ARGV"):
        return "ARGV"
    if tok.startswith("DT"):
        return "DT"
    return tok

DET = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def convert_7to8(tokens_7):
    out = list(tokens_7)
    has_pc = any(t.startswith("PC:") for t in out)
    if not has_pc:
        dst_idx = next((i for i, t in enumerate(out) if t.startswith("DST:")), 5)
        out.insert(dst_idx + 1, "PC:NONE")
    return out


def calibrate_vm(model_dir, benign_jsonl, hour_filter=None, n_max=20000):
    """在 VM 良性数据上标定 per-slot τ"""
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"),
                      map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    slot_nlls = defaultdict(list)
    window = []
    n = 0

    for line in open(benign_jsonl):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if hour_filter and e["ts"][11:13] not in hour_filter:
            continue
        if n >= n_max:
            break

        tokens = convert_7to8(e["tokens"])
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

    # 标定 τ：每 slot 取良性分布的 p95
    new_tau = {}
    print(f"标定样本: {n} 事件")
    print(f"\n{'Slot':<10}{'n':>8}{'mean':>8}{'p50':>8}{'p90':>8}{'p95':>8}{'p99':>8}{'τ':>8}")
    for slot in sorted(slot_nlls.keys()):
        vals = np.array(slot_nlls[slot])
        p50 = np.percentile(vals, 50)
        p90 = np.percentile(vals, 90)
        p95 = np.percentile(vals, 95)
        p99 = np.percentile(vals, 99)
        tau = float(p95)
        new_tau[slot] = round(tau, 3)
        print(f"{slot:<10}{len(vals):>8}{np.mean(vals):>8.2f}{p50:>8.2f}"
              f"{p90:>8.2f}{p95:>8.2f}{p99:>8.2f}{tau:>8.2f}")

    return new_tau


def main():
    # 优先用 VM 通用模型标定
    model_dir = os.path.join(DET, "model-vm-universal")
    if not os.path.exists(os.path.join(model_dir, "prior.pt")):
        model_dir = os.path.join(DET, "model-current")

    clone_path = os.path.expanduser(
        "~/data/telemetry/clone_events.jsonl")

    print(f"=== 在 VM 良性夜间数据上标定 slot_tau (模型: {os.path.basename(model_dir)}) ===\n")
    vm_tau = calibrate_vm(model_dir, clone_path,
                          hour_filter={"02", "03", "04", "05"}, n_max=20000)

    # 与通用模型 τ 对比（如果有旧 τ）
    old_tau_path = os.path.join(DET, "model-current", "slot_tau.json")
    if os.path.exists(old_tau_path):
        old_tau = json.load(open(old_tau_path))["slot_tau"]
        print(f"\n=== 宿主机旧 τ vs VM 通用 τ ===")
        print(f"{'Slot':<10}{'Host τ':>10}{'VM τ':>10}{'Δ':>10}")
        print("-" * 40)
        for slot in sorted(set(list(old_tau.keys()) + list(vm_tau.keys()))):
            h = old_tau.get(slot, 0)
            v = vm_tau.get(slot, 0)
            print(f"{slot:<10}{h:>10.3f}{v:>10.3f}{v-h:>+10.3f}")
    else:
        print("（无旧 τ 可对比）")

    # 保存 VM τ
    vm_tau_path = os.path.join(model_dir, "slot_tau_vm.json")
    with open(vm_tau_path, "w") as f:
        json.dump({"slot_tau": vm_tau, "source": "vm_universal_model",
                   "n_calib": 20000}, f, indent=2)
    print(f"\nVM τ 已保存 → {vm_tau_path}")

    return vm_tau


if __name__ == "__main__":
    main()
