#!/usr/bin/env python3
"""上下文异常用例测试：合法原语 + 异常上下文（父进程/时序链/组合）
验证检测器对"大概率能识别但没实测"档位的真实能力。
每个用例配阴性对照：同一原语在正常上下文中应低分。

用法: test_context_cases.py <model_dir> <tokens.jsonl>
"""
import json
import os
import sys

import numpy as np
import torch

from train_prior import CTX, DEVICE, TinyGPT

# 用例：名字 -> (token 列表, 说明)。链式用例为多个事件的 token 直接拼接。
CASES = {
    "F-awk读敏感文件_父python3": (
        ["ET:EXEC", "PROC:awk", "ARGV:N1P", "PARENT:python3", "UID:1000", "DST:NONE", "DT2"],
        "awk 读文件本身合法，但被 python3 拉起是脚本植入的典型指纹"),
    "F-ctl_awk_父bash": (
        ["ET:EXEC", "PROC:awk", "ARGV:N1P", "PARENT:bash", "UID:1000", "DST:NONE", "DT2"],
        "阴性对照：交互 shell 里的 awk"),
    "G-ss侦察_父node": (
        ["ET:EXEC", "PROC:ss", "ARGV:N2-", "PARENT:node", "UID:1000", "DST:NONE", "DT1"],
        "Web 服务进程发起网络侦察 =  webshell 指纹"),
    "G-ctl_ss_父bash": (
        ["ET:EXEC", "PROC:ss", "ARGV:N2-", "PARENT:bash", "UID:1000", "DST:NONE", "DT1"],
        "阴性对照：管理员手动 ss"),
    "H-crontab持久化": (
        ["ET:EXEC", "PROC:crontab", "ARGV:N1-", "PARENT:bash", "UID:1000", "DST:NONE", "DT3"],
        "写 crontab（-e/写入），持久化手法"),
    "J-systemctl启用服务_root": (
        ["ET:EXEC", "PROC:systemctl", "ARGV:N2P", "PARENT:bash", "UID:0", "DST:NONE", "DT1"],
        "systemctl enable 类持久化"),
    "I-侦察链_ss_whoami_uname_id": (
        ["ET:EXEC", "PROC:ss", "ARGV:N2-", "PARENT:bash", "UID:1000", "DST:NONE", "DT0",
         "ET:EXEC", "PROC:whoami", "ARGV0", "PARENT:bash", "UID:1000", "DST:NONE", "DT0",
         "ET:EXEC", "PROC:uname", "ARGV:N1-", "PARENT:bash", "UID:1000", "DST:NONE", "DT0",
         "ET:EXEC", "PROC:id", "ARGV0", "PARENT:bash", "UID:1000", "DST:NONE", "DT0"],
        "侦察链：单条都常见，密集连发(DT0)是脚本行为"),
    "K-nc高端口外联": (
        ["ET:CONN", "PROC:nc", "ARGV0", "PARENT:bash", "UID:1000", "DST:EXT:HIGH", "DT0"],
        "nc 反连，经典反向 shell"),
    # —— 真正的上下文测试：全部 token 都在词表内且各自常见，只有组合/父进程异常 ——
    "L-cat读文件_父python3": (
        ["ET:EXEC", "PROC:cat", "ARGV:N1P", "PARENT:python3", "UID:1000", "DST:NONE", "DT2"],
        "脚本批量读文件指纹"),
    "L-ctl_cat_父bash": (
        ["ET:EXEC", "PROC:cat", "ARGV:N1P", "PARENT:bash", "UID:1000", "DST:NONE", "DT2"],
        "阴性对照"),
    "M-bash_父python3": (
        ["ET:EXEC", "PROC:bash", "ARGV:N1-", "PARENT:python3", "UID:1000", "DST:NONE", "DT1"],
        "python 拉 shell = 经典植入/反连前置"),
    "M-ctl_bash_父sshd": (
        ["ET:EXEC", "PROC:bash", "ARGV:N1-", "PARENT:sshd-session", "UID:1000", "DST:NONE", "DT1"],
        "阴性对照：SSH 登录 shell"),
    "N-scp外传_父python3": (
        ["ET:EXEC", "PROC:scp", "ARGV:N2P", "PARENT:python3", "UID:1000", "DST:NONE", "DT1"],
        "脚本化数据外传"),
    "N-ctl_scp_父bash": (
        ["ET:EXEC", "PROC:scp", "ARGV:N2P", "PARENT:bash", "UID:1000", "DST:NONE", "DT1"],
        "阴性对照：手动传文件"),
    "O-sh_父python3": (
        ["ET:EXEC", "PROC:sh", "ARGV:N1-", "PARENT:python3", "UID:1000", "DST:NONE", "DT0"],
        "python -c 调 sh"),
    "P-ping_父python3": (
        ["ET:EXEC", "PROC:ping", "ARGV:N1P", "PARENT:python3", "UID:1000", "DST:NONE", "DT1"],
        "脚本化探测"),
    "P-ctl_ping_父bash": (
        ["ET:EXEC", "PROC:ping", "ARGV:N1P", "PARENT:bash", "UID:1000", "DST:NONE", "DT1"],
        "阴性对照"),
}


def main():
    model_dir, tokens_path = sys.argv[1], sys.argv[2]
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tau = ckpt["baseline_nll"]["p995"]

    events = [json.loads(l) for l in open(tokens_path)]
    ctx = [t for ev in events[-4:] for t in ev["tokens"]]

    import json as _json
    slot_tau = _json.load(open(os.path.join(model_dir, "slot_tau.json")))["slot_tau"]

    def score(ev_tokens, detail=False):
        # 兼容 7-token 旧用例：缺 PC 槽时在 ARGV 后补 PC:NONE
        if not any(t.startswith("PC:") for t in ev_tokens):
            ev_tokens = ev_tokens[:3] + ["PC:NONE"] + ev_tokens[3:]
        ids = [stoi.get(t, 0) for t in ctx + ev_tokens][-CTX:]
        n_ev, L = len(ev_tokens), len(ids)
        n_ev = min(n_ev, L - 1)
        x = torch.tensor(ids, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            lp = torch.log_softmax(model(x), dim=-1)[0]
        tgt = torch.tensor(ids[L - n_ev:], device=DEVICE)
        nll = -lp[L - n_ev - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        n_unk = sum(1 for t in ev_tokens if t not in stoi)
        # 分维度判定：每个 token 对照自己槽位的 τ
        fired = []
        for t, v in zip(ev_tokens[-n_ev:], nll.tolist()):
            if ":" in t:
                slot = t.split(":")[0]
            elif t.startswith("ARGV"):
                slot = "ARGV"
            elif t.startswith("DT"):
                slot = "DT"
            else:
                slot = t
            if v > slot_tau.get(slot, tau):
                fired.append((t, round(v, 2)))
        if detail:
            itos = {v: k for k, v in stoi.items()}
            return nll.max().item(), n_unk, [(itos.get(i, "<UNK>"), round(float(n), 2))
                                             for i, n in zip(ids[L - n_ev:], nll.tolist())]
        return nll.max().item(), n_unk, fired

    # 阴性对照补充：真实数据中这些进程的自然出现
    real = {}
    for ev in events[-30000:]:
        p = ev["tokens"][1]
        if p in ("PROC:ss", "PROC:awk", "PROC:crontab", "PROC:systemctl"):
            real.setdefault(p, []).append(ev["tokens"])

    print(f"τ(p995)={tau:.3f} | 分维度 τ={ {k: round(v,2) for k,v in slot_tau.items()} }\n")
    print(f"{'用例':<26}{'max分':>8}{'UNK':>5}{'判定':>6}  命中维度")
    for name, (toks, desc) in CASES.items():
        sc, n_unk, fired = score(toks)
        verdict = "异常" if fired or n_unk >= 1 else "正常"
        dim = ",".join(t for t, _ in fired) if fired else "-"
        print(f"{name:<26}{sc:8.2f}{n_unk:>5}{verdict:>6}  {dim}")

    print("\n关键用例逐 token 分解（哪个 token 在报警）:")
    for name in ("F-ctl_awk_父bash", "L-cat读文件_父python3", "L-ctl_cat_父bash",
                 "M-bash_父python3", "M-ctl_bash_父sshd"):
        toks = CASES[name][0]
        sc, n_unk, brk = score(toks, detail=True)
        print(f"  {name} (max={sc:.2f}):")
        print("    " + " | ".join(f"{t}:{n}" for t, n in brk))

    print("\n真实对照（近3万事件中自然出现的同原语，取最高分）:")
    for p, lst in sorted(real.items()):
        scores = [score(t)[0] for t in lst]
        print(f"  {p:<16} n={len(lst):<4} max={max(scores):6.2f} mean={np.mean(scores):5.2f}")


if __name__ == "__main__":
    main()
