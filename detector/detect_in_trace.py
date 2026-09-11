#!/usr/bin/env python3
"""轨迹检测：对一份 VM tracee 文件跑三网融合，输出告警数与命中技术号。
用法: detect_in_trace.py <model_dir> <trace.jsonl>
输出最后一行 JSON: {"alerts": N, "proto_hits": [...], "pattern_hits": [...]}
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_raw_tracee import event_to_tokens
from pattern_db import PatternDB
from proto_head import embed_sequence
from train_prior import CTX, DEVICE, TinyGPT


def slot_of(tok):
    if ":" in tok:
        return tok.split(":")[0]
    if tok.startswith("ARGV"):
        return "ARGV"
    if tok.startswith("DT"):
        return "DT"
    return tok


def main():
    model_dir, trace = sys.argv[1], sys.argv[2]
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    det = os.path.dirname(os.path.abspath(__file__))
    slot_tau = json.load(open(os.path.join(model_dir, "slot_tau.json")))["slot_tau"]
    db = PatternDB(os.path.join(det, "patterns.jsonl"))
    proto = json.load(open(os.path.join(det, "prototypes.jsonl")))
    P = {t: (torch.tensor(v["prototypes"]).to(DEVICE), v["radii"][0])
         for t, v in proto["techniques"].items()}

    events = []
    for line in open(trace, errors="replace"):
        if line.strip().startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    window, prev_ts = [], None
    alerts = 0
    proto_hits, pattern_hits = set(), set()
    for ev in events:
        ts = ev.get("timestamp", 0)
        delta = 0 if prev_ts is None else max(0, (ts - prev_ts) // 1_000_000)
        prev_ts = ts
        toks = event_to_tokens(ev, delta)
        ids = [stoi.get(t, 0) for t in toks]
        window = (window + ids)[-CTX:]
        n, L = len(ids), len(window)
        start = max(L - n, 1)
        with torch.no_grad():
            x = torch.tensor(window, device=DEVICE).unsqueeze(0)
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(window[start:L], device=DEVICE)
            nll = -lp[start - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        fired = any(v > slot_tau.get(slot_of(t), 1.0) for t, v in zip(toks, nll.tolist()))
        n_unk = any(t not in stoi for t in toks)
        hits = db.match(toks)
        e = embed_sequence(model, stoi, toks)
        phit = None
        for t, (ps, r) in P.items():
            if (ps - e).norm(dim=1).min().item() <= r:
                phit = t
                break
        if fired or n_unk or hits or phit:
            alerts += 1
        for h in hits:
            pattern_hits.add(h["technique"])
        if phit:
            proto_hits.add(phit)

    print(json.dumps({"alerts": alerts, "proto_hits": sorted(proto_hits),
                      "pattern_hits": sorted(pattern_hits), "events": len(events)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
