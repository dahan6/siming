#!/usr/bin/env python3
"""统计层复合升级：PREV 槽 + EWMA 滑动窗口序列分

升级内容：
1. PREV 槽：每个事件加第 9 个 token = 前一个事件的 PROC 名
   - 让模型学到跨事件依赖（find→cat 比 find→ls 更可疑）
2. EWMA 序列分：不只看单事件 max(NLL)，看窗口级趋势
   - 平滑噪声，抓持续异常，降 FPR

用法:
  # 生成 PREV 增强数据
  python3 stat_layer_upgrade.py --prep data/audit_all.jsonl --out data/audit_prev.jsonl

  # 训练带 PREV 的 prior
  python3 stat_layer_upgrade.py --train data/audit_prev.jsonl --out model-stat-v2

  # EWMA 评估
  python3 stat_layer_upgrade.py --eval model-stat-v2 --data data/audit_all.jsonl
"""
import json, os, sys, random, argparse, math
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import TinyGPT, CTX, D_MODEL, N_LAYER, N_HEAD, DEVICE

DET = os.path.dirname(os.path.abspath(__file__))

# ═══ 1. PREV 槽：数据预处理 ═══

def add_prev_token(events):
    """给每条事件的 tokens 加 PREV:xxx（前一个事件的 PROC）"""
    result = []
    prev_proc = "NONE"
    for ev in events:
        tokens = list(ev["tokens"])
        # 找当前 PROC
        cur_proc = "NONE"
        for t in tokens:
            if t.startswith("PROC:"):
                cur_proc = t.split(":", 1)[1]
                break
        # 在 PROC 后面插入 PREV
        prev_token = f"PREV:{prev_proc}"
        # 插到 ARGV 后面（位置 3）
        tokens.insert(3, prev_token)
        result.append({**ev, "tokens": tokens})
        prev_proc = cur_proc
    return result


def prep_data(input_path, output_path):
    """预处理：加 PREV 槽"""
    events = []
    for line in open(input_path):
        line = line.strip()
        if not line or line.startswith("#"): continue
        try:
            e = json.loads(line)
            if "tokens" in e and len(e["tokens"]) >= 6:
                events.append(e)
        except: continue

    augmented = add_prev_token(events)

    with open(output_path, "w") as f:
        for ev in augmented:
            f.write(json.dumps(ev) + "\n")

    # 统计 PREV 分布
    prev_dist = Counter()
    for ev in augmented:
        for t in ev["tokens"]:
            if t.startswith("PREV:"):
                prev_dist[t] += 1
                break
    print(f"PREV 增强: {len(augmented)} 事件 → {output_path}")
    print(f"PREV 分布 (top 10): {dict(prev_dist.most_common(10))}")
    return augmented


# ═══ 2. 训练带 PREV 的 prior ═══

def train_prior_prev(data_path, out_dir, epochs=5):
    """训练带 PREV 槽的 TinyGPT"""
    seqs = [json.loads(l)["tokens"] for l in open(data_path)]
    c = Counter(t for ev in seqs for t in ev)
    vocab = ["<UNK>"] + sorted(t for t, n in c.items() if n >= 2)
    stoi = {t: i for i, t in enumerate(vocab)}
    print(f"词表: {len(stoi)} tokens (含 PREV)")

    ids = torch.tensor([stoi.get(t, 0) for ev in seqs for t in ev], dtype=torch.long)
    split = int(len(ids) * 0.85)
    train_ids, val_ids = ids[:split], ids[split:]

    model = TinyGPT(len(stoi)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()
    random.seed(42)

    for ep in range(epochs):
        starts = list(range(0, len(train_ids) - CTX - 1))
        random.shuffle(starts)
        starts = starts[:5000]
        tot, nb = 0, 0
        for i in range(0, len(starts) - 128, 128):
            chunk = starts[i:i+128]
            xs = torch.stack([train_ids[s:s+CTX] for s in chunk]).to(DEVICE)
            ys = torch.stack([train_ids[s+1:s+CTX+1] for s in chunk]).to(DEVICE)
            loss = lossf(model(xs).view(-1, len(stoi)), ys.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  ep{ep+1}: loss={tot/nb:.4f}")

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "stoi": stoi,
        "config": {"d_model": D_MODEL, "n_layer": N_LAYER, "n_head": N_HEAD, "ctx": CTX},
        "has_prev": True,
    }, os.path.join(out_dir, "prior.pt"))
    print(f"保存 → {out_dir}/prior.pt")
    return model, stoi


# ═══ 3. EWMA 滑动窗口序列分 ═══

class EWMAWindowScorer:
    """EWMA 滑动窗口异常分

    不只看单事件 max(NLL)，看窗口级趋势：
    - EWMA：指数加权移动平均，平滑噪声
    - burst_score：窗口内连续高 NLL 事件的比例
    - trend_score：NLL 是否在上升（渐进式攻击）

    阈值自适应：用前 N 个事件的分布标定 baseline，
    后续事件与 baseline 对比。
    """

    def __init__(self, alpha=0.3, window=32, warmup=100):
        self.alpha = alpha          # EWMA 平滑系数
        self.window = window        # 滑窗大小
        self.warmup = warmup        # 预热期：前 warmup 个事件用于标定 baseline
        self.ewma = 0.0
        self.recent_nlls = deque(maxlen=window)
        self.baseline_nlls = []     # 预热期收集的 NLL
        self.baseline_p95 = None    # baseline 的 p95
        self.baseline_p99 = None    # baseline 的 p99
        self.baseline_mean = None
        self.baseline_std = None
        self._calibrated = False

    def update(self, event_nll_max):
        """喂一个事件的 max(NLL)，返回判定结果"""
        # 预热期：收集 baseline
        if not self._calibrated:
            self.baseline_nlls.append(event_nll_max)
            self.ewma = self.alpha * event_nll_max + (1 - self.alpha) * self.ewma
            self.recent_nlls.append(event_nll_max)
            if len(self.baseline_nlls) >= self.warmup:
                arr = np.array(self.baseline_nlls)
                self.baseline_mean = arr.mean()
                self.baseline_std = arr.std()
                self.baseline_p95 = float(np.percentile(arr, 95))
                self.baseline_p99 = float(np.percentile(arr, 99))
                self._calibrated = True
                # 标定完成，返回 normal
            return {"ewma": round(self.ewma, 3), "status": "warmup", "is_anomalous": False}

        # 标定后：用 baseline 阈值判定
        self.ewma = self.alpha * event_nll_max + (1 - self.alpha) * self.ewma
        self.recent_nlls.append(event_nll_max)

        # burst score：窗口内超 baseline p95 的比例
        burst_count = sum(1 for n in self.recent_nlls if n > self.baseline_p95)
        burst_score = burst_count / max(len(self.recent_nlls), 1)

        # trend score：后半窗口均值 - 前半窗口均值
        n = len(self.recent_nlls)
        if n >= 8:
            half = n // 2
            trend = np.mean(list(self.recent_nlls)[half:]) - np.mean(list(self.recent_nlls)[:half])
        else:
            trend = 0.0

        # 判定：EWMA 超 baseline p99 或 burst >50%
        # 不用固定阈值，用 baseline 动态阈值
        ewma_anomalous = self.ewma > self.baseline_p99
        burst_anomalous = burst_score > 0.5

        # 额外：z-score（当前 EWMA 偏离 baseline 多少个标准差）
        z_score = (self.ewma - self.baseline_mean) / max(self.baseline_std, 0.01)
        z_anomalous = z_score > 3  # 3σ 规则

        return {
            "ewma": round(self.ewma, 3),
            "z_score": round(z_score, 2),
            "burst": round(burst_score, 3),
            "trend": round(trend, 3),
            "baseline_p95": round(self.baseline_p95, 3) if self.baseline_p95 else None,
            "baseline_p99": round(self.baseline_p99, 3) if self.baseline_p99 else None,
            "is_anomalous": (ewma_anomalous or burst_anomalous) and z_anomalous,
            "status": "active",
        }


from collections import deque


# ═══ 4. 评估：PREV + EWMA 联合 ═══

def evaluate(model, stoi, events_path, prev_enhanced=False):
    """在数据上评估统计层（PREV + EWMA）"""
    events = [json.loads(l) for l in open(events_path)]

    # 如果需要 PREV 增强
    if prev_enhanced:
        events = add_prev_token(events)

    model.eval()
    ewma_scorer = EWMAWindowScorer(alpha=0.3, window=32)

    # per-slot τ
    slot_tau = {}
    slot_nlls = defaultdict(list)

    window = []
    all_nlls = []

    # 先收集所有 NLL 用于标定 τ
    for ev in events:
        tokens = ev["tokens"]
        ids = [stoi.get(t, 0) for t in tokens]
        window_ids = (window + ids)[-CTX:]
        n_toks = len(ids)
        L = len(window_ids)
        start = max(L - n_toks, 1)
        if L < 2: continue

        x = torch.tensor(window_ids).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(window_ids[start:L]).to(DEVICE)
            nll = -lp[start-1:L-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

        for tok, val in zip(tokens, nll.tolist()):
            slot = tok.split(":")[0] if ":" in tok else (tok[:4] if tok.startswith("DT") else tok[:5])
            slot_nlls[slot].append(val)

        window = window_ids

    # 标定 τ
    MIN_TAU = 3.0
    for slot, vals in slot_nlls.items():
        arr = np.array(vals)
        if len(arr) >= 10:
            p995 = float(np.percentile(arr, 99.5))
            slot_tau[slot] = max(p995, MIN_TAU)

    # EWMA 评估
    print(f"\n=== EWMA 滑动窗口评估 ===")
    window = []
    n_alert_single = 0   # 单事件级告警
    n_alert_ewma = 0     # EWMA 级告警
    n_alert_burst = 0    # burst 级告警
    n_total = 0

    for ev in events:
        tokens = ev["tokens"]
        ids = [stoi.get(t, 0) for t in tokens]
        window_ids = (window + ids)[-CTX:]
        n_toks = len(ids)
        L = len(window_ids)
        start = max(L - n_toks, 1)
        if L < 2: continue
        n_total += 1

        x = torch.tensor(window_ids).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(window_ids[start:L]).to(DEVICE)
            nll = -lp[start-1:L-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

        event_max_nll = nll.max().item()

        # 单事件级
        slot_fired = False
        for tok, val in zip(tokens, nll.tolist()):
            slot = tok.split(":")[0] if ":" in tok else (tok[:4] if tok.startswith("DT") else tok[:5])
            if val > slot_tau.get(slot, 999):
                slot_fired = True
                break
        if slot_fired:
            n_alert_single += 1

        # EWMA 级
        result = ewma_scorer.update(event_max_nll)
        if result["is_anomalous"]:
            n_alert_ewma += 1
        if result["burst"] > 0.3:
            n_alert_burst += 1

        window = window_ids

    print(f"事件: {n_total}")
    print(f"单事件告警: {n_alert_single} ({n_alert_single/max(n_total,1)*100:.1f}%)")
    print(f"EWMA 告警: {n_alert_ewma} ({n_alert_ewma/max(n_total,1)*100:.1f}%)")
    print(f"Burst 告警: {n_alert_burst} ({n_alert_burst/max(n_total,1)*100:.1f}%)")
    print(f"\nτ:")
    for s, t in sorted(slot_tau.items()):
        print(f"  {s}: {t:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", help="预处理数据加 PREV 槽")
    ap.add_argument("--out", help="预处理输出路径")
    ap.add_argument("--train", help="训练数据路径")
    ap.add_argument("--eval", help="评估模型目录")
    ap.add_argument("--data", help="评估数据路径")
    args = ap.parse_args()

    if args.prep:
        prep_data(args.prep, args.out or args.prep.replace(".jsonl", "_prev.jsonl"))
    elif args.train:
        train_prior_prev(args.train, args.out or "model-stat-v2")
    elif args.eval:
        model_dir = args.eval
        ckpt = torch.load(os.path.join(model_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
        stoi = ckpt["stoi"]
        model = TinyGPT(len(stoi)).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        has_prev = ckpt.get("has_prev", False)
        print(f"模型: {model_dir} (词表 {len(stoi)}, PREV={has_prev})")
        evaluate(model, stoi, args.data, prev_enhanced=has_prev)


if __name__ == "__main__":
    main()
