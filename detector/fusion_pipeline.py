#!/usr/bin/env python3
"""司命六网融合管道 v4

融合五层检测结果，用交叉确认降低误报：

P0 最高置信度：统计=异常 + 语义=攻击 → 自动处置
P1 高置信度：自适应层命中已知攻击模式
P2 中置信度：语义=攻击（统计未确认）→ 告警
P3 中置信度：统计=异常（语义未确认）→ 参考
P4 时序异常：CV<1.5 或 FFT 周期检测 → 告警
P5 低置信度：弱信号 → 记录不告警

用法:
  python3 fusion_pipeline.py --eval data/audit_all.jsonl
  python3 fusion_pipeline.py --eval data/synth_attacks_v4.jsonl
"""
import json, os, sys, time, argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter, defaultdict, deque
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import TinyGPT, CTX
from train_semantic import ContrastiveProjection, ClassificationHead, LABELS
from temporal_analyzer import TemporalAnalyzer
from temporal_fft import FFTAnalyzer
from adaptive_detector import AdaptiveDetector

DET = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"


class FusionPipeline:
    """六网融合管道"""

    def __init__(self):
        # 统计层
        self.stat_model = None
        self.stat_stoi = None
        self.stat_tau = {}

        # 语义层
        self.sem_model = None
        self.sem_stoi = None
        self.proj = None
        self.clf = None

        # 时序层
        self.temporal = TemporalAnalyzer(min_samples=20)
        self.fft = FFTAnalyzer(min_samples=32, max_samples=256)

        # 自适应层
        self.adaptive = AdaptiveDetector(window_size=200, cooldown=50)

        # EWMA
        self.ewma = 0.0
        self.ewma_alpha = 0.3
        self.ewma_baseline = []
        self.ewma_p99 = None
        self.ewma_warmup = 100
        self.ewma_count = 0

        # 统计
        self.n_alert = Counter()

    def load_stat_model(self, model_dir):
        """加载统计层模型"""
        ckpt = torch.load(os.path.join(model_dir, "prior.pt"),
                          map_location=DEVICE, weights_only=False)
        self.stat_stoi = ckpt["stoi"]
        self.stat_model = TinyGPT(len(self.stat_stoi))
        self.stat_model.load_state_dict(ckpt["model"])
        self.stat_model.causal_mask = torch.triu(
            torch.full((CTX, CTX), float("-inf")), diagonal=1)
        self.stat_model.eval()

        # 加载 τ
        for name in ("slot_tau_local.json", "slot_tau_vm.json", "slot_tau.json"):
            path = os.path.join(model_dir, name)
            if os.path.exists(path):
                self.stat_tau = json.load(open(path))["slot_tau"]
                break

        # PREV 压缩映射
        self.has_prev = ckpt.get("has_prev", False)

    def load_semantic_model(self, model_dir):
        """加载语义层模型"""
        ckpt = torch.load(os.path.join(model_dir, "full_model.pt"),
                          map_location=DEVICE, weights_only=False)
        self.sem_stoi = ckpt["stoi"]
        self.sem_model = TinyGPT(len(self.sem_stoi))
        self.sem_model.load_state_dict(ckpt["model"])
        self.sem_model.causal_mask = torch.triu(
            torch.full((CTX, CTX), float("-inf")), diagonal=1)
        self.sem_model.eval()

        self.proj = ContrastiveProjection()
        self.proj.load_state_dict(ckpt["proj"])
        self.proj.eval()

        self.clf = ClassificationHead()
        self.clf.load_state_dict(ckpt["classifier"])
        self.clf.eval()

    def _classify_prev(self, proc):
        """压缩 PREV 类别"""
        if proc in ("systemd","systemd-executor","systemd-resolve","systemd-journal",
                    "systemd-udevd","systemd-logind","systemd-timesyncd"): return "PREV:system"
        if proc in ("snap","snapctl","snap-confine","snap-seccomp","snap-exec"): return "PREV:snap"
        if proc.startswith("kworker") or proc.startswith("kthread") or proc=="kthreadd": return "PREV:kernel"
        if proc in ("apt","dpkg","apt-get"): return "PREV:pkg"
        if proc in ("bash","sh","dash","zsh"): return "PREV:shell"
        if proc in ("cat","head","tail","less","more"): return "PREV:fileread"
        if proc in ("find","locate"): return "PREV:search"
        if proc in ("ss","netstat","lsof","ip","ifconfig","nmap"): return "PREV:netrecon"
        if proc in ("ps","top","pgrep","pidof"): return "PREV:procrecon"
        if proc in ("journalctl","dmesg"): return "PREV:log"
        if proc in ("sudo","su"): return "PREV:priv"
        if proc in ("curl","wget","nc","ssh","scp"): return "PREV:nettool"
        if proc in ("python3","python","perl","ruby","node"): return "PREV:lang"
        if proc in ("grep","awk","sed","sort","cut","tr","readlink"): return "PREV:parser"
        if proc in ("suricata","falco"): return "PREV:sec"
        if proc in ("sleep","env","date","hostname","uname","whoami","id"): return "PREV:misc"
        if proc.startswith("(") or proc.startswith("firmware") or proc.startswith("nm-"): return "PREV:daemon"
        if proc in ("cron","CRON","atd"): return "PREV:scheduler"
        return "PREV:other"

    def _slot_of(self, tok):
        if ":" in tok: return tok.split(":")[0]
        if tok.startswith("ARGV"): return "ARGV"
        if tok.startswith("DT"): return "DT"
        return tok

    def _stat_score(self, tokens):
        """统计层打分：per-slot NLL"""
        ids = [self.stat_stoi.get(t, 0) for t in tokens]
        window = ids[-CTX:]
        n_toks = len(ids)
        L = len(window)
        start = max(L - n_toks, 1)
        if L < 2:
            return False, 0.0, []

        x = torch.tensor(window).unsqueeze(0)
        with torch.no_grad():
            lp = torch.log_softmax(self.stat_model(x), dim=-1)[0]
            tgt = torch.tensor(window[start:L])
            nll = -lp[start-1:L-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

        event_max = nll.max().item()
        fired = []
        for tok, val in zip(tokens, nll.tolist()):
            s = self._slot_of(tok)
            if val > self.stat_tau.get(s, 999):
                fired.append((s, val))
        return len(fired) > 0, event_max, fired

    def _semantic_score(self, tokens):
        """语义层打分：分类头 + 后置规则"""
        ids = [self.sem_stoi.get(t, 0) for t in tokens][-CTX:]
        if len(ids) < 2: ids *= 2
        x = torch.tensor(ids).unsqueeze(0)
        with torch.no_grad():
            t = x.size(1)
            h = self.sem_model.tok(x) + self.sem_model.pos(torch.arange(t))
            h = self.sem_model.blocks(h, mask=self.sem_model.causal_mask[:t, :t])
            emb = self.sem_model.norm(h).mean(dim=1).squeeze(0)
            z = self.proj(emb.unsqueeze(0))
            out = self.clf(z)
            probs = F.softmax(out, dim=-1)[0]

        pred = out.argmax(-1).item()
        conf = probs[pred].item()
        label = LABELS[pred]

        # 后置规则
        proc = ""
        uid = ""
        parent = ""
        for tok in tokens:
            if tok.startswith("PROC:"): proc = tok.split(":", 1)[1]
            elif tok.startswith("UID:"): uid = tok.split(":", 1)[1]
            elif tok.startswith("PARENT:"): parent = tok.split(":", 1)[1]

        if label == "privesc" and (proc == "sudo" or (uid == "0" and parent == "sudo")):
            label = "benign"
        if label in ("privesc", "persist") and proc in ("systemd", "systemd-executor", "systemd-resolve", "modprobe"):
            label = "benign"

        # 扩展后置规则：宿主机常见良性命令
        # 扩展后置规则：宿主机常见良性命令
        BENIGN_PROCS = {
            # 系统/工具命令
            "readlink", "dash", "dirname", "md5sum", "hostname", "whoami",
            "date", "env", "uname", "pwd", "echo", "id", "ls", "wc",
            # 解析/处理
            "grep", "awk", "sed", "sort", "cut", "tr", "stat", "gawk",
            # 文件读取（良性上下文）
            "head", "tail", "cat", "less", "more",
            # 文件操作（非 SUID 操作时是良性）
            "du",
            # 网络（非攻击上下文）
            "dig",
            # Shell
            "bash", "sh",
            # 安全/系统
            "suricata", "unix_chkpwd", "getent", "snap", "snapctl",
            # 版本控制
            "git",
            # 开发运行时
            "node", "cargo",
            # python3 变体
            "python3.12", "python3.11", "python3.10",
        }

        # find 和 chmod 是条件白名单——SUID 相关操作不算良性
        CONDITIONAL_BENIGN = {"find", "chmod"}
        # 检查是否有 SUID 相关的 ARGV 或 PC
        argv_str = ""
        for tok in tokens:
            if tok.startswith("ARGV:"): argv_str = tok
        is_suid_related = ("N3P" in argv_str or "4000" in argv_str or
                           uid == "0" and proc in CONDITIONAL_BENIGN)

        if label in ("privesc", "recon", "persist") and proc in BENIGN_PROCS:
            label = "benign"
        elif label in ("privesc", "persist") and proc in CONDITIONAL_BENIGN and not is_suid_related:
            label = "benign"

        if conf < 0.7:
            label = "benign"

        is_attack = label != "benign"
        return is_attack, label, conf

    def process_event(self, tokens, ts_str=None, prev_proc=None):
        """处理一个事件，返回融合判定

        Returns:
            dict: {prio, stat_anom, sem_label, sem_conf, temporal, adaptive, fft}
        """
        # ── 统计层 ──
        stat_anom, stat_max_nll, stat_fired = self._stat_score(tokens)

        # ── 语义层 ──
        sem_attack, sem_label, sem_conf = self._semantic_score(tokens)

        # ── 时序层 ──
        proc_name = "?"
        uid_val = "?"
        for t in tokens:
            if t.startswith("PROC:"): proc_name = t.split(":", 1)[1]
            elif t.startswith("UID:"): uid_val = t.split(":", 1)[1]

        ts_s = 0
        if ts_str:
            try: ts_s = datetime.fromisoformat(ts_str).timestamp()
            except: ts_s = time.time()
        else:
            ts_s = time.time()

        temp_result = self.temporal.update(proc_name, ts_s * 1000, self.ewma, uid=uid_val)
        # 只对 UID:1000 用户进程做时序检测（系统进程天然是机器节奏）
        temp_anom = (temp_result and temp_result.get("verdict") == "anomalous"
                     and uid_val not in ("0", "?", "1"))

        # FFT
        fft_result = {"periodic": False}
        if len(self.fft.timestamps) >= 2:
            last_ts = self.fft.timestamps[-1] if self.fft.timestamps else ts_s
            fft_result = self.fft.update(ts_s)

        # ── 自适应层 ──
        adapt_alerts = self.adaptive.update(tokens, ts=ts_str or "")
        adapt_high = any(a["severity"] >= 5 for a in adapt_alerts)
        adapt_low = any(a["severity"] < 5 for a in adapt_alerts)

        # ── EWMA 更新 ──
        self.ewma = self.ewma_alpha * stat_max_nll + (1 - self.ewma_alpha) * self.ewma
        self.ewma_count += 1
        if self.ewma_count <= self.ewma_warmup:
            self.ewma_baseline.append(stat_max_nll)
            if self.ewma_count == self.ewma_warmup:
                arr = np.array(self.ewma_baseline)
                self.ewma_p99 = float(np.percentile(arr, 99))

        ewma_anom = False
        if self.ewma_p99 and self.ewma_count > self.ewma_warmup:
            ewma_anom = self.ewma > self.ewma_p99

        # ═══ 融合判定 ═══
        if stat_anom and sem_attack:
            prio = "P0"  # 交叉确认：最高置信度
        elif adapt_high:
            prio = "P1"  # 自适应层命中
        elif sem_attack:
            prio = "P2"  # 语义层独判
        elif stat_anom or ewma_anom:
            prio = "P3"  # 统计层独判
        elif temp_anom:
            # 时序异常需要连续 3 次才报（避免单次噪声）
            self._temp_streak = getattr(self, '_temp_streak', 0) + 1
            if self._temp_streak >= 3:
                prio = "P4"
            else:
                prio = None
        elif fft_result.get("periodic"):
            prio = "P4"  # FFT 周期检测
        elif adapt_low:
            prio = "P5"  # 弱信号
        else:
            prio = None
            self._temp_streak = 0

        if prio:
            self.n_alert[prio] += 1

        return {
            "prio": prio,
            "stat_anom": stat_anom,
            "stat_nll": round(stat_max_nll, 2),
            "sem_label": sem_label,
            "sem_conf": round(sem_conf, 3),
            "temporal": temp_anom,
            "fft_periodic": fft_result.get("periodic", False),
            "fft_period": fft_result.get("dominant_period_s"),
            "adaptive": [a["type"] for a in adapt_alerts] if adapt_alerts else None,
            "ewma_anom": ewma_anom,
        }

    def summary(self):
        total = sum(self.n_alert.values())
        lines = [f"告警 {total} 条:"]
        for p in ["P0", "P1", "P2", "P3", "P4", "P5"]:
            if self.n_alert[p] > 0:
                lines.append(f"  {p}: {self.n_alert[p]}")
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="六网融合管道")
    ap.add_argument("--eval", help="评估数据文件")
    ap.add_argument("--stat-model", default=os.path.join(DET, "model-stat-v3"),
                    help="统计层模型目录")
    ap.add_argument("--sem-model", default=os.path.join(DET, "model-semantic-v5"),
                    help="语义层模型目录")
    args = ap.parse_args()

    pipe = FusionPipeline()
    pipe.load_stat_model(args.stat_model)
    pipe.load_semantic_model(args.sem_model)

    print(f"统计层: {args.stat_model} (词表 {len(pipe.stat_stoi)})")
    print(f"语义层: {args.sem_model} (词表 {len(pipe.sem_stoi)})")

    if not args.eval:
        print("用法: --eval <data.jsonl>")
        return

    # 加载数据
    events = []
    for line in open(args.eval):
        line = line.strip()
        if not line or line.startswith("#"): continue
        try:
            e = json.loads(line)
            if "tokens" in e:
                events.append(e)
        except: continue

    print(f"\n评估: {len(events)} 事件")

    # 如果统计层用 PREV，需要加 PREV token
    prev_class = "PREV:none"
    results = []
    for ev in events:
        tokens = list(ev["tokens"])

        # 加 PREV（如果统计层需要）
        if pipe.has_prev:
            cur_proc = "unknown"
            for t in tokens:
                if t.startswith("PROC:"):
                    cur_proc = t.split(":", 1)[1]
                    break
            tokens.insert(3, prev_class)
            prev_class = pipe._classify_prev(cur_proc)

        ts = ev.get("ts", "")
        result = pipe.process_event(tokens, ts_str=ts)
        if result["prio"]:
            results.append({**result, "label": ev.get("label", "benign"),
                           "proc": tokens[1] if len(tokens) > 1 else "?"})

    total = len(events)
    n_alert = sum(1 for r in results if r["prio"])
    fpr = n_alert / max(total, 1) * 100

    print(f"\n告警: {n_alert}/{total} = {fpr:.1f}%")
    print(pipe.summary())

    # 如果有标签，计算 TPR
    if any(e.get("label") for e in events):
        print(f"\n=== 按标签 ===")
        by_label = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
        for r in results:
            label = r.get("label", "benign")
            if label == "benign":
                by_label["benign"]["fp"] += 1
            else:
                by_label[label]["tp"] += 1

        # 统计未告警的
        alerted_procs = set()
        for r in results:
            alerted_procs.add(id(r))

        for label in sorted(by_label):
            d = by_label[label]
            if label == "benign":
                print(f"  benign FPR: {d['fp']}")
            else:
                print(f"  {label:10s}: {d['tp']} detected")

    # 告警详情（前 20 条）
    if results:
        print(f"\n=== 告警详情 (前 20) ===")
        print(f"{'Prio':<4} {'Label':<10} {'Stat':>5} {'SemLab':<10} {'SemConf':>7} {'Proc':<15} {'Adaptive'}")
        print("-" * 75)
        for r in results[:20]:
            adapt = ",".join(r.get("adaptive", []) or [])
            print(f"{r['prio']:<4} {r.get('label','?'):<10} {'Y' if r['stat_anom'] else 'N':>5} "
                  f"{r['sem_label']:<10} {r['sem_conf']:>7.2f} {r['proc']:<15} {adapt}")


if __name__ == "__main__":
    main()
