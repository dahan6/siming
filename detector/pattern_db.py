#!/usr/bin/env python3
"""模式库：统一 JSONL 库 + 匹配引擎（骨架）
设计理念（2026-07-30 讨论定稿）：
- 一个库两种条目：pattern（抽象模式，手写/泛化来）与 sequence（实采具体序列）
- 样本→模式升级流水线是防臃肿核心：同技术样本攒够后泛化为模式，样本退役
- 匹配纪律：字段级谓词（eq/in），不用正则；proc 可按具体名或类别匹配
- 每个模式标 precision 与 context_required：强信号可独立报警，弱信号必须带上下文

条目 schema（一行一条 JSON）：
  公共: id, type("pattern"|"sequence"), technique(ATT&CK号), name, severity(1-5),
        source, created, notes
  pattern 专有:
        match: { et?, proc_in?|proccat_in?, argv?, parent_in?, uid?, dst?, dt_in? }
        precision: "exact"|"category"
        context_required: bool  （true 时 parent/dst/dt 至少一项命中才算）
  sequence 专有:
        sequence: [token,...]  （原型学习阶段的正样本）

用法:
  from pattern_db import PatternDB
  db = PatternDB("patterns.jsonl")
  hits = db.match(["ET:EXEC","PROC:bash",...])   # -> [条目,...]
"""
import json
import time

# 进程类别抽象（跨机迁移的关键：稀有度按类别走，不按具体名）
PROC_CATEGORIES = {
    "shell": {"bash", "sh", "dash", "zsh", "ash"},
    "script": {"python3", "python", "perl", "ruby", "node", "php"},
    "nettool": {"nc", "ncat", "socat", "curl", "wget", "ssh", "scp", "sftp", "telnet"},
    "recon": {"ss", "netstat", "lsof", "ps", "whoami", "id", "uname", "hostname",
              "ifconfig", "ip", "arp", "ping", "nmap", "awk", "cat", "head", "tail"},
    "persist": {"crontab", "systemctl", "at", "batch", "update-rc.d", "chkconfig"},
    "pkg": {"apt", "apt-get", "dpkg", "snap", "pip", "pip3", "npm", "cargo"},
    "encode": {"base64", "openssl", "xxd", "od"},
    "compiler": {"gcc", "cc", "g++", "make", "rustc", "ld.lld", "rust-lld", "collect2"},
}


def _tokmap(tokens):
    """token 列表 -> {槽位: 值}（ET/PROC/ARGV/PARENT/UID/DST/DT）
    注意 ARGV0 和 DT0~DT5 是无冒号形态，需特判。"""
    m = {}
    for t in tokens:
        if ":" in t:
            k, v = t.split(":", 1)
            m[k] = v
        elif t.startswith("ARGV"):
            m["ARGV"] = t
        elif t.startswith("DT"):
            m["DT"] = t
    return m


class PatternDB:
    def __init__(self, path):
        self.path = path
        self.entries = []
        if not hasattr(self, "_noop"):
            try:
                for line in open(path):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self.entries.append(json.loads(line))
            except FileNotFoundError:
                pass

    def add(self, entry):
        entry.setdefault("created", time.strftime("%F %T"))
        self.entries.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _match_pattern(self, match, tm):
        """字段级谓词匹配。返回 (命中?, 上下文维度是否参与命中)"""
        ctx_hit = False
        if "et" in match and tm.get("ET") != match["et"]:
            return False, ctx_hit
        if "proc_in" in match and tm.get("PROC") not in match["proc_in"]:
            return False, ctx_hit
        if "proc_not_in" in match and tm.get("PROC") in match["proc_not_in"]:
            return False, ctx_hit
        if "proccat_in" in match:
            cats = match["proccat_in"]
            proc = tm.get("PROC")
            if not any(proc in PROC_CATEGORIES.get(c, set()) for c in cats):
                return False, ctx_hit
        if "argv" in match and tm.get("ARGV") != match["argv"]:
            return False, ctx_hit
        if "parent_in" in match:
            if tm.get("PARENT") not in match["parent_in"]:
                return False, ctx_hit
            ctx_hit = True
        if "uid" in match and tm.get("UID") != str(match["uid"]):
            return False, ctx_hit
        if "dst" in match:
            if tm.get("DST") != match["dst"]:
                return False, ctx_hit
            ctx_hit = True
        if "dt_in" in match:
            if tm.get("DT") not in match["dt_in"]:
                return False, ctx_hit
            ctx_hit = True
        return True, ctx_hit

    def match(self, tokens):
        """对一条事件 token 序列跑全部 pattern 条目，返回命中列表（附归因）"""
        tm = _tokmap(tokens)
        hits = []
        for e in self.entries:
            if e.get("type") != "pattern":
                continue  # sequence 条目留给原型学习阶段，此处不匹配
            ok, ctx_hit = self._match_pattern(e["match"], tm)
            if not ok:
                continue
            if e.get("context_required") and not ctx_hit:
                continue  # 弱信号：必须带上下文维度命中
            hits.append({**e, "matched_tokens": tokens})
        return hits

    def stats(self):
        pat = sum(1 for e in self.entries if e.get("type") == "pattern")
        seq = sum(1 for e in self.entries if e.get("type") == "sequence")
        return {"patterns": pat, "sequences": seq, "total": len(self.entries)}
