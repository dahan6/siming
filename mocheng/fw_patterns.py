#!/usr/bin/env python3
"""墨城防火墙: 网络攻击模式库

字段级谓词匹配引擎（复用司命 PatternDB 设计理念）。
每条模式标 precision 与 context_required：强信号可独立报警，弱信号必须带上下文。

条目 schema（一行一条 JSON）:
  公共: id, type("pattern"), technique, name, severity(1-5), action, created
  pattern 专有:
        match: { proto?, dir?, srcnet?, dstnet?, portcls_in?, sizecls_in?, dt_in? }
        precision: "exact"|"category"
        context_required: bool (true 时 dstnet/dt 至少一项命中才算)

action 值:
  DROP    — 立即丢弃/阻断
  THROTTLE — 限速
  ALERT   — 仅告警
  LOG     — 仅记录
"""
import json
import os
import time

sys_dir = os.path.dirname(os.path.abspath(__file__))


def _tokmap(tokens):
    """token 列表 → {槽位: 值}"""
    m = {}
    for t in tokens:
        if ":" in t:
            k, v = t.split(":", 1)
            m[k] = v
        elif t.startswith("DT"):
            m["DT"] = t
    return m


class FwPatternDB:
    """网络攻击模式匹配引擎"""

    def __init__(self, path=None):
        self.path = path or os.path.join(sys_dir, "fw_patterns.jsonl")
        self.entries = []
        self.load()

    def load(self):
        self.entries = []
        try:
            for line in open(self.path):
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
        if "proto" in match and tm.get("PROTO") != match["proto"]:
            return False, ctx_hit
        if "dir" in match and tm.get("DIR") != match["dir"]:
            return False, ctx_hit
        if "srcnet" in match and tm.get("SRCNET") != match["srcnet"]:
            return False, ctx_hit
        if "dstnet" in match and tm.get("DSTNET") != match["dstnet"]:
            return False, ctx_hit
        if "portcls_in" in match:
            if tm.get("PORTCLS") not in match["portcls_in"]:
                return False, ctx_hit
        if "portcls_not_in" in match:
            if tm.get("PORTCLS") in match["portcls_not_in"]:
                return False, ctx_hit
        if "sizecls_in" in match:
            if tm.get("SIZECLS") not in match["sizecls_in"]:
                return False, ctx_hit
        if "dt_in" in match:
            if tm.get("DT") not in match["dt_in"]:
                return False, ctx_hit
            ctx_hit = True
        # 注意：dstnet 是核心匹配条件，不算上下文佐证
        # 只有 DT（时序维度）才是上下文佐证
        return True, ctx_hit

    def match(self, tokens):
        """对一条事件 token 序列跑全部 pattern 条目，返回命中列表"""
        tm = _tokmap(tokens)
        hits = []
        for e in self.entries:
            if e.get("type") != "pattern":
                continue
            ok, ctx_hit = self._match_pattern(e.get("match", {}), tm)
            if not ok:
                continue
            if e.get("context_required") and not ctx_hit:
                continue
            hits.append({**e, "matched_tokens": tokens})
        return hits

    def stats(self):
        return {"patterns": len(self.entries), "total": len(self.entries)}


# ── 默认模式库 ──
DEFAULT_PATTERNS = [
    {
        "id": "FW-001", "type": "pattern", "technique": "T1046",
        "name": "端口扫描-高端口探测", "severity": 4, "action": "DROP",
        "match": {"proto": "TCP", "dir": "OUT", "dstnet": "EXT",
                  "portcls_in": ["HIGHPORT", "FTP", "TELNET", "SMB", "RDP", "MSSQL", "MYSQL", "POSTGRES", "REDIS", "MONGO", "ES"]},
        "precision": "category", "context_required": False,
        "notes": "外联到 EXT 高端口/数据库端口，正常工作站基本不会出现"
    },
    {
        "id": "FW-002", "type": "pattern", "technique": "T1071",
        "name": "C2信标-非标准端口外联", "severity": 5, "action": "DROP",
        "match": {"proto": "TCP", "dir": "OUT", "dstnet": "EXT", "portcls_in": ["HIGHPORT"]},
        "precision": "category", "context_required": True,
        "notes": "外联 EXT 高端口，需配合 DT 周期性确认（context_required=true）"
    },
    {
        "id": "FW-003", "type": "pattern", "technique": "T1041",
        "name": "数据外泄-大流量外联", "severity": 5, "action": "DROP",
        "match": {"dir": "OUT", "dstnet": "EXT", "sizecls_in": ["XL"]},
        "precision": "category", "context_required": True,
        "notes": "外联 EXT 超大流量（>100KB/流），需配合 DT 确认非正常下载"
    },
    {
        "id": "FW-004", "type": "pattern", "technique": "T1021",
        "name": "横向移动-SMB外联", "severity": 4, "action": "DROP",
        "match": {"proto": "TCP", "dir": "OUT", "dstnet": "LAN", "portcls_in": ["SMB", "RDP"]},
        "precision": "category", "context_required": False,
        "notes": "本机主动外联内部 SMB/RDP，工作站不常见"
    },
    {
        "id": "FW-005", "type": "pattern", "technique": "T1572",
        "name": "DNS隧道-异常大DNS", "severity": 3, "action": "THROTTLE",
        "match": {"proto": "UDP", "portcls_in": ["DNS"], "sizecls_in": ["M", "L", "XL"]},
        "precision": "category", "context_required": False,
        "notes": "DNS 响应超过 200 字节，疑似 DNS 隧道"
    },
    {
        "id": "FW-006", "type": "pattern", "technique": "T1190",
        "name": "入站高端口连接", "severity": 3, "action": "ALERT",
        "match": {"dir": "IN", "dstnet": "LAN", "portcls_in": ["HIGHPORT"]},
        "precision": "category", "context_required": False,
        "notes": "入站连接到本机高端口，可能是反弹 shell"
    },
]


def init_default_db(path=None):
    """初始化默认模式库"""
    db = FwPatternDB(path)
    if db.entries:
        return db
    for p in DEFAULT_PATTERNS:
        db.add(p)
    print(f"已初始化 {len(DEFAULT_PATTERNS)} 条默认模式 -> {db.path}")
    return db


if __name__ == "__main__":
    db = init_default_db()
    print(f"模式库: {db.stats()}")
    for e in db.entries:
        print(f"  [{e['id']}] sev={e['severity']} {e['action']:<8} {e['name']}")
