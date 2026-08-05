#!/usr/bin/env python3
"""自适应事件检测器：检测隐翅虫 v4 的自适应行为模式

检测目标（从 v4 文档 + 21 轮实验提取）：
1. rebirth 重生序列：cp→chmod→setsid→rm（优雅退休）
2. 伪装名池轮换：node/python3/go-build/cargo-watch 等开发工具名
3. 侦察链均匀轮换：ss/ip/ps/ls/find/free/df/uptime 均匀分布
4. C2 轮询：伪装进程 + EXT:HIGH 外联
5. SUID 提权：python3 UID:0 无 sudo 链
6. sleep 步进：sleep→动作→sleep 交替（CV<0.5）

设计：滑窗事件缓冲 + 规则匹配，轻量无 ML 依赖。
"""
import json
import os
import sys
from collections import Counter, deque
from datetime import datetime

DET = os.path.dirname(os.path.abspath(__file__))

# 隐翅虫 v4 伪装名池
DISGUISE_POOL = {
    "node", "python3", "go-build", "cargo-watch", "dotnet", "ruby",
    "java", "graalvm", "docker-credential", "kubectl", "terraform",
    "ansible-runner",
}

# 侦察命令集
RECON_CMDS = {
    "ss", "ip", "ifconfig", "netstat", "ps", "ls", "find", "grep",
    "env", "id", "whoami", "uname", "hostname", "uptime", "free",
    "df", "wc", "du", "journalctl", "pgrep", "readlink",
}

# rebirth 序列模式（优雅退休版）
REBIRTH_SEQ = ["cp", "chmod", "setsid", "rm"]


class AdaptiveDetector:
    """滑窗自适应行为检测器"""

    def __init__(self, window_size=200, cooldown=50):
        self.window_size = window_size
        self.cooldown = cooldown  # 同类型告警冷却事件数
        self.events = deque(maxlen=window_size)
        # 追踪
        self.proc_history = deque(maxlen=500)
        self.uid_transitions = []  # (ts, proc, uid_from, uid_to)
        self.alerts = []
        self._last_alert_type = {}  # type -> event_index
        self._event_idx = 0

    def update(self, tokens_8, ts=None):
        """喂一条 8-token 事件，返回告警列表（可能为空）

        Args:
            tokens_8: ["ET:EXEC", "PROC:bash", "ARGV:N1P", "PARENT:sh",
                       "UID:1000", "DST:NONE", "PC:NONE", "DT3"]
            ts: 时间戳字符串（可选）

        Returns:
            list[dict]: 告警列表
        """
        ev = self._parse(tokens_8, ts)
        self.events.append(ev)
        self.proc_history.append(ev.get("PROC", "?"))
        self._event_idx += 1

        raw_alerts = []

        # 检测 1: rebirth 序列
        a = self._detect_rebirth()
        if a:
            raw_alerts.append(a)

        # 检测 2: 伪装名池进程 + 外联
        a = self._detect_disguise_c2(ev)
        if a:
            raw_alerts.append(a)

        # 检测 3: 侦察链均匀轮换
        a = self._detect_recon_uniform()
        if a:
            raw_alerts.append(a)

        # 检测 4: SUID 提权
        a = self._detect_suid_privesc(ev)
        if a:
            raw_alerts.append(a)

        # 检测 5: sleep 步进模式
        a = self._detect_sleep_stepping()
        if a:
            raw_alerts.append(a)

        # 冷却去重：同类型告警在 cooldown 事件内不重复
        alerts = []
        for a in raw_alerts:
            atype = a["type"]
            # P5 不冷却（高危即报）
            if a["severity"] >= 5:
                alerts.append(a)
                continue
            last = self._last_alert_type.get(atype, -self.cooldown)
            if self._event_idx - last >= self.cooldown:
                alerts.append(a)
                self._last_alert_type[atype] = self._event_idx

        self.alerts.extend(alerts)
        return alerts

    def _parse(self, tokens, ts):
        ev = {"ts": ts, "raw": tokens}
        for t in tokens:
            if t.startswith("DT") and ":" not in t:
                ev["DT"] = t[2:]
            elif ":" in t:
                k, v = t.split(":", 1)
                ev[k] = v
        return ev

    def _detect_rebirth(self):
        """检测 cp→chmod→setsid→rm 序列（变体容忍版）

        升级：允许中间插入 1-2 个噪声命令（sleep/dash/env 等），
        用滑动窗口模糊匹配替代精确匹配。
        """
        if len(self.events) < 4:
            return None
        recent = list(self.events)[-15:]  # 扩大窗口到 15
        procs = [e.get("PROC", "") for e in recent]

        # 噪声命令（允许插入的无关命令）
        NOISE = {"sleep", "dash", "env", "date", "echo", "pwd", "whoami",
                 "hostname", "uname", "sh", "bash"}

        # 模糊匹配：在 15 事件窗口里找 4 个关键命令（顺序不变，中间可有噪声）
        key_seq = ["cp", "chmod", "setsid", "rm"]
        key_idx = 0
        matched_positions = []
        tmp_found = False

        for i, proc in enumerate(procs):
            if proc == key_seq[key_idx]:
                matched_positions.append(i)
                pc = recent[i].get("PC", "")
                if "TMP" in pc:
                    tmp_found = True
                key_idx += 1
                if key_idx == len(key_seq):
                    # 全部匹配
                    # 检查噪声数量：匹配位置之间允许最多 2 个噪声
                    gaps = [matched_positions[j+1] - matched_positions[j] - 1
                            for j in range(len(matched_positions)-1)]
                    max_gap = max(gaps) if gaps else 0
                    if max_gap <= 3 and tmp_found:
                        return {
                            "type": "MORPH_TRANSFORM",
                            "severity": 5,
                            "technique": "T1070.004",
                            "detail": f"Transformation sequence: {'→'.join(key_seq)} (gaps={gaps})",
                            "ts": recent[matched_positions[0]].get("ts"),
                        }
                    break
            elif proc in NOISE:
                continue  # 噪声命令，跳过
            elif key_idx > 0 and proc not in key_seq[key_idx:]:
                # 遇到非关键非噪声命令，重置
                key_idx = 0
                matched_positions = []

        return None

    def _detect_disguise_c2(self, ev):
        """检测伪装名池进程 + EXT:HIGH 外联"""
        proc = ev.get("PROC", "")
        dst = ev.get("DST", "NONE")
        et = ev.get("ET", "")
        if et == "CONN" and proc in DISGUISE_POOL and "EXT" in dst:
            return {
                "type": "DISGUISE_C2",
                "severity": 5,
                "technique": "T1071.001",
                "detail": f"伪装进程 {proc} 外联 {dst}",
                "ts": ev.get("ts"),
            }
        return None

    def _detect_sequence_fuzzy(self):
        """通用模糊序列匹配（自动模式发现）

        从 patterns.jsonl 中加载 sequence 条目，
        用模糊匹配（允许噪声插入）检测攻击序列。
        """
        if not hasattr(self, '_seq_patterns'):
            self._load_seq_patterns()
        if not self._seq_patterns or len(self.events) < 3:
            return None

        recent = list(self.events)[-20:]
        recent_procs = [e.get("PROC", "") for e in recent]

        NOISE = {"sleep", "dash", "env", "date", "echo", "pwd", "whoami",
                 "hostname", "uname", "sh", "bash"}

        for pattern in self._seq_patterns:
            key_procs = pattern["key_procs"]
            technique = pattern.get("technique", "?")
            severity = pattern.get("severity", 4)

            key_idx = 0
            matched = 0
            for proc in recent_procs:
                if proc == key_procs[key_idx]:
                    matched += 1
                    key_idx += 1
                    if matched == len(key_procs):
                        if severity >= 5:
                            return {
                                "type": f"PATTERN_{technique}",
                                "severity": severity,
                                "technique": technique,
                                "detail": f"Fuzzy match: {'→'.join(key_procs)}",
                                "ts": recent[-1].get("ts"),
                            }
                elif proc in NOISE:
                    continue
                elif key_idx > 0 and proc not in key_procs[key_idx:]:
                    key_idx = 0
                    matched = 0

        return None

    def _load_seq_patterns(self):
        """从 patterns.jsonl 加载序列模式"""
        import os as _os
        self._seq_patterns = []
        patterns_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "patterns.jsonl")
        if not _os.path.exists(patterns_path):
            return

        for line in open(patterns_path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                e = json.loads(line)
                if e.get("type") == "sequence" and e.get("review") == "approved":
                    # 提取 PROC token 序列
                    seq = e.get("sequence", [])
                    key_procs = []
                    for t in seq:
                        if t.startswith("PROC:"):
                            key_procs.append(t.split(":", 1)[1])
                    if len(key_procs) >= 2:
                        self._seq_patterns.append({
                            "id": e.get("id", "?"),
                            "technique": e.get("technique", "?"),
                            "severity": e.get("severity", 4),
                            "key_procs": key_procs,
                        })
            except:
                continue
    def _detect_recon_uniform(self):
        """检测侦察命令均匀轮换（滑窗内 ≥6 种不同侦察命令，分布均匀）"""
        if len(self.events) < 30:
            return None
        recent = list(self.events)[-50:]
        recon_procs = [e.get("PROC", "") for e in recent
                       if e.get("PROC", "") in RECON_CMDS
                       and e.get("PARENT", "") in ("bash", "sh")]
        if len(recon_procs) < 12:
            return None
        dist = Counter(recon_procs)
        n_unique = len(dist)
        if n_unique < 6:
            return None
        # 均匀性：最大/最小比 < 3（正常管理偏斜 > 5）
        counts = list(dist.values())
        ratio = max(counts) / max(min(counts), 1)
        if ratio < 3.0:
            return {
                "type": "RECON_UNIFORM",
                "severity": 3,
                "technique": "T1082",
                "detail": f"均匀侦察轮换: {n_unique} 种命令, "
                          f"max/min={ratio:.1f}, 分布={dict(dist.most_common(8))}",
                "ts": recent[-1].get("ts"),
            }
        return None

    def _detect_suid_privesc(self, ev):
        """检测 SUID python3 提权（UID:0 + python3 + 无 sudo 父进程）"""
        proc = ev.get("PROC", "")
        uid = ev.get("UID", "")
        parent = ev.get("PARENT", "")
        et = ev.get("ET", "")
        if (et == "EXEC" and proc in ("python3", "python")
                and uid == "0" and parent in ("bash", "sh")):
            # 检查近期是否有 sudo 进程（合法提权路径）
            # sudo 的 PARENT:bash，所以看近 10 条事件的 PROC 链
            recent_procs = [e.get("PROC", "") for e in list(self.events)[-10:]]
            # 排除当前事件自身
            recent_procs = [p for p in recent_procs if p != proc]
            if "sudo" not in recent_procs:
                return {
                    "type": "SUID_PRIVESC",
                    "severity": 5,
                    "technique": "T1548.001",
                    "detail": f"SUID {proc} 提权到 UID:0, 父进程={parent}",
                    "ts": ev.get("ts"),
                }
        return None

    def _detect_sleep_stepping(self):
        """检测 sleep→动作→sleep 交替步进"""
        if len(self.events) < 20:
            return None
        recent = list(self.events)[-20:]
        procs = [e.get("PROC", "") for e in recent]
        # 计算 sleep 占比和交替模式
        n_sleep = sum(1 for p in procs if p == "sleep")
        if n_sleep < 8:  # 20 事件中 ≥8 个 sleep
            return None
        # 检查交替模式：sleep 后面跟非 sleep
        alternations = 0
        for i in range(len(procs) - 1):
            if procs[i] == "sleep" and procs[i+1] != "sleep":
                alternations += 1
        if alternations >= 6:
            return {
                "type": "SLEEP_STEPPING",
                "severity": 3,
                "technique": "TA0005",
                "detail": f"sleep 步进: {n_sleep}/20 sleep, "
                          f"{alternations} 次交替",
                "ts": recent[-1].get("ts"),
            }
        return None

    def summary(self):
        """输出检测摘要"""
        if not self.alerts:
            return "无告警"
        type_cnt = Counter(a["type"] for a in self.alerts)
        lines = [f"告警 {len(self.alerts)} 条:"]
        for t, c in type_cnt.most_common():
            sev = max(a["severity"] for a in self.alerts if a["type"] == t)
            lines.append(f"  {t}: {c} 条 (severity={sev})")
        return "\n".join(lines)


def test_on_telemetry(path, hour_filter=None, n_max=50000):
    """在遥测数据上测试检测器"""
    det = AdaptiveDetector(window_size=200)
    n_events = 0
    n_alerts = 0

    for line in open(path):
        e = json.loads(line)
        if hour_filter and e["ts"][11:13] not in hour_filter:
            continue
        if n_events >= n_max:
            break

        # 7-token → 8-token（补 PC）
        toks = e["tokens"]
        toks_8 = list(toks)
        # 在 DST 后插入 PC:NONE
        dst_idx = next((i for i, t in enumerate(toks) if t.startswith("DST:")), 5)
        toks_8.insert(dst_idx + 1, "PC:NONE")

        alerts = det.update(toks_8, ts=e["ts"])
        n_events += 1
        n_alerts += len(alerts)

        if alerts:
            for a in alerts:
                if a["severity"] >= 5:
                    print(f"  [P{a['severity']}] {a['type']}: {a['detail']} "
                          f"({a.get('ts', '?')})")

    print(f"\n事件: {n_events}, 告警: {n_alerts}")
    print(det.summary())
    return det


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        hours = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
        test_on_telemetry(path, hour_filter=hours)
    else:
        print("用法: adaptive_detector.py <events.jsonl> [hours]")
        print("示例: adaptive_detector.py regime_events.jsonl 13,14")
