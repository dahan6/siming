#!/usr/bin/env python3
"""司命检测引擎 - 宿主机 procfs 行为采集器

通过只读 /proc 文件系统采集进程行为事件，解析为司命 8-token 格式。
每秒扫描一次 /proc/[pid]/，捕获：
  - 新进程创建 (cmdline + stat 的 PPID)
  - 进程执行 (exe 路径)
  - 网络连接 (/proc/net/tcp + ss)

用法:
  采集良性: collect_host_procfs.py --duration 300 --out data/host_real_benign.jsonl
  采集攻击: collect_host_procfs.py --duration 60 --out data/host_real_attack.jsonl --attack
"""
import json
import os
import re
import sys
import time
import argparse
import subprocess
from datetime import datetime

# ---------------------------------------------------------------------------
# 离散化函数 — 直接复用 parse_events.py 的逻辑，保持 token 格式一致
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_events import argv_skeleton, pathclass_token, dt_bucket

BASE64ISH = re.compile(r"^[A-Za-z0-9+/=]{20,}$")
HAS_URL = re.compile(r"https?://")
HAS_IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

# /proc 读取工具 -------------------------------------------------------------


def read_file_safe(path):
    """安全读取 /proc 下的小文件，失败返回 None。"""
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except (OSError, IOError):
        return None


def parse_cmdline(pid):
    """解析 /proc/[pid]/cmdline → argv list。"""
    raw = read_file_safe(f"/proc/{pid}/cmdline")
    if not raw:
        return []
    # cmdline 用 \0 分隔
    parts = raw.split("\0")
    # 去掉末尾空元素
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def parse_stat(pid):
    """解析 /proc/[pid]/stat → (comm, ppid, uid)。"""
    raw = read_file_safe(f"/proc/{pid}/stat")
    if not raw:
        return None, None, None
    try:
        # comm 在括号里，可能包含空格
        rparen = raw.rfind(")")
        comm = raw[:rparen].split("(", 1)[1]
        fields = raw[rparen + 2:].split()
        ppid = int(fields[1])
        return comm, ppid, None
    except (IndexError, ValueError):
        return None, None, None


def get_uid(pid):
    """读取 /proc/[pid]/status 中的 Uid。"""
    raw = read_file_safe(f"/proc/{pid}/status")
    if not raw:
        return "?"
    for line in raw.splitlines():
        if line.startswith("Uid:"):
            try:
                return line.split()[1]  # real uid
            except (IndexError, ValueError):
                return "?"
    return "?"


def get_exe(pid):
    """读取 /proc/[pid]/exe 的链接目标。"""
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return ""


def get_parent_comm(ppid):
    """通过 PPID 获取父进程名。"""
    if ppid <= 0:
        return "?"
    comm, _, _ = parse_stat(ppid)
    return comm or "?"


def proc_name_from_comm(pid):
    """从 /proc/[pid]/stat 读取 comm。"""
    comm, _, _ = parse_stat(pid)
    return comm or "?"


# ---------------------------------------------------------------------------
# 网络连接采集
# ---------------------------------------------------------------------------


def parse_hex_ip(hex_str):
    """把 /proc/net/tcp 中的 hex IP:port 转成可读格式。"""
    if ":" not in hex_str:
        return None
    ip_hex, port_hex = hex_str.split(":")
    try:
        port = int(port_hex, 16)
        # IP 是小端 hex
        ip_int = int(ip_hex, 16)
        a = ip_int & 0xFF
        b = (ip_int >> 8) & 0xFF
        c = (ip_int >> 16) & 0xFF
        d = (ip_int >> 24) & 0xFF
        return f"{a}.{b}.{c}.{d}:{port}"
    except ValueError:
        return None


def collect_tcp_connections():
    """解析 /proc/net/tcp，返回 ESTABLISHED 连接列表 (remote_addr)。"""
    conns = []
    for proto_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        raw = read_file_safe(proto_file)
        if not raw:
            continue
        for line in raw.splitlines()[1:]:  # 跳过表头
            fields = line.split()
            if len(fields) < 4:
                continue
            # field[1]=local, field[2]=remote, field[3]=state
            state = fields[3]
            # 01 = ESTABLISHED
            if state != "01":
                continue
            remote = parse_hex_ip(fields[2])
            if remote:
                conns.append(remote)
    return conns


def dst_token_from_addr(remote_addr):
    """从 remote address 生成 DST token。"""
    m = HAS_IP.search(remote_addr)
    if not m:
        return "DST:OTHER"
    ip = m.group(0)
    port = 0
    pm = re.search(r":(\d+)$", remote_addr)
    if pm:
        port = int(pm.group(1))
    pc = "WELL" if port in (22, 53, 80, 443, 514, 123) else "HIGH"
    # 判断是否本地网段
    if ip.startswith("127.") or ip.startswith("10.") or ip.startswith("192.168."):
        return f"DST:LAN:{pc}"
    return f"DST:EXT:{pc}"


# ---------------------------------------------------------------------------
# 核心：进程扫描 & 事件生成
# ---------------------------------------------------------------------------


class ProcfsCollector:
    def __init__(self):
        self.prev_pids = set()      # 上一次扫描到的 pid 集合
        self.prev_exe = {}          # pid -> exe path
        self.prev_conns = set()     # 上一次的连接集合
        self.events = []
        self.prev_ts = None

    def _emit(self, tokens_list):
        """生成一条 8-token 事件。"""
        now = time.time()
        if self.prev_ts is None:
            delta_ms = 0
        else:
            delta_ms = int((now - self.prev_ts) * 1000)
        self.prev_ts = now

        ts_iso = datetime.now().astimezone().isoformat()
        for tokens in tokens_list:
            # 确保 8 token
            if len(tokens) == 7:
                # 7-token 格式：在 DST 后插入 PC:NONE
                dst_idx = next(
                    (i for i, t in enumerate(tokens) if t.startswith("DST:")), 5)
                tokens.insert(dst_idx + 1, "PC:NONE")
            event = {
                "ts": ts_iso,
                "host": os.uname().nodename,
                "tokens": tokens,
            }
            self.events.append(event)

    def _make_exec_tokens(self, argv, proc_name, ppid, uid):
        """构建 EXEC 事件的 8-token。"""
        parent_comm = get_parent_comm(ppid) if ppid else "?"
        skel = argv_skeleton(argv) if argv else "ARGV0"
        pc = pathclass_token(argv) if argv else "PC:NONE"
        return [
            "ET:EXEC",
            f"PROC:{proc_name}",
            skel,
            f"PARENT:{parent_comm}",
            f"UID:{uid}",
            "DST:NONE",
            pc,
            "DT0",  # 会在 _emit 中被覆盖
        ]

    def _make_conn_tokens(self, remote_addr, proc_name, uid):
        """构建 CONN 事件的 8-token。"""
        dst = dst_token_from_addr(remote_addr)
        return [
            "ET:CONN",
            f"PROC:{proc_name}",
            "ARGV0",
            "PARENT:?",
            f"UID:{uid}",
            dst,
            "PC:NONE",
            "DT0",
        ]

    def scan_once(self):
        """扫描一次 /proc，返回本轮产生的事件列表。"""
        batch = []
        try:
            current_pids = set(int(d) for d in os.listdir("/proc") if d.isdigit())
        except OSError:
            return batch

        # 1. 检测新进程
        new_pids = current_pids - self.prev_pids
        for pid in sorted(new_pids):
            argv = parse_cmdline(pid)
            comm, ppid, _ = parse_stat(pid)
            if not comm:
                continue
            uid = get_uid(pid)
            exe = get_exe(pid)

            # 用 cmdline[0] 的 basename 作为 proc_name，回退到 comm
            if argv:
                proc_name = os.path.basename(argv[0])[:15]
            else:
                proc_name = comm[:15]

            tokens = self._make_exec_tokens(argv, proc_name, ppid, uid)
            # 更新 exe 记录
            if exe:
                self.prev_exe[pid] = exe
            batch.append(tokens)

        # 2. 检测 exe 变化（进程重新 exec）
        for pid in sorted(current_pids & self.prev_pids):
            exe = get_exe(pid)
            if exe and pid in self.prev_exe and exe != self.prev_exe[pid]:
                argv = parse_cmdline(pid)
                comm, ppid, _ = parse_stat(pid)
                if not comm:
                    continue
                uid = get_uid(pid)
                proc_name = os.path.basename(argv[0])[:15] if argv else comm[:15]
                tokens = self._make_exec_tokens(argv, proc_name, ppid, uid)
                batch.append(tokens)
            if exe:
                self.prev_exe[pid] = exe

        # 3. 检测新网络连接
        current_conns = set(collect_tcp_connections())
        new_conns = current_conns - self.prev_conns
        if new_conns:
            # 找一个可能产生连接的进程
            # 简化：用最常见的网络进程
            net_proc = "unknown"
            for pid in sorted(current_pids):
                comm = proc_name_from_comm(pid)
                if comm in ("sshd", "python3", "node", "curl", "wget", "chrome",
                            "firefox", "systemd-resolve", "dnsmasq"):
                    net_proc = comm
                    break
            for addr in sorted(new_conns)[:5]:  # 限制每轮最多 5 条
                tokens = self._make_conn_tokens(addr, net_proc, "1000")
                batch.append(tokens)

        self.prev_pids = current_pids
        self.prev_conns = current_conns
        return batch

    def init_baseline(self):
        """建立基线快照，后续扫描只报告增量。"""
        try:
            self.prev_pids = set(int(d) for d in os.listdir("/proc") if d.isdigit())
        except OSError:
            self.prev_pids = set()
        for pid in self.prev_pids:
            exe = get_exe(pid)
            if exe:
                self.prev_exe[pid] = exe
        self.prev_conns = set(collect_tcp_connections())

    def run(self, duration_sec, out_path, label=None):
        """运行采集器指定时长。"""
        print(f"开始采集 {duration_sec}s，输出 → {out_path}")
        self.init_baseline()
        print(f"基线: {len(self.prev_pids)} 进程, {len(self.prev_conns)} 连接")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        n_events = 0
        t_start = time.time()

        with open(out_path, "w") as fout:
            while time.time() - t_start < duration_sec:
                batch = self.scan_once()
                if batch:
                    self._emit(batch)
                    for ev in self.events:
                        if label:
                            ev["label"] = label
                        fout.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_events += len(self.events)
                    self.events.clear()
                elapsed = int(time.time() - t_start)
                if elapsed % 30 == 0 and elapsed > 0:
                    print(f"  [{elapsed}s] 已采集 {n_events} 事件")
                time.sleep(1.0)

        print(f"采集完成: {n_events} 事件 → {out_path}")
        return n_events


# ---------------------------------------------------------------------------
# 攻击模拟
# ---------------------------------------------------------------------------

ATTACK_COMMANDS = [
    ["crontab", "-l"],
    ["ss", "-tlnp"],
    ["cat", "/etc/passwd"],
    ["find", "/", "-perm", "-4000", "-type", "f"],
    ["ps", "aux"],
]


def run_attack_simulation(collector, out_path):
    """在采集过程中执行攻击命令，标记为 attack。

    使用高频扫描（100ms）确保捕获短生命周期进程的完整 cmdline。
    """
    print("\n=== 攻击模拟开始 ===")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # 建立基线，避免首次扫描报告所有已存在进程
    collector.init_baseline()
    print(f"基线: {len(collector.prev_pids)} 进程, {len(collector.prev_conns)} 连接")

    n_events = 0
    with open(out_path, "w") as fout:
        for cmd in ATTACK_COMMANDS:
            print(f"\n[*] 执行: {' '.join(cmd)}")
            self_events = []
            collector.events = []

            # 高频扫描线程：在命令执行期间每 50ms 扫描一次
            import threading
            scanning = [True]

            def fast_scan():
                while scanning[0]:
                    batch = collector.scan_once()
                    if batch:
                        collector._emit(batch)
                        self_events.extend(collector.events[:])
                        collector.events.clear()
                    time.sleep(0.05)

            t = threading.Thread(target=fast_scan, daemon=True)
            t.start()

            # 执行命令
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                scanning[0] = False
                t.join(timeout=1)
                print(f"  命令不存在: {cmd[0]}")
                continue

            # 等命令结束
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

            # 停止扫描线程
            scanning[0] = False
            t.join(timeout=2)

            # 标记并写入
            for ev in self_events:
                ev["label"] = "attack"
                ev["attack_cmd"] = " ".join(cmd)
                fout.write(json.dumps(ev, ensure_ascii=False) + "\n")
            fout.flush()
            n_events += len(self_events)
            print(f"  捕获 {len(self_events)} 事件")

    print(f"\n攻击采集完成: {n_events} 事件 → {out_path}")
    return n_events


def main():
    ap = argparse.ArgumentParser(description="宿主机 procfs 行为采集器")
    ap.add_argument("--duration", type=int, default=300, help="采集时长(秒)")
    ap.add_argument("--out", default="data/host_real_benign.jsonl", help="输出文件")
    ap.add_argument("--attack", action="store_true", help="运行攻击模拟模式")
    args = ap.parse_args()

    det_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(det_dir, args.out)

    collector = ProcfsCollector()

    if args.attack:
        run_attack_simulation(collector, out_path)
    else:
        collector.run(args.duration, out_path)


if __name__ == "__main__":
    main()
