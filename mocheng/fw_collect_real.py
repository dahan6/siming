#!/usr/bin/env python3
"""墨城防火墙: 宿主机真实流量采集器

从宿主机实时采集网络连接，解析为墨城 7-token 格式。

数据源策略（--method auto，默认混合模式）:
  1. tcpdump 抓包（包级，有真实字节数）+ conntrack -L 快照（连接级，有连接多样性）
  2. conntrack -E -o timestamp -e NEW（sudo 实时事件流）
  3. conntrack -L 周期快照（sudo，次选）
  4. ss -tnu（无需 root，兜底）

输出:
  data/host_real_flows.jsonl   — 原始流事件（与 fw_collect.py 格式一致）
  data/host_real_tokens.jsonl  — 7-token 离散化结果

用法:
  fw_collect_real.py [--duration 120] [--interval 3] [--method auto] [--out-dir data/]
  SUDO_PWD=<password> fw_collect_real.py --duration 120 --method auto
"""
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fw_paths import DATA_DIR
from fw_tokens import flow_to_tokens, size_class, net_class
from fw_collect import (
    detect_local_ips, infer_direction,
    CT_LIST_RE, CT_EVENT_RE, CT_BYTES_RE, SS_RE,
    make_flow,
)

SUDO_PWD = os.environ.get("SUDO_PWD", "")
HOST_FLOWS = os.path.join(DATA_DIR, "host_real_flows.jsonl")
HOST_TOKENS = os.path.join(DATA_DIR, "host_real_tokens.jsonl")


# ── sudo 辅助 ──

def sudo_cmd(cmd_args):
    """构造 sudo 命令（通过 echo | sudo -S 传密码）"""
    if SUDO_PWD:
        return f"echo {shlex.quote(SUDO_PWD)} | sudo -S " + " ".join(
            shlex.quote(a) for a in cmd_args)
    else:
        return "sudo -n " + " ".join(shlex.quote(a) for a in cmd_args)


def run_sudo(cmd_args, timeout=10):
    """执行 sudo 命令，返回 (stdout, returncode)"""
    cmd_str = sudo_cmd(cmd_args)
    r = subprocess.run(cmd_str, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    return r.stdout, r.returncode


# ── tcpdump 抓包 + 解析（有真实字节数）──

PKT_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"(\S+)\s+(\S+)\s+(IP6?)\s+"
    r"(\d+\.\d+\.\d+\.\d+)\.(\d+)\s+>\s+"
    r"(\d+\.\d+\.\d+\.\d+)\.(\d+):.*?length\s+(\d+)"
)

# conntrack -L 无字节时按端口估算
EST_BYTES = {
    ("udp", 53): 128,    # DNS
    ("udp", 123): 48,    # NTP
    ("udp", 137): 80,    # NetBIOS
    ("tcp", 22): 5000,   # SSH
    ("tcp", 443): 50000, # HTTPS
    ("tcp", 80): 10000,  # HTTP
    ("tcp", 445): 3000,  # SMB
}


def estimate_bytes(proto, port):
    return EST_BYTES.get((proto, port), 200)


def capture_pcap(duration, pcap_path):
    """用 tcpdump 抓包 duration 秒，保存到 pcap_path"""
    max_pkts = 5000
    cmd_str = sudo_cmd([
        "tcpdump", "-i", "any", "-c", str(max_pkts),
        "-w", pcap_path
    ])
    try:
        proc = subprocess.Popen(
            cmd_str, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=os.setsid)
    except Exception as e:
        print(f"  [WARN] tcpdump 启动失败: {e}")
        return False

    start = time.time()
    try:
        while proc.poll() is None:
            if time.time() - start >= duration:
                break
            time.sleep(0.5)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
    return os.path.exists(pcap_path)


def parse_pcap(pcap_path, local_ips):
    """解析 pcap 文件（通过 tcpdump 文本输出），提取包级流，按 5 元组聚合"""
    try:
        result = subprocess.run(
            ["tcpdump", "-nn", "-tttt", "-r", pcap_path],
            capture_output=True, text=True, timeout=30)
    except Exception as e:
        print(f"  [WARN] tcpdump -r 解析失败: {e}")
        return []

    hostname = socket.gethostname()
    seen_iface_pkt = set()  # 去重 bridge 多接口重复包
    flows = {}

    for line in result.stdout.strip().split("\n"):
        m = PKT_RE.search(line)
        if not m:
            continue
        ts_str = m.group(1)
        src_ip = m.group(5)
        src_port = int(m.group(6))
        dst_ip = m.group(7)
        dst_port = int(m.group(8))
        length = int(m.group(9))

        proto = "tcp"
        if "UDP" in line.split(":")[0]:
            proto = "udp"

        # 去重：同一时间戳+5元组+长度的包只计一次
        pkt_key = (ts_str, src_ip, src_port, dst_ip, dst_port, length)
        if pkt_key in seen_iface_pkt:
            continue
        seen_iface_pkt.add(pkt_key)

        dir_raw = m.group(3).lower()
        direction = infer_direction(src_ip, dst_ip, local_ips)
        if dir_raw in ("in", "out"):
            direction = dir_raw

        ts_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        ts_epoch = ts_dt.timestamp()

        key = (proto, src_ip, src_port, dst_ip, dst_port)
        if key in flows:
            flows[key]["bytes"] += length
            flows[key]["pkt_count"] += 1
        else:
            flows[key] = {
                "bytes": length, "pkt_count": 1,
                "ts_epoch": ts_epoch, "proto": proto,
                "src_ip": src_ip, "src_port": src_port,
                "dst_ip": dst_ip, "dst_port": dst_port,
                "dir": direction,
            }

    result_list = []
    for fl in sorted(flows.values(), key=lambda f: f["ts_epoch"]):
        result_list.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(fl["ts_epoch"])),
            "ts_epoch": fl["ts_epoch"], "ts_sort": fl["ts_epoch"],
            "proto": fl["proto"], "dir": fl["dir"],
            "src_ip": fl["src_ip"], "src_port": fl["src_port"],
            "dst_ip": fl["dst_ip"], "dst_port": fl["dst_port"],
            "bytes": fl["bytes"], "pkt_count": fl["pkt_count"],
            "action": "accept", "host": hostname, "state": "",
        })
    return result_list


def collect_conntrack_snapshot_once(local_ips):
    """单次 conntrack -L 快照，返回去重后的流列表"""
    stdout, rc = run_sudo(["conntrack", "-L"], timeout=10)
    if rc != 0:
        return []
    hostname = socket.gethostname()
    flows = []
    ct_seen = set()
    ts_base = time.time()
    for line in stdout.split("\n"):
        m = CT_LIST_RE.search(line)
        if not m:
            continue
        proto = m.group("proto")
        src_ip = m.group("src_ip")
        dst_ip = m.group("dst_ip")
        src_port = m.group("src_port")
        dst_port = m.group("dst_port")
        rev_key = (proto, dst_ip, dst_port, src_ip, src_port)
        if rev_key in ct_seen:
            continue
        key = (proto, src_ip, src_port, dst_ip, dst_port)
        ct_seen.add(key)

        bm = CT_BYTES_RE.search(line)
        if bm:
            n_bytes = int(bm.group("bytes"))
        else:
            n_bytes = estimate_bytes(proto, int(dst_port))

        direction = infer_direction(src_ip, dst_ip, local_ips)
        ts_base += 0.01
        flows.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts_base)),
            "ts_epoch": ts_base, "ts_sort": ts_base,
            "proto": proto, "dir": direction,
            "src_ip": src_ip, "src_port": int(src_port),
            "dst_ip": dst_ip, "dst_port": int(dst_port),
            "bytes": n_bytes, "action": "accept",
            "host": hostname, "state": m.group("state") or "",
        })
    return flows


def collect_hybrid(duration, pcap_path, local_ips, interval=5):
    """混合模式: tcpdump 后台抓包 + 周期 conntrack -L 快照，最后合并。

    tcpdump 提供真实字节数；conntrack 周期快照提供连接多样性。
    """
    hostname = socket.gethostname()

    # 1. 启动 tcpdump 后台抓包
    max_pkts = 10000
    cmd_str = sudo_cmd(["tcpdump", "-i", "any", "-c", str(max_pkts),
                        "-w", pcap_path])
    print(f"  [tcpdump] 后台抓包 {duration}s (max {max_pkts} pkts) -> {pcap_path}")
    try:
        proc = subprocess.Popen(
            cmd_str, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=os.setsid)
    except Exception as e:
        print(f"  [WARN] tcpdump 启动失败: {e}")
        proc = None

    # 2. 周期 conntrack -L 快照（tcpdump 运行期间）
    ct_flows_map = {}  # key -> flow dict
    rounds = 0
    start = time.time()
    while time.time() - start < duration:
        stdout, rc = run_sudo(["conntrack", "-L"], timeout=10)
        n_new = 0
        ts_base = time.time()
        round_seen = set()
        for line in stdout.split("\n"):
            m = CT_LIST_RE.search(line)
            if not m:
                continue
            proto = m.group("proto")
            src_ip = m.group("src_ip")
            dst_ip = m.group("dst_ip")
            src_port = m.group("src_port")
            dst_port = m.group("dst_port")
            rev_key = (proto, dst_ip, dst_port, src_ip, src_port)
            if rev_key in round_seen:
                continue
            key = (proto, src_ip, src_port, dst_ip, dst_port)
            round_seen.add(key)
            if key in ct_flows_map:
                continue  # 已采集过的连接，跳过
            port = int(dst_port)
            bm = CT_BYTES_RE.search(line)
            if bm:
                n_bytes = int(bm.group("bytes"))
            else:
                n_bytes = estimate_bytes(proto, port)
            direction = infer_direction(src_ip, dst_ip, local_ips)
            ts_base += 0.01
            ct_flows_map[key] = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts_base)),
                "ts_epoch": ts_base, "ts_sort": ts_base,
                "proto": proto, "dir": direction,
                "src_ip": src_ip, "src_port": int(src_port),
                "dst_ip": dst_ip, "dst_port": port,
                "bytes": n_bytes, "pkt_count": 0,
                "action": "accept", "host": hostname,
                "state": m.group("state") or "",
            }
            n_new += 1
        rounds += 1
        elapsed = time.time() - start
        print(f"  [conntrack] 轮次 {rounds} [{elapsed:.0f}s]: 新增 {n_new}, 累计 {len(ct_flows_map)}")
        remaining = duration - (time.time() - start)
        if remaining > interval:
            time.sleep(interval)

    # 3. 停止 tcpdump
    if proc:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()

    # 4. 解析 pcap
    pcap_flows_map = {}  # key -> flow dict
    if os.path.exists(pcap_path):
        pcap_flows = parse_pcap(pcap_path, local_ips)
        print(f"  [tcpdump] 解析出 {len(pcap_flows)} 条包级流")
        for fl in pcap_flows:
            key = (fl["proto"], fl["src_ip"], fl["src_port"],
                   fl["dst_ip"], fl["dst_port"])
            pcap_flows_map[key] = fl

    # 5. 合并：pcap 字节数覆盖 conntrack 估算值
    merged = {}
    for key, fl in ct_flows_map.items():
        if key in pcap_flows_map:
            fl["bytes"] = pcap_flows_map[key]["bytes"]
            fl["pkt_count"] = pcap_flows_map[key].get("pkt_count", 0)
        merged[key] = fl
    # pcap 中有但 conntrack 没有的流也加入
    for key, fl in pcap_flows_map.items():
        if key not in merged:
            merged[key] = fl

    result = sorted(merged.values(), key=lambda f: f["ts_epoch"])
    print(f"  [合并] conntrack={len(ct_flows_map)}, pcap={len(pcap_flows_map)}, "
          f"合并去重={len(result)}")
    return result


# ── conntrack -E 实时事件流 ──

def collect_conntrack_events(duration, local_ips):
    """用 conntrack -E -e NEW 采集实时新建连接事件"""
    cmd_args = ["conntrack", "-E", "-o", "timestamp", "-e", "NEW"]
    cmd_str = sudo_cmd(cmd_args)
    try:
        proc = subprocess.Popen(
            cmd_str, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
            preexec_fn=os.setsid)
    except Exception as e:
        print(f"[WARN] conntrack -E 启动失败: {e}")
        return []

    flows = []
    start = time.time()
    try:
        for line in proc.stdout:
            if time.time() - start > duration:
                break
            m = CT_EVENT_RE.search(line)
            if not m:
                continue
            ts = float(m.group("ts"))
            bm = CT_BYTES_RE.search(line)
            flow = make_flow(
                ts, m.group("proto"),
                m.group("src_ip"), m.group("dst_ip"),
                m.group("src_port"), m.group("dst_port"),
                bm.group("bytes") if bm else 0,
                m.group("state") or "", local_ips)
            flows.append(flow)
            if len(flows) % 100 == 0:
                print(f"  [{time.time()-start:.0f}s] conntrack -E 累计 {len(flows)} 事件")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
    return flows


# ── conntrack -L 周期快照 ──

def collect_conntrack_snapshots(duration, interval, local_ips):
    """周期性 conntrack -L 快照，去重累积"""
    flows = []
    seen_keys = set()
    rounds = 0
    start = time.time()
    while time.time() - start < duration:
        stdout, rc = run_sudo(["conntrack", "-L"], timeout=10)
        if rc != 0:
            print(f"  [WARN] conntrack -L rc={rc}")
            time.sleep(interval)
            continue
        n_new = 0
        ts_now = time.time()
        for line in stdout.split("\n"):
            m = CT_LIST_RE.search(line)
            if not m:
                continue
            bm = CT_BYTES_RE.search(line)
            src_ip = m.group("src_ip")
            dst_ip = m.group("dst_ip")
            src_port = m.group("src_port")
            dst_port = m.group("dst_port")
            proto = m.group("proto")
            # conntrack -L 每条流有正反两行，只取正向（源端口 < 对端源端口 或源非本机）
            key = (proto, src_ip, src_port, dst_ip, dst_port)
            # 跳过反向行（assured 回包）
            if key in seen_keys:
                continue
            # 去重：只保留方向合理的行（src 或 dst 是本机）
            direction = infer_direction(src_ip, dst_ip, local_ips)
            if direction == "fwd":
                # 转发流量也保留，但去重：只取 src_port < dst_port 的那条
                if int(src_port) > int(dst_port):
                    continue
            seen_keys.add(key)
            flow = make_flow(
                ts_now, proto, src_ip, dst_ip,
                src_port, dst_port,
                bm.group("bytes") if bm else 0,
                m.group("state") or "", local_ips)
            # 给每条微调时间戳，保证排序
            ts_now += 0.0001
            flows.append(flow)
            n_new += 1
        rounds += 1
        elapsed = time.time() - start
        print(f"  [{elapsed:.0f}s] 轮次 {rounds}: 新增 {n_new}, 累计唯一 {len(flows)}")
        if time.time() - start < duration:
            time.sleep(interval)
    return flows


# ── ss 兜底 ──

def collect_ss_snapshots(duration, interval, local_ips):
    """用 ss -tnu 周期采集"""
    flows = []
    seen_keys = set()
    start = time.time()
    while time.time() - start < duration:
        try:
            r = subprocess.run(["ss", "-tnu"], capture_output=True,
                               text=True, timeout=5)
        except Exception:
            time.sleep(interval)
            continue
        n_new = 0
        ts_now = time.time()
        for line in r.stdout.split("\n")[1:]:
            m = SS_RE.search(line)
            if not m:
                continue
            local_ip = m.group("local_ip").strip("[]")
            peer_ip = m.group("peer_ip").strip("[]")
            proto = "tcp" if "tcp" in line.lower() else "udp"
            key = (proto, local_ip, m.group("local_port"),
                   peer_ip, m.group("peer_port"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            flow = make_flow(
                ts_now, proto, local_ip, peer_ip,
                m.group("local_port"), m.group("peer_port"),
                0, m.group("state"), local_ips)
            ts_now += 0.0001
            flows.append(flow)
            n_new += 1
        elapsed = time.time() - start
        print(f"  [{elapsed:.0f}s] ss 快照: 新增 {n_new}, 累计 {len(flows)}")
        if time.time() - start < duration:
            time.sleep(interval)
    return flows


# ── Token 化 ──

def tokenize_flows(flows, out_token_path):
    """将 flow 列表转成 7-token 序列文件"""
    # 按时间戳排序
    flows_sorted = sorted(flows, key=lambda f: f.get("ts_sort", f.get("ts_epoch", 0)))
    n = 0
    prev_ts = None
    with open(out_token_path, "w") as out:
        for flow in flows_sorted:
            ts_epoch = flow.get("ts_epoch", 0)
            if prev_ts and ts_epoch:
                delta_ms = max(0, int((ts_epoch - prev_ts) * 1000))
            else:
                delta_ms = 0
            prev_ts = ts_epoch or None
            tokens = flow_to_tokens(flow, delta_ms)
            out.write(json.dumps({
                "ts": flow.get("ts", ""),
                "host": flow.get("host", "fw"),
                "tokens": tokens,
                "raw": {k: flow.get(k) for k in
                        ("src_ip", "dst_ip", "dst_port", "proto", "bytes", "dir")},
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


# ── 主函数 ──

def main():
    import argparse
    ap = argparse.ArgumentParser(description="墨城防火墙宿主机真实流量采集器")
    ap.add_argument("--duration", type=int, default=120,
                    help="采集时长（秒），默认 120")
    ap.add_argument("--interval", type=int, default=3,
                    help="快照采样间隔（秒），默认 3")
    ap.add_argument("--method", choices=["auto", "hybrid", "event", "snapshot", "ss"],
                    default="auto", help="采集方法(auto=hybrid)，默认 auto")
    ap.add_argument("--out-dir", default=DATA_DIR, help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    flows_path = os.path.join(args.out_dir, "host_real_flows.jsonl")
    tokens_path = os.path.join(args.out_dir, "host_real_tokens.jsonl")
    pcap_path = os.path.join(args.out_dir, "host_traffic.pcap")

    # auto = hybrid（最佳数据质量）
    method = "hybrid" if args.method == "auto" else args.method

    local_ips = detect_local_ips()
    print(f"本机 IP: {', '.join(sorted(local_ips)) if local_ips else '(未检测到)'}")
    print(f"采集方法: {method}, 时长: {args.duration}s, 间隔: {args.interval}s")

    flows = []

    if method == "hybrid":
        print(f"\n[1/3] 混合模式采集 (tcpdump + conntrack -L)...")
        flows = collect_hybrid(args.duration, pcap_path, local_ips)

    if method == "event" or (method != "hybrid" and not flows):
        print(f"\n[1/3] 尝试 conntrack -E 实时事件流 ({args.duration}s)...")
        flows = collect_conntrack_events(args.duration, local_ips)
        if flows:
            print(f"  conntrack -E 采集 {len(flows)} 条事件")
        else:
            print("  conntrack -E 无数据，降级到快照模式")
            method = "snapshot"

    if method == "snapshot" and not flows:
        print(f"\n[1/3] 尝试 conntrack -L 周期快照 ({args.duration}s)...")
        flows = collect_conntrack_snapshots(args.duration, args.interval, local_ips)
        if not flows:
            print("  conntrack -L 无数据，降级到 ss")
            method = "ss"

    if method == "ss" and not flows:
        print(f"\n[1/3] 尝试 ss -tnu 兜底 ({args.duration}s)...")
        flows = collect_ss_snapshots(args.duration, args.interval, local_ips)

    if not flows:
        print("[ERROR] 所有数据源均无数据")
        sys.exit(1)

    # 写原始流
    print(f"\n[2/3] 写入原始流文件...")
    flows_sorted = sorted(flows, key=lambda f: f.get("ts_sort", f.get("ts_epoch", 0)))
    with open(flows_path, "w") as f:
        for fl in flows_sorted:
            f.write(json.dumps(fl, ensure_ascii=False) + "\n")
    print(f"  {len(flows_sorted)} 条 -> {flows_path}")

    # Token 化
    print(f"\n[3/3] Token 化 (7-token 格式)...")
    n_tok = tokenize_flows(flows_sorted, tokens_path)
    print(f"  {n_tok} 条 -> {tokens_path}")

    # 统计
    proto_set = set()
    dir_set = set()
    port_set = set()
    for fl in flows_sorted:
        proto_set.add(fl["proto"])
        dir_set.add(fl["dir"])
        port_set.add(fl["dst_port"])
    print(f"\n== 采集统计 ==")
    print(f"  总流数: {len(flows_sorted)}")
    print(f"  协议: {sorted(proto_set)}")
    print(f"  方向: {sorted(dir_set)}")
    print(f"  目标端口数: {len(port_set)}")
    print(f"  Top10 端口: {sorted(port_set)[:10]}")

    # Token 分布 + 字节分布
    tok_counter = Counter()
    size_counter = Counter()
    srcnet_counter = Counter()
    dstnet_counter = Counter()
    for fl in flows_sorted:
        tokens = flow_to_tokens(fl, 0)
        tok_counter.update(tokens)
        size_counter[size_class(fl.get("bytes", 0))] += 1
        srcnet_counter[net_class(fl.get("src_ip", "?"))] += 1
        dstnet_counter[net_class(fl.get("dst_ip", "?"))] += 1
    print(f"  字节分布: {dict(size_counter)}")
    print(f"  源网络: {dict(srcnet_counter)}")
    print(f"  目标网络: {dict(dstnet_counter)}")
    print(f"  Token 种类: {len(tok_counter)}")
    print(f"  Top10 Token:")
    for tok, cnt in tok_counter.most_common(10):
        print(f"    {tok}: {cnt}")

    print(f"\n完成！")


if __name__ == "__main__":
    main()
