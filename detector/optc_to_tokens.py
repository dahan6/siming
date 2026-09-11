#!/usr/bin/env python3
"""OpTC eCAR (parquet) → 司命 8-token 语法适配器

把 OpTC 的 eCAR 端点事件（Windows 10 端点，PROCESS/FLOW）映射为司命 8-token：
  ET / PROC / ARGV / PC / PARENT / UID / DST / DT

设计：
  - PROCESS 事件 → ET:EXEC，PROC=image_path 基名，ARGV=command_line 骨架，
    PARENT=parent_image_path 基名，UID=user 分类
  - FLOW 事件 → ET:CONN，PROC=pid→进程映射（PROCESS 事件建表），
    DST=dest_ip/dest_port 分类，UID=pid→进程映射
  - 按 timestamp 排序计算 DT

用法: optc_to_tokens.py <parquet> <out.jsonl> [--host NAME]
"""
import argparse
import json
import os
import re
import sys

import pyarrow.parquet as pq

BASE64ISH = re.compile(r"^[A-Za-z0-9+/=]{20,}$")
HAS_URL = re.compile(r"https?://")
HAS_IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

PATH_CLASSES = [
    ("SYS32", ("system32", "syswow64")),
    ("TMP", ("\\temp\\", "\\tmp\\", "appdata\\local\\temp")),
    ("STARTUP", ("\\start menu\\programs\\startup", "\\startup\\")),
    ("REG_RUN", ("\\currentversion\\run",)),
    ("PROGFILES", ("program files",)),
    ("USERDIR", ("\\users\\",)),
]

DT_BUCKETS_MS = [1, 10, 100, 1000, 10_000, 60_000]


def basename(p):
    if not p or p != p:  # NaN
        return "?"
    p = str(p).replace("\\", "/")
    return p.rsplit("/", 1)[-1] or "?"


def pathclass(cmdline):
    if not cmdline or cmdline != cmdline:
        return "PC:NONE"
    low = str(cmdline).lower()
    for cls, pats in PATH_CLASSES:
        if any(p in low for p in pats):
            return f"PC:{cls}"
    return "PC:OTHER" if ("\\" in low or "/" in low) else "PC:NONE"


def argv_skeleton(cmdline):
    if not cmdline or cmdline != cmdline:
        return "ARGV0"
    parts = str(cmdline).split()
    n = max(0, len(parts) - 1)
    nb = "N0" if n <= 0 else "N1" if n == 1 else "N2" if n <= 4 else "N3"
    flags = ""
    rest = parts[1:]
    if any(HAS_URL.search(a) for a in rest):
        flags += "U"
    if any(HAS_IP.search(a) for a in rest):
        flags += "I"
    if any(a.startswith(("/", "\\")) or (len(a) > 2 and a[1] == ":") for a in rest):
        flags += "P"
    if any(BASE64ISH.match(a) for a in rest):
        flags += "B"
    return f"ARGV:{nb}{flags or '-'}"


def user_class(u):
    if not u or u != u:
        return "?"
    u = str(u).upper()
    if "SYSTEM" in u or "LOCAL SERVICE" in u or "NETWORK SERVICE" in u:
        return "SYSTEM"
    if "ADMIN" in u:
        return "ADMIN"
    return "USER"


def dst_token(dest_ip, dest_port, proto):
    if not dest_ip or dest_ip != dest_ip:
        return "DST:NONE"
    ip = str(dest_ip)
    try:
        port = int(dest_port) if dest_port == dest_port else 0
    except (ValueError, TypeError):
        port = 0
    # 端口分级
    well = port in (53, 80, 135, 137, 138, 139, 389, 443, 445, 88, 464, 636)
    pc = "WELL" if well else "HIGH"
    # OpTC 内网 10.x；外联极少（隔离网），出现即强信号
    if ip.startswith("10."):
        return f"DST:LAN:{pc}"
    if ip.startswith(("192.168.", "172.")):
        return f"DST:LAN:{pc}"
    return f"DST:EXT:{pc}"


def dt_bucket(ms):
    for i, b in enumerate(DT_BUCKETS_MS):
        if ms < b:
            return f"DT{i}"
    return "DT6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("out")
    ap.add_argument("--host", default=None)
    a = ap.parse_args()

    df = pq.read_table(a.parquet).to_pandas()
    host = a.host or str(df["hostname"].iloc[0]).split(".")[0] if len(df) else "optc"

    # pid → 进程名 映射（PROCESS 事件）
    pid2proc = {}
    for _, r in df[df["object"] == "PROCESS"].iterrows():
        if r["pid"] == r["pid"]:
            pid2proc[int(r["pid"])] = basename(r["image_path"])

    # 排序 + 逐事件映射
    df = df.sort_values("timestamp")
    out_rows = []
    prev_ts = None
    for _, r in df.iterrows():
        ts = r["timestamp"]
        try:
            import pandas as pd
            tms = pd.Timestamp(ts).timestamp() * 1000
        except Exception:
            continue
        delta = 0 if prev_ts is None else max(0, tms - prev_ts)
        prev_ts = tms

        obj = r["object"]
        if obj == "PROCESS":
            proc = basename(r["image_path"])
            tokens = [
                "ET:EXEC", f"PROC:{proc}",
                argv_skeleton(r.get("command_line")),
                pathclass(r.get("command_line")),
                f"PARENT:{basename(r['parent_image_path'])}",
                f"UID:{user_class(r['user'])}",
                "DST:NONE",
                dt_bucket(delta),
            ]
        elif obj == "FLOW":
            pid = int(r["pid"]) if r["pid"] == r["pid"] else -1
            proc = pid2proc.get(pid, "?")
            tokens = [
                "ET:CONN", f"PROC:{proc}",
                "ARGV0", "PC:NONE",
                "PARENT:?",
                f"UID:{user_class(r.get('user'))}",
                dst_token(r["dest_ip"], r["dest_port"], r.get("l4protocol")),
                dt_bucket(delta),
            ]
        else:
            continue  # 跳过 FILE/MODULE/THREAD/REGISTRY 等

        out_rows.append({"ts": tms, "host": host, "tokens": tokens,
                         "object": obj, "pid": int(r["pid"]) if r["pid"] == r["pid"] else -1,
                         "id": str(r["id"])})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{os.path.basename(a.parquet)}: {len(df)} 事件 → {len(out_rows)} token 事件 "
          f"(PROCESS/FLOW) -> {a.out}")


if __name__ == "__main__":
    main()
