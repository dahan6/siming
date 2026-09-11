#!/usr/bin/env python3
"""宿主机原始 tracee JSONL → token 流（与 parse_events.py 同一套 token 规则）
宿主机直采不经过 rsyslog，无截断问题，PROC 名干净。

用法: parse_raw_tracee.py <raw.jsonl> <out.jsonl> [--host NAME]
"""
import json
import os
import sys

from parse_events import args_to_dict, argv_skeleton, dst_token, dt_bucket, pathclass_token


def event_to_tokens(event, delta_ms):
    name = event.get("eventName", "unknown")
    args = args_to_dict(event)
    proc = event.get("processName", "?")
    parent = str(args.get("prev_comm") or "?")
    uid = event.get("userId", "?")
    et = {"sched_process_exec": "EXEC", "security_socket_connect": "CONN"}.get(name, name)
    return [
        f"ET:{et}",
        f"PROC:{proc}",
        argv_skeleton(args.get("argv")),
        pathclass_token(args.get("argv")),
        f"PARENT:{parent}",
        f"UID:{uid}",
        dst_token(name, args),
        dt_bucket(delta_ms),
    ]


def main():
    src, out_path = sys.argv[1], sys.argv[2]
    host = "host-workstation"
    if "--host" in sys.argv:
        host = sys.argv[sys.argv.index("--host") + 1]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n, bad, prev_ts = 0, 0, None
    with open(src, errors="replace") as f, open(out_path, "w") as out:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            ts = ev.get("timestamp", 0)
            delta_ms = 0 if prev_ts is None else max(0, (ts - prev_ts) // 1_000_000)
            prev_ts = ts
            out.write(json.dumps({
                "ts": ts, "host": host,
                "tokens": event_to_tokens(ev, delta_ms),
            }) + "\n")
            n += 1
    print(f"解析 {n} 个事件（跳过坏行 {bad}）-> {out_path}")


if __name__ == "__main__":
    main()
