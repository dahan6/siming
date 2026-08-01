#!/usr/bin/env python3
"""Atomic Red Team 全量 Linux 攻击数据采集脚本

在 Linux 靶机上遍历 Atomic Red Team 的所有 Linux 测试用例，用 tracee
采集行为，解析成 8-token 格式（ET/PROC/ARGV/PC/PARENT/UID/DST/DT）输出 JSONL。

典型流程（每条技术）:
  1. 启动 tracee 采集 → 短暂基线窗（良性命令）
  2. 执行原子测试命令（+ cleanup）
  3. cooldown 等事件落盘
  4. 停止 tracee
  5. parse_raw_tracee.event_to_tokens() 解析 → 带标签写入 atomic_attacks.jsonl

依赖:
  - tracee（eBPF），需 root 运行
  - git（克隆 atomic-red-team）
  - pyyaml（解析 YAML 测试定义）

用法:
  # 完整采集（在靶机 root 下执行）
  sudo python3 collect_atomic.py

  # 只列出所有 Linux 测试用例
  python3 collect_atomic.py --list-only

  # 指定单项技术
  sudo python3 collect_atomic.py --technique T1053.003

  # 干跑（不执行命令，只打印）
  python3 collect_atomic.py --dry-run

  # 自定义路径
  sudo python3 collect_atomic.py --atomic-dir /opt/atomic-red-team \\
      --tracee-bin /usr/local/bin/tracee \\
      --output /tmp/atomic_attacks.jsonl
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── 复用同目录的 8-token 解析逻辑 ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 尝试导入 parse_raw_tracee；失败时做惰性导入（--list-only 不需要）
_have_parser = False
try:
    from parse_raw_tracee import event_to_tokens  # noqa: F401
    _have_parser = True
except Exception:
    pass

# ── 常量 ───────────────────────────────────────────────────────────────
ATOMIC_REPO = "https://github.com/redcanaryco/atomic-red-team.git"
TRACEE_EVENTS = "sched_process_exec,security_socket_connect"
BASELINE_CMDS = [
    "ls /var/log",
    "ps aux --sort=-%mem | head -5",
    "df -h",
    "cat /proc/loadavg",
    "uptime",
]
DEFAULT_COOLDOWN = 3          # 测试结束后等几秒让事件落盘
DEFAULT_TRACEE_STARTUP = 3    # tracee 加载 BPF 的启动等待
DEFAULT_BASELINE_LINES = 20   # 基线窗事件数目标（用于 meta 标记）

# ── YAML 解析 ──────────────────────────────────────────────────────────

def load_yaml(path):
    """安全加载 YAML 文件。"""
    try:
        import yaml
    except ImportError:
        print("[!] 缺少 pyyaml，请安装: pip3 install pyyaml", file=sys.stderr)
        sys.exit(1)
    with open(path, errors="replace") as f:
        return yaml.safe_load(f)


def discover_linux_tests(atomics_dir):
    """扫描 atomics/ 目录，返回所有 Linux 测试用例列表。

    每条: {
        technique, test_name, command, cleanup_command,
        elevation_required, executor_name, yaml_path
    }
    """
    tests = []
    yaml_files = sorted(Path(atomics_dir).rglob("T*.yaml"))
    for yf in yaml_files:
        try:
            doc = load_yaml(yf)
        except Exception as e:
            print(f"  [warn] 解析失败 {yf.name}: {e}", file=sys.stderr)
            continue
        if not doc or not isinstance(doc, dict):
            continue
        technique = doc.get("attack_technique", yf.stem)
        for t in doc.get("atomic_tests", []) or []:
            platforms = t.get("supported_platforms", [])
            if "linux" not in platforms:
                continue
            executor = t.get("executor", {})
            if not isinstance(executor, dict):
                continue
            cmd = (executor.get("command") or "").strip()
            if not cmd:
                continue
            cleanup_exec = t.get("cleanup_executor") or {}
            cleanup_cmd = ""
            if isinstance(cleanup_exec, dict):
                cleanup_cmd = (cleanup_exec.get("command") or "").strip()
            tests.append({
                "technique": technique,
                "test_name": t.get("name", technique),
                "command": cmd,
                "cleanup_command": cleanup_cmd,
                "elevation_required": bool(executor.get("elevation_required", False)),
                "executor_name": executor.get("name", "bash"),
                "yaml_path": str(yf),
            })
    return tests


# ── tracee 采集 ────────────────────────────────────────────────────────

def find_tracee(tracee_bin):
    """查找可用的 tracee 二进制。"""
    if tracee_bin and os.path.isfile(tracee_bin):
        return tracee_bin
    for name in (tracee_bin or "tracee", "tracee-defense", "/usr/local/bin/tracee"):
        p = shutil.which(name) if not os.path.isabs(name) else name
        if p and os.path.isfile(p):
            return p
    return None


def start_tracee(tracee_path, capture_file, events=TRACEE_EVENTS):
    """后台启动 tracee，返回 Popen 对象。"""
    log_file = str(capture_file) + ".err"
    ferr = open(log_file, "w")
    proc = subprocess.Popen(
        [tracee_path, "--events", events, "--output", "json"],
        stdout=open(capture_file, "w"),
        stderr=ferr,
        preexec_fn=os.setsid,   # 新进程组，方便整组 kill
    )
    return proc


def stop_tracee(proc):
    """优雅停止 tracee 进程（SIGINT → SIGTERM → SIGKILL）。"""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=3)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


# ── 事件解析 ───────────────────────────────────────────────────────────

def parse_capture(capture_file, technique, test_name, baseline_lines, host):
    """把 tracee 原始 JSONL 解析成 8-token 记录列表。

    延迟导入 parse_raw_tracee，确保 --list-only 无需解析器。
    """
    try:
        from parse_raw_tracee import event_to_tokens
    except Exception:
        print("[!] 无法导入 parse_raw_tracee.event_to_tokens", file=sys.stderr)
        return []

    records = []
    prev_ts = None
    n = 0
    with open(capture_file, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = ev.get("timestamp", 0)
            delta_ms = 0 if prev_ts is None else max(0, (ts - prev_ts) // 1_000_000)
            prev_ts = ts
            tokens = event_to_tokens(ev, delta_ms)
            records.append({
                "ts": ts,
                "host": host,
                "technique": technique,
                "test_name": test_name,
                "baseline": n < baseline_lines,
                "tokens": tokens,
            })
            n += 1
    return records


# ── 主流程 ─────────────────────────────────────────────────────────────

def ensure_atomic_repo(atomic_dir):
    """确保 atomic-red-team 仓库存在，不存在则 clone。"""
    atomics = os.path.join(atomic_dir, "atomics")
    if os.path.isdir(atomics):
        return atomics
    parent = os.path.dirname(atomic_dir) or "."
    os.makedirs(parent, exist_ok=True)
    print(f"[setup] 克隆 atomic-red-team → {atomic_dir}")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", ATOMIC_REPO, atomic_dir],
        stdout=sys.stderr,
    )
    return atomics


def run_test(test, tracee_path, staging_dir, cooldown, baseline_cmds, host,
             output_file, tracee_startup):
    """采集单条原子测试，返回事件记录数。"""
    tech = test["technique"]
    name = test["test_name"]
    safe = tech.replace(".", "_")
    capture_file = os.path.join(staging_dir, f"{safe}.jsonl")
    meta_file = os.path.join(staging_dir, f"{safe}.meta.json")

    print(f"\n{'='*70}")
    print(f"[run] {tech} | {name}")
    print(f"{'='*70}")

    # ── 启动 tracee ──
    proc = start_tracee(tracee_path, capture_file)
    time.sleep(tracee_startup)

    # ── 基线窗 ──
    for cmd in baseline_cmds:
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        time.sleep(0.3)
    # 记录基线行数
    baseline_lines = 0
    if os.path.isfile(capture_file):
        with open(capture_file) as f:
            baseline_lines = sum(1 for _ in f)

    # ── 执行原子测试 ──
    executor = test["executor_name"]
    command = test["command"]
    t0 = time.time()
    print(f"  [exec] {command[:120]}{'…' if len(command) > 120 else ''}")
    rc = subprocess.run(
        command, shell=True, executable=f"/bin/{executor}"
        if executor in ("bash", "sh") else "/bin/bash",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
    )
    elapsed = time.time() - t0
    if rc.returncode != 0:
        err = rc.stderr.decode(errors="replace")[:200]
        print(f"  [warn] 退出码={rc.returncode} ({elapsed:.1f}s) {err}")
    else:
        print(f"  [ok] 完成 ({elapsed:.1f}s)")

    # ── cleanup ──
    if test["cleanup_command"]:
        print(f"  [cleanup] {test['cleanup_command'][:80]}")
        subprocess.run(test["cleanup_command"], shell=True,
                       executable="/bin/bash",
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=60)

    # ── cooldown + 停止 tracee ──
    time.sleep(cooldown)
    stop_tracee(proc)

    # ── 解析 ──
    raw_n = 0
    if os.path.isfile(capture_file):
        with open(capture_file) as f:
            raw_n = sum(1 for line in f if line.strip().startswith("{"))
    print(f"  [capture] {raw_n} 条原始事件")

    records = parse_capture(capture_file, tech, name, baseline_lines, host)
    attack_records = [r for r in records if not r["baseline"]]
    print(f"  [parse] {len(attack_records)} 条攻击窗事件 (基线 {baseline_lines})")

    # 写入 meta
    with open(meta_file, "w") as f:
        json.dump({"id": tech, "baseline_lines": baseline_lines,
                    "test_name": name, "raw_file": capture_file}, f)

    # 追加到输出
    if attack_records:
        with open(output_file, "a") as out:
            for r in attack_records:
                out.write(json.dumps(r) + "\n")

    return len(attack_records)


def main():
    ap = argparse.ArgumentParser(
        description="Atomic Red Team 全量 Linux 攻击数据采集 → 8-token JSONL")
    ap.add_argument("--atomic-dir", default=os.path.expanduser("~/atomic-red-team"),
                    help="atomic-red-team 仓库路径（不存在则自动 clone）")
    ap.add_argument("--tracee-bin", default="tracee",
                    help="tracee 二进制路径")
    ap.add_argument("--output", default="atomic_attacks.jsonl",
                    help="输出 JSONL 路径")
    ap.add_argument("--staging-dir", default=None,
                    help="每条技术的原始 tracee 文件存放目录（默认 data/atomic_full/）")
    ap.add_argument("--technique", action="append", default=[],
                    help="只跑指定技术（可多次，如 --technique T1053.003）")
    ap.add_argument("--list-only", action="store_true",
                    help="只列出 Linux 测试用例，不执行")
    ap.add_argument("--dry-run", action="store_true",
                    help="打印将执行的命令但不跑 tracee/atomic")
    ap.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN,
                    help=f"测试后等几秒让事件落盘（默认 {DEFAULT_COOLDOWN}）")
    ap.add_argument("--tracee-startup", type=int, default=DEFAULT_TRACEE_STARTUP,
                    help=f"tracee 启动等待秒数（默认 {DEFAULT_TRACEE_STARTUP}）")
    ap.add_argument("--host", default=None,
                    help="主机标签（默认自动获取 hostname）")
    args = ap.parse_args()

    host = args.host or subprocess.check_output("hostname", shell=True).decode().strip()

    # ── 获取 atomic-red-team ──
    atomics_dir = ensure_atomic_repo(args.atomic_dir)
    print(f"[setup] atomics 目录: {atomics_dir}")

    # ── 发现 Linux 测试 ──
    tests = discover_linux_tests(atomics_dir)
    print(f"[scan] 发现 {len(tests)} 条 Linux 测试用例")

    # 技术过滤
    if args.technique:
        want = {t.upper() for t in args.technique}
        tests = [t for t in tests if t["technique"].upper() in want]
        print(f"[filter] 技术过滤后 {len(tests)} 条")

    if args.list_only:
        print(f"\n{'技术':<16}{'提权':<6}{'名称'}")
        print("-" * 80)
        for t in tests:
            elev = "Y" if t["elevation_required"] else " "
            print(f"{t['technique']:<16}{elev:<6}{t['test_name']}")
        # 技术级统计
        techs = sorted({t["technique"] for t in tests})
        print(f"\n共 {len(techs)} 个技术, {len(tests)} 条测试用例")
        return

    if args.dry_run:
        print("\n[dry-run] 将执行以下命令:")
        for t in tests:
            print(f"\n--- {t['technique']} | {t['test_name']} ---")
            print(t["command"][:300])
        return

    # ── 检查 root ──
    if os.geteuid() != 0:
        print("[!] tracee 需要 root，请用 sudo 运行", file=sys.stderr)
        sys.exit(1)

    # ── 检查 tracee ──
    tracee_path = find_tracee(args.tracee_bin)
    if not tracee_path:
        print(f"[!] 找不到 tracee 二进制: {args.tracee_bin}", file=sys.stderr)
        print("    安装: https://github.com/aquasecurity/tracee/releases", file=sys.stderr)
        sys.exit(1)
    print(f"[setup] tracee: {tracee_path}")

    # ── 准备输出 ──
    output_file = args.output
    staging_dir = args.staging_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "atomic_full")
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # 恢复支持：读已有输出，跳过已完成的技术
    done_techs = set()
    if os.path.isfile(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    done_techs.add(json.loads(line).get("technique"))
                except json.JSONDecodeError:
                    pass
        print(f"[resume] 已有输出中 {len(done_techs)} 个技术已完成，将跳过")

    # ── 逐条采集 ──
    total_events = 0
    total_tests = len(tests)
    for i, t in enumerate(tests, 1):
        if t["technique"] in done_techs:
            print(f"[skip] ({i}/{total_tests}) {t['technique']} 已采集")
            continue
        print(f"\n[{i}/{total_tests}]", end="")
        try:
            n = run_test(t, tracee_path, staging_dir, args.cooldown,
                         BASELINE_CMDS, host, output_file, args.tracee_startup)
            total_events += n
        except KeyboardInterrupt:
            print("\n[!] 用户中断，已保存当前进度")
            break
        except Exception as e:
            print(f"  [error] {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"[done] 采集完成: {total_events} 条攻击窗事件 → {output_file}")
    print(f"       原始 tracee 文件 → {staging_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
