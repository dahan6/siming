#!/usr/bin/env bash
# collect_atomic_local.sh — Atomic Red Team 全量 Linux 采集（shell 版）
#
# 两种模式:
#   A) pwsh + Invoke-AtomicTest（官方方式，推荐）
#   B) 直接解析 YAML 执行（无 pwsh 依赖）
#
# tracee 采集 + parse_raw_tracee.py 解析为 8-token JSONL
#
# 用法:
#   sudo bash collect_atomic_local.sh                    # 全量
#   sudo bash collect_atomic_local.sh T1053.003           # 指定技术
#   bash collect_atomic_local.sh --list                   # 只列出 Linux 测试
#
# 环境变量:
#   ATOMIC_DIR   atomic-red-team 仓库路径  (默认 ~/atomic-red-team)
#   TRACEE_BIN   tracee 二进制            (默认 tracee)
#   OUTPUT       输出 JSONL               (默认 atomic_attacks.jsonl)
#   STAGING      原始 tracee 目录         (默认 data/atomic_full/)
#   COOLDOWN     测试后等待秒数           (默认 3)
set -euo pipefail

# ── 路径 ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ATOMIC_DIR="${ATOMIC_DIR:-$HOME/atomic-red-team}"
TRACEE_BIN="${TRACEE_BIN:-tracee}"
OUTPUT="${OUTPUT:-$SCRIPT_DIR/atomic_attacks.jsonl}"
STAGING="${STAGING:-$SCRIPT_DIR/data/atomic_full}"
COOLDOWN="${COOLDOWN:-3}"
TRACEE_EVENTS="sched_process_exec,security_socket_connect"
HOST="$(hostname 2>/dev/null || echo unknown)"

mkdir -p "$STAGING"
mkdir -p "$(dirname "$OUTPUT")"

# ── 颜色 ───────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn()  { echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $*"; }
error() { echo -e "${RED}[$(date +%H:%M:%S)]${NC} $*" >&2; }

# ── 1. 确保 atomic-red-team 仓库 ──────────────────────────────────────
ensure_repo() {
    if [[ ! -d "$ATOMIC_DIR/atomics" ]]; then
        info "克隆 atomic-red-team → $ATOMIC_DIR"
        git clone --depth 1 https://github.com/redcanaryco/atomic-red-team.git "$ATOMIC_DIR"
    fi
}

# ── 2. 发现 Linux 测试用例 ────────────────────────────────────────────
#    用 Python + pyyaml 提取（如果无 pyyaml 则用 grep 降级方案）
list_linux_tests() {
    local filter="${1:-}"
    if python3 -c "import yaml" 2>/dev/null; then
        python3 - "$ATOMIC_DIR/atomics" "$filter" <<'PYEOF'
import json, os, sys, glob
atomics, filt = sys.argv[1], sys.argv[2] if len(sys.argv)>2 else ""
import yaml
rows = []
for yf in sorted(glob.glob(os.path.join(atomics, "**", "T*.yaml"), recursive=True)):
    try:
        doc = yaml.safe_load(open(yf, errors="replace"))
    except Exception:
        continue
    if not doc: continue
    tech = doc.get("attack_technique", os.path.basename(yf)[:-5])
    if filt and tech.upper() != filt.upper(): continue
    for t in doc.get("atomic_tests") or []:
        if "linux" not in (t.get("supported_platforms") or []):
            continue
        ex = t.get("executor", {})
        if not ex or not ex.get("command", "").strip():
            continue
        rows.append(json.dumps({
            "technique": tech,
            "test_name": t.get("name", tech),
            "command": ex["command"].strip(),
            "cleanup_command": (t.get("cleanup_executor") or {}).get("command", "").strip(),
            "elevation_required": bool(ex.get("elevation_required", False)),
        }))
    # 只输出第一个匹配的 yaml（每技术一个文件）
    if filt: break
for r in rows:
    print(r)
PYEOF
    else
        # 降级：用 grep 粗略提取（可能遗漏多行命令）
        warn "无 pyyaml，使用 grep 降级提取（精度较低）"
        for yf in $(find "$ATOMIC_DIR/atomics" -name 'T*.yaml' | sort); do
            tech=$(basename "$yf" .yaml)
            if [[ -n "$filter" && "$tech" != "$filter"* ]]; then continue; fi
            # 粗略检查是否含 linux 平台
            if ! grep -q "linux" "$yf" 2>/dev/null; then continue; fi
            echo "{\"technique\":\"$tech\",\"test_name\":\"(grep-fallback)\",\"command\":\"echo $tech\",\"cleanup_command\":\"\",\"elevation_required\":false}"
        done
    fi
}

# ── 3. tracee 采集封装 ────────────────────────────────────────────────
start_tracee() {
    local cap_file="$1"
    "$TRACEE_BIN" --events "$TRACEE_EVENTS" --output json \
        > "$cap_file" 2>"${cap_file}.err" &
    TRACEE_PID=$!
    sleep 3   # 等 BPF 加载
}

stop_tracee() {
    if [[ -n "${TRACEE_PID:-}" ]] && kill -0 "$TRACEE_PID" 2>/dev/null; then
        kill -INT "$TRACEE_PID" 2>/dev/null || true
        sleep 1
        kill -TERM "$TRACEE_PID" 2>/dev/null || true
        sleep 1
        kill -KILL "$TRACEE_PID" 2>/dev/null || true
        wait "$TRACEE_PID" 2>/dev/null || true
    fi
}

# ── 4. 基线窗 ─────────────────────────────────────────────────────────
run_baseline() {
    for cmd in "ls /var/log" "ps aux --sort=-%mem | head -5" "df -h" "cat /proc/loadavg" "uptime"; do
        eval "$cmd" > /dev/null 2>&1 || true
        sleep 0.3
    done
}

# ── 5. 解析事件为 8-token JSONL ───────────────────────────────────────
parse_capture() {
    local cap_file="$1" technique="$2" test_name="$3" baseline_n="$4"
    python3 - "$cap_file" "$technique" "$test_name" "$baseline_n" "$HOST" "$OUTPUT" <<'PYEOF'
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if False else ".")
cap_file, technique, test_name, baseline_n, host, output = sys.argv[1:7]
try:
    from parse_raw_tracee import event_to_tokens
except Exception as e:
    print(f"[!] 无法导入 parse_raw_tracee: {e}", file=sys.stderr)
    sys.exit(1)
bn = int(baseline_n)
n = 0
prev_ts = None
with open(cap_file, errors="replace") as f, open(output, "a") as out:
    for line in f:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = ev.get("timestamp", 0)
        delta = 0 if prev_ts is None else max(0, (ts - prev_ts) // 1_000_000)
        prev_ts = ts
        tokens = event_to_tokens(ev, delta)
        if n >= bn:  # 只输出攻击窗事件
            out.write(json.dumps({
                "ts": ts, "host": host, "technique": technique,
                "test_name": test_name, "baseline": False, "tokens": tokens,
            }) + "\n")
        n += 1
print(f"  [parse] {n} 总事件, 攻击窗 {max(0, n-bn)} 条")
PYEOF
}

# ── 6. 单条测试执行 ───────────────────────────────────────────────────
run_one_test() {
    local tech="$1" name="$2" command="$3" cleanup="$4"
    local safe="${tech//./_}"
    local cap_file="$STAGING/${safe}.jsonl"
    local meta_file="$STAGING/${safe}.meta.json"

    echo ""
    info "========================================"
    info "$tech | $name"
    info "========================================"

    # 启动 tracee
    start_tracee "$cap_file"

    # 基线窗
    run_baseline
    local baseline_n=0
    if [[ -f "$cap_file" ]]; then
        baseline_n=$(grep -c '^{' "$cap_file" 2>/dev/null || echo 0)
    fi

    # 执行测试
    echo "  [exec] ${command:0:120}"
    if timeout 120 bash -c "$command" > /tmp/atomic_exec.out 2>&1; then
        info "ok"
    else
        warn "退出码=$? (可能预期失败)"
    fi

    # cleanup
    if [[ -n "$cleanup" ]]; then
        echo "  [cleanup] ${cleanup:0:80}"
        timeout 60 bash -c "$cleanup" > /dev/null 2>&1 || true
    fi

    # cooldown + 停止 tracee
    sleep "$COOLDOWN"
    stop_tracee

    # 统计原始事件
    local raw_n=0
    if [[ -f "$cap_file" ]]; then
        raw_n=$(grep -c '^{' "$cap_file" 2>/dev/null || echo 0)
    fi
    echo "  [capture] $raw_n 条原始事件"

    # 写 meta
    echo "{\"id\":\"$tech\",\"baseline_lines\":$baseline_n,\"test_name\":\"$name\",\"raw_file\":\"$cap_file\"}" > "$meta_file"

    # 解析
    parse_capture "$cap_file" "$tech" "$name" "$baseline_n"
}

# ── 7. pwsh + Invoke-AtomicTest 模式 ──────────────────────────────────
have_pwsh() {
    command -v pwsh &>/dev/null && \
    pwsh -Command "Get-Module -ListAvailable Invoke-AtomicTest" 2>/dev/null | grep -q Invoke
}

run_with_pwsh() {
    local techs=("$@")
    info "使用 pwsh + Invoke-AtomicTest 模式"
    for tech in "${techs[@]}"; do
        local safe="${tech//./_}"
        local cap_file="$STAGING/${safe}.jsonl"
        info "$tech: 启动 tracee"
        start_tracee "$cap_file"
        run_baseline
        local baseline_n=0
        [[ -f "$cap_file" ]] && baseline_n=$(grep -c '^{' "$cap_file" 2>/dev/null || echo 0)

        info "$tech: Invoke-AtomicTest"
        pwsh -Command "Invoke-AtomicTest $tech -GetPrereqs" 2>/dev/null || true
        pwsh -Command "Invoke-AtomicTest $tech" 2>/dev/null || warn "执行失败"
        pwsh -Command "Invoke-AtomicTest $tech -Cleanup" 2>/dev/null || true

        sleep "$COOLDOWN"
        stop_tracee
        echo "{\"id\":\"$tech\",\"baseline_lines\":$baseline_n}" > "$STAGING/${safe}.meta.json"
        parse_capture "$cap_file" "$tech" "Invoke-AtomicTest" "$baseline_n"
    done
}

# ── 主逻辑 ─────────────────────────────────────────────────────────────
main() {
    local filter="${1:-}"

    ensure_repo

    # --list 模式
    if [[ "$filter" == "--list" ]]; then
        info "Linux 测试用例列表:"
        printf "%-16s %-6s %s\n" "技术" "提权" "名称"
        printf '%0.s-' {1..80}; echo
        list_linux_tests "" | while IFS= read -r line; do
            local tech name elev
            tech=$(echo "$line" | python3 -c "import json,sys;print(json.load(sys.stdin)['technique'])" 2>/dev/null || echo "?")
            name=$(echo "$line" | python3 -c "import json,sys;print(json.load(sys.stdin)['test_name'][:50])" 2>/dev/null || echo "?")
            elev=$(echo "$line" | python3 -c "import json,sys;print('Y' if json.load(sys.stdin)['elevation_required'] else ' ')" 2>/dev/null || echo " ")
            printf "%-16s %-6s %s\n" "$tech" "$elev" "$name"
        done
        local total
        total=$(list_linux_tests "" | wc -l)
        echo ""
        info "共 $total 条 Linux 测试用例"
        exit 0
    fi

    # root 检查
    if [[ $EUID -ne 0 ]]; then
        error "tracee 需要 root，请用 sudo 运行"
        exit 1
    fi

    # tracee 检查
    if ! command -v "$TRACEE_BIN" &>/dev/null; then
        error "找不到 tracee: $TRACEE_BIN"
        error "安装: https://github.com/aquasecurity/tracee/releases"
        exit 1
    fi

    info "atomic-red-team: $ATOMIC_DIR"
    info "tracee: $(command -v "$TRACEE_BIN")"
    info "输出: $OUTPUT"
    info "staging: $STAGING"

    # 选择模式
    if have_pwsh && [[ -n "$filter" ]]; then
        # pwsh 模式（指定技术时）
        run_with_pwsh "$filter"
    else
        # YAML 直接执行模式
        info "使用 YAML 直接执行模式"
        local tests
        tests=$(list_linux_tests "$filter")
        local total
        total=$(echo "$tests" | grep -c . || echo 0)
        info "共 $total 条测试用例"

        local idx=0
        echo "$tests" | while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            idx=$((idx + 1))
            # 解析 JSON 字段
            local tech name command cleanup
            tech=$(echo "$line" | python3 -c "import json,sys;print(json.load(sys.stdin)['technique'])" 2>/dev/null)
            name=$(echo "$line" | python3 -c "import json,sys;print(json.load(sys.stdin)['test_name'])" 2>/dev/null)
            command=$(echo "$line" | python3 -c "import json,sys;print(json.load(sys.stdin)['command'])" 2>/dev/null)
            cleanup=$(echo "$line" | python3 -c "import json,sys;print(json.load(sys.stdin).get('cleanup_command',''))" 2>/dev/null)

            [[ -z "$tech" ]] && continue
            echo ""
            info "($idx/$total)"

            run_one_test "$tech" "$name" "$command" "$cleanup"
        done
    fi

    # 汇总
    echo ""
    info "========================================"
    local lines=0
    [[ -f "$OUTPUT" ]] && lines=$(wc -l < "$OUTPUT" || echo 0)
    info "采集完成: $lines 条事件 → $OUTPUT"
    info "原始文件 → $STAGING/"
    info "========================================"
}

main "$@"
