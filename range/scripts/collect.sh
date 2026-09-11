#!/usr/bin/env bash
# 采集执行器：驱动防御 VM 逐个执行攻击技术并采集 tracee 遥测
# 流程（每技术）：快照还原 → 等 ssh 就绪 → 部署 tracee → 启动采集 →
#                基线窗(10s) → attack → 观察窗(5s) → cleanup → 拉取遥测 → 落盘 staging
# 用法: collect.sh <vm_ip> [只跑指定技术id]
set -euo pipefail
VM_IP="${1:?用法: collect.sh <vm_ip> [技术id]}"
ONLY="${2:-}"
VM=defense-atomic-0
DL=~/defense-lab
OUT=$DL/data/atomic_staging
mkdir -p "$OUT"
SSH="sshpass -p <vm-password> ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=8 ubuntu@$VM_IP"
SCP="sshpass -p <vm-password> scp -o StrictHostKeyChecking=no"
SSHPASS="<vm-password>"

wait_ssh() {  # 快照还原后等 VM 回来（最多 120s）
  for i in $(seq 1 24); do
    $SSH true 2>/dev/null && return 0
    sleep 5
  done
  echo "!! VM 120s 未就绪" >&2; return 1
}

deploy_tracee() {  # 幂等部署并冒烟验证（否则本技术判失败，不许静默空采）
  $SCP /usr/local/bin/tracee-defense "ubuntu@$VM_IP:/tmp/tracee-bin" >/dev/null 2>&1
  $SSH "chmod +x /tmp/tracee-bin"
  run_vm 'nohup /tmp/tracee-bin --events sched_process_exec,security_socket_connect --output json > /tmp/smoke.jsonl 2>/tmp/smoke.err &'
  sleep 4
  N=$($SSH "echo $SSHPASS | sudo -S wc -l < /tmp/smoke.jsonl" 2>/dev/null || echo 0)
  run_vm 'pkill tracee; rm -f /tmp/smoke.jsonl /tmp/smoke.err'
  [ "${N:-0}" -gt 0 ]
}

run_vm() { $SSH "echo $SSHPASS | sudo -S bash -c '$1'" ; }

python3 - "$DL/range/atomic/collect_manifest.yaml" "$ONLY" <<'EOF' > /tmp/techniques.tsv
import sys, yaml
man = yaml.safe_load(open(sys.argv[1]))
only = sys.argv[2]
for t in man:
    if only and t["id"] != only:
        continue
    import json
    print(t["id"], json.dumps(t, ensure_ascii=False), sep="\t")
EOF

while IFS=$'\t' read -r tid tjson; do
  echo "=== $tid ==="
  virsh -c qemu:///system snapshot-revert $VM clean >/dev/null 2>&1 || true
  wait_ssh
  if ! deploy_tracee; then
    echo "  !! tracee 冒烟失败，跳过 $tid"
    continue
  fi

  # 攻击载荷文件（所有技术共用的假木马）
  $SSH 'printf "#!/bin/sh\ncurl http://198.51.100.1/c2 2>/dev/null || ping -c1 198.51.100.1\n" > /tmp/.persist.sh; chmod +x /tmp/.persist.sh'

  run_vm 'nohup /tmp/tracee-bin --events sched_process_exec,security_socket_connect --output json > /tmp/atk_trace.jsonl 2>/dev/null &'
  sleep 3
  BASE_START=$($SSH "echo $SSHPASS | sudo -S wc -l < /tmp/atk_trace.jsonl" 2>/dev/null || echo 0)
  sleep 10   # 基线窗

  # setup + attack（逐条执行，记录失败的）
  echo "$tjson" | python3 -c "
import json,sys
t=json.load(sys.stdin)
for c in t.get('setup',[])+t['attack']: print(c)
" | while read -r cmd; do
    [ -n "$cmd" ] && $SSH "$cmd" >/dev/null 2>&1 || echo "  (命令非零退出: ${cmd:0:60})"
  done
  sleep 5    # 观察窗

  # cleanup
  echo "$tjson" | python3 -c "
import json,sys
t=json.load(sys.stdin)
for c in t.get('cleanup',[]): print(c)
" | while read -r cmd; do
    [ -n "$cmd" ] && $SSH "$cmd" >/dev/null 2>&1 || true
  done

  run_vm 'pkill tracee || true'
  SAFE=$(echo "$tid" | tr './' '__')
  $SSH "echo $SSHPASS | sudo -S cat /tmp/atk_trace.jsonl" > "$OUT/${SAFE}.jsonl" 2>/dev/null
  LINES=$(wc -l < "$OUT/${SAFE}.jsonl")
  echo "{\"id\":\"$tid\",\"baseline_lines\":$BASE_START}" > "$OUT/${SAFE}.meta.json"
  echo "  -> $OUT/${SAFE}.jsonl ($LINES 行)"
  [ "$LINES" -lt 5 ] && echo "  !! 行数过少，需人工检查"
done < /tmp/techniques.tsv

echo "采集完成 -> $OUT"
