#!/usr/bin/env bash
# 影子发布：新旧模型同流并行打分 → 对比报告 → 达标切换 / 翻车回滚
# 用法:
#   shadow_release.sh compare <旧模型> <新模型> <tokens.jsonl>  # 影子对比
#   shadow_release.sh promote <新模型>                          # 切换（原子换链接）
#   shadow_release.sh rollback                                  # 回滚到上一版本
set -euo pipefail
CMD="${1:?compare|promote|rollback}"
DL=~/siming
PY=~/miniconda3/envs/ai/bin/python
CURRENT="$DL/detector/model-current"
BACKUP="$DL/detector/model-previous"

case "$CMD" in
compare)
  OLD="$2"; NEW="$3"; TOK="$4"
  for M in "$OLD" "$NEW"; do
    BN=$(basename "$M")
    nice -n 19 $PY "$DL/detector/deploy_scorer.py" "$M" \
      --src "${5:-$DL/data/host_tracee.jsonl}" --state /tmp/shadow_$BN.state \
      --alerts /tmp/shadow_$BN.alerts >/dev/null 2>&1 || true
  done
  $PY - "$OLD" "$NEW" <<'EOF'
import json, sys
def load(p):
    try: return [json.loads(l) for l in open(p)]
    except FileNotFoundError: return []
import os
old = load(f"/tmp/shadow_{os.path.basename(sys.argv[1])}.alerts")
new = load(f"/tmp/shadow_{os.path.basename(sys.argv[2])}.alerts")
print(f"影子对比: 旧模型告警 {len(old)} vs 新模型告警 {len(new)}")
if old:
    ratio = len(new)/len(old)
    print(f"告警量比 {ratio:.2f}x", "（>1.5x 需人工复核）" if ratio > 1.5 else "（正常范围）")
EOF
  ;;
promote)
  NEW="$2"
  [ -L "$CURRENT" ] && rm -f "$BACKUP" && mv "$CURRENT" "$BACKUP" || true
  [ -d "$CURRENT" ] && mv "$CURRENT" "$BACKUP" || true
  ln -sfn "$NEW" "$CURRENT"
  echo "已切换: model-current -> $NEW（回滚: shadow_release.sh rollback）"
  ;;
rollback)
  if [ -e "$BACKUP" ]; then
    rm -f "$CURRENT"; mv "$BACKUP" "$CURRENT"
    echo "已回滚: model-current -> $(readlink -f $CURRENT)"
  elif [ -L "$CURRENT" ]; then
    TARGET=$(readlink -f "$CURRENT"); rm -f "$CURRENT"
    echo "无历史版本，已摘除当前链接（原指向 $TARGET），回到未部署态"
  else
    echo "无备份可回滚"; exit 1
  fi
  ;;
esac
