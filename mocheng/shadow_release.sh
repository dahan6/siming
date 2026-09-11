#!/usr/bin/env bash
# 墨城防火墙: 影子发布（shadow release）
# 新模型在影子模式跑一段时间，对比告警差异，确认后 promote，出问题 rollback。
#
# 用法:
#   shadow_release.sh compare <new_model_dir>   # 对比新旧模型告警差异
#   shadow_release.sh promote <new_model_dir>   # 提升新模型为当前
#   shadow_release.sh rollback                  # 回滚到上一版本
set -euo pipefail
FW=~/mocheng-firewall
PY=~/miniconda3/envs/ai/bin/python
CURRENT="$FW/model-current"
BACKUP="$FW/model-backup"

ACTION="${1:?用法: shadow_release.sh <compare|promote|rollback> [new_model_dir]}"

case "$ACTION" in
  compare)
    NEW="${2:?缺 new_model_dir}"
    echo "== 影子对比: $NEW vs $(readlink -f $CURRENT 2>/dev/null || echo $CURRENT) =="

    # 用同一份数据分别跑两个模型
    DATA="$FW/data/mixed_flows.jsonl"
    [ -f "$DATA" ] || { echo "无测试数据 $DATA"; exit 1; }

    echo "--- 当前模型 ---"
    $PY "$FW/fw_daemon.py" "$CURRENT" --src "$DATA" --once --no-enforce \
      --alerts /tmp/shadow_current.jsonl --state /tmp/shadow_cur_state.json 2>/dev/null
    CUR_ALERTS=$(wc -l < /tmp/shadow_current.jsonl 2>/dev/null || echo 0)

    echo "--- 新模型 ---"
    $PY "$FW/fw_daemon.py" "$NEW" --src "$DATA" --once --no-enforce \
      --alerts /tmp/shadow_new.jsonl --state /tmp/shadow_new_state.json 2>/dev/null
    NEW_ALERTS=$(wc -l < /tmp/shadow_new.jsonl 2>/dev/null || echo 0)

    echo ""
    echo "当前模型告警: $CUR_ALERTS"
    echo "新模型告警:   $NEW_ALERTS"
    echo "差异:         $(( NEW_ALERTS - CUR_ALERTS ))"

    # 对比 P0/P1 级别差异
    $PY -c "
import json
cur = [json.loads(l) for l in open('/tmp/shadow_current.jsonl')]
new = [json.loads(l) for l in open('/tmp/shadow_new.jsonl')]
cur_p01 = sum(1 for a in cur if a['prio'] in ('P0','P1'))
new_p01 = sum(1 for a in new if a['prio'] in ('P0','P1'))
print(f'P0+P1 执法级: 当前={cur_p01} 新={new_p01}')
" 2>/dev/null || true
    ;;

  promote)
    NEW="${2:?缺 new_model_dir}"
    [ -d "$NEW" ] || { echo "目录不存在: $NEW"; exit 1; }
    # 备份当前
    if [ -L "$CURRENT" ] || [ -d "$CURRENT" ]; then
      rm -rf "$BACKUP"
      cp -a "$(readlink -f $CURRENT 2>/dev/null || echo $CURRENT)" "$BACKUP"
      echo "已备份当前模型 -> $BACKUP"
    fi
    # 切换
    ln -sfn "$(readlink -f $NEW)" "$CURRENT"
    echo "== 已提升: $NEW -> model-current =="
    ;;

  rollback)
    [ -d "$BACKUP" ] || { echo "无备份可回滚"; exit 1; }
    ln -sfn "$BACKUP" "$CURRENT"
    echo "== 已回滚 -> model-current =="
    ;;

  *)
    echo "未知操作: $ACTION (compare/promote/rollback)"
    exit 1
    ;;
esac
