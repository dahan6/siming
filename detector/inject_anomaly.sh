#!/usr/bin/env bash
# LADO 靶场验证：三类安全异常行为注入（在 range-l2-a 内执行）
# 全部无害：不出网、不破坏，仅产生偏离基线的行为信号
set -x

# A. bash 内建 /dev/tcp 连管理节点高端口（无监听，连接失败但 CONN 事件已产生）
bash -c 'exec 3<>/dev/tcp/192.0.2.1/4444' 2>/dev/null
bash -c 'exec 3<>/dev/tcp/192.0.2.1/31337' 2>/dev/null

# B. argv 携带 base64 长串（典型payload投递特征）
echo "UydWduZXh0JyBpcyBhIHRlc3Qgc3RyaW5nIGZvciBkZXRlY3Rpb24gcHVycG9zZXM=" | base64 -d > /dev/null 2>&1
bash -c 'x="QWxhZGRpbjpvcGVuIHNlc2FtZSB0aGlzIGlzIGEgbG9uZyBlbmNvZGVkIHBheWxvYWQ="; echo ${x:0:8}'

# C. 陌生进程名链：复制系统工具为怪异名字再执行
cp /usr/bin/ls /tmp/.kworker_u9 2>/dev/null || sudo cp /usr/bin/ls /tmp/.kworker_u9
/tmp/.kworker_u9 /tmp > /dev/null
cp /usr/bin/cat /tmp/.sysd-helper 2>/dev/null || sudo cp /usr/bin/cat /tmp/.sysd-helper
/tmp/.sysd-helper /etc/hostname > /dev/null
rm -f /tmp/.kworker_u9 /tmp/.sysd-helper

set +x
echo "anomaly injection done"
