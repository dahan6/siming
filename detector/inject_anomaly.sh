#!/usr/bin/env bash
# Test-environment validation: inject three classes of anomalous behavior
# (run inside the test environment). All actions are harmless: no outbound
# traffic, no damage — they only produce behavioral signals off baseline.
set -x

# A. bash built-in /dev/tcp to an unreachable TEST-NET-1 address (RFC 5737)
#    (no listener; the connection fails but the CONN event is produced)
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
