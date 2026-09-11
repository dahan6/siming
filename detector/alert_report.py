#!/usr/bin/env python3
"""告警取证摘要：统计融合告警的优先级分布、模式命中分布、Top 上下文维度。
用法: alert_report.py <alerts.jsonl>
"""
import json
import sys
from collections import Counter

prio_c, pat_c, dim_c, proc_c = Counter(), Counter(), Counter(), Counter()
n = 0
for line in open(sys.argv[1]):
    a = json.loads(line)
    n += 1
    prio_c[a["prio"]] += 1
    for p in a.get("patterns") or []:
        pat_c[p["id"]] += 1
    for d in a.get("fired_dims") or []:
        dim_c[d.split(":")[0]] += 1
    toks = a.get("tokens", [])
    if len(toks) > 1:
        proc_c[toks[1]] += 1

print(f"告警总数 {n}")
print(f"优先级: {dict(prio_c)}")
print(f"模式命中 Top: {pat_c.most_common(10)}")
print(f"命中维度 Top: {dim_c.most_common(10)}")
print(f"涉及进程 Top: {proc_c.most_common(10)}")
