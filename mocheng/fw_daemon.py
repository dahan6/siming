#!/usr/bin/env python3
"""墨城防火墙: 五网融合实时守护

判定优先级: P0 原型 > P1 模式 > P_NDR 周期 > P2 上下文 > P3 稀有度

用法:
  fw_daemon.py [model_dir] [--loop] [--interval 5] [--config fw_config.toml]
  fw_daemon.py model/ --loop --interval 10  # 持续监控，每 10 秒扫描
  fw_daemon.py model/ --once                # 单次扫描
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fw_paths import (CONFIG_FILE, PATTERNS_FILE, PROTOTYPES_FILE,
                      FLOWS_FILE, ALERTS_FILE)
from train_prior import TinyGPT, CTX
from fw_tokens import flow_to_tokens, set_vpn_prefixes
from fw_patterns import FwPatternDB
from fw_enforce import Enforcer
from fw_calibrate import slot_of
from fw_ndr import NDRDetector
from fw_alert import AlertSink
from fw_state import ConnectionStateTracker
from fw_metrics import MetricsExporter
from fw_incremental import AdaptiveTau
from fw_whitelist import Whitelist
from fw_aggregate import AlertAggregator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WINDOW = 2 * CTX          # 256，防止 window 无限增长
BATCH_SIZE = 64               # 批量推理大小
INTEGRITY_CHECK_INTERVAL = 5  # 每 N 轮扫描做一次 iptables 篡改检测


def load_config(path=None):
    """加载 fw_config.toml"""
    path = path or CONFIG_FILE
    cfg = {}
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return cfg
    if os.path.exists(path):
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    return cfg


def load_prototypes(path):
    """加载原型码本"""
    if not path or not os.path.exists(path):
        return None
    meta = json.load(open(path))
    protos = {}
    for tid, info in meta.get("techniques", {}).items():
        ps = torch.tensor(info["prototypes"], device=DEVICE)
        r = info["radii"][0] if info["radii"] else 1.0
        protos[tid] = (ps, r)
    return protos if protos else None


def embed_event(model, stoi, tokens):
    """token 序列 → 128d 嵌入"""
    ids = [stoi.get(t, 0) for t in tokens][-CTX:]
    if len(ids) < 2:
        ids = ids * 2
    x = torch.tensor(ids, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        t = x.size(1)
        h = model.tok(x) + model.pos(torch.arange(t, device=DEVICE))
        h = model.blocks(h, mask=model.causal_mask[:t, :t])
        h = model.norm(h)
        return h.mean(dim=1).squeeze(0)


def proto_match(protos, embedding):
    """原型空间匹配"""
    if not protos:
        return None
    best_tech, best_dist = None, 1e9
    for tid, (ps, r) in protos.items():
        d = (ps - embedding.unsqueeze(0)).norm(dim=1).min().item()
        if d < best_dist:
            best_tech, best_dist = tid, d
    if best_tech and best_dist <= protos[best_tech][1]:
        return best_tech, best_dist
    return None


def main():
    ap = argparse.ArgumentParser(description="墨城防火墙守护进程")
    ap.add_argument("model_dir", nargs="?", default="model",
                    help="模型目录（默认 model/）")
    ap.add_argument("--loop", action="store_true",
                    help="持续监控模式（默认单次扫描）")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="持续模式扫描间隔秒数")
    ap.add_argument("--config", default=None, help="配置文件路径")
    ap.add_argument("--src", default=None, help="网络流日志路径")
    ap.add_argument("--alerts", default=None, help="告警输出文件")
    ap.add_argument("--patterns", default=None, help="模式库路径")
    ap.add_argument("--prototypes", default=None, help="原型码本路径")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="不实际操作 iptables（默认开启）")
    ap.add_argument("--enforce", action="store_true",
                    help="启用 iptables 执法（覆盖 dry-run）")
    ap.add_argument("--no-enforce", action="store_true",
                    help="完全不执法，仅检测")
    args = ap.parse_args()

    # ── 加载配置 ──
    cfg = load_config(args.config)
    daemon_cfg = cfg.get("daemon", {})
    model_cfg = cfg.get("model", {})
    enforce_cfg = cfg.get("enforce", {})
    alert_cfg = cfg.get("alert", {})
    slots_cfg = cfg.get("slots", {})

    # 从配置填充默认值
    src_path = args.src or daemon_cfg.get("src", FLOWS_FILE)
    alerts_path = args.alerts or daemon_cfg.get("alerts", ALERTS_FILE)
    patterns_path = args.patterns or PATTERNS_FILE
    proto_path = args.prototypes or PROTOTYPES_FILE
    alpha = model_cfg.get("alpha", 0.3)

    # VPN 前缀
    net_cfg = cfg.get("network", {})
    if "vpn_prefixes" in net_cfg:
        set_vpn_prefixes(net_cfg["vpn_prefixes"])

    # 槽位分组
    CTX_SLOTS = set(slots_cfg.get("context", ["DSTNET", "DT", "PORTCLS"]))
    RARE_SLOTS = set(slots_cfg.get("rarity", ["PROTO", "SRCNET", "SIZECLS", "DIR"]))

    # 模型路径
    model_dir = args.model_dir
    prior_path = os.path.join(model_dir, "prior.pt")
    if not os.path.exists(prior_path):
        print(f"[ERROR] 模型不存在: {prior_path}")
        print("请先运行: python fw_collect.py --event --duration 300")
        print("        python fw_tokens.py data/event_flows.jsonl data/event_tokens.jsonl")
        print("        python train_prior.py data/event_tokens.jsonl model/")
        sys.exit(1)

    # ── 加载模型 ──
    ckpt = torch.load(prior_path, map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tau = ckpt["baseline_nll"]["p995"]
    print(f"模型: {model_dir} (词表 {len(stoi)}, τ={tau:.3f})")

    # 分维度阈值
    slot_tau_path = os.path.join(model_dir, "slot_tau.json")
    slot_tau = {}
    if os.path.exists(slot_tau_path):
        slot_tau = json.load(open(slot_tau_path))["slot_tau"]

    # 模式库
    db = FwPatternDB(patterns_path)
    print(f"模式库: {db.stats()}")

    # 白名单
    whitelist = Whitelist()
    print(f"白名单: {whitelist.stats()}")

    # 告警聚合器（60s 窗口内同组只报一次）
    agg_window = alert_cfg.get("agg_window_sec", 60)
    aggregator = AlertAggregator(window_sec=agg_window)
    # NDR 信标专用聚合器（按 beacon target 去重，更长窗口）
    ndr_agg_window = alert_cfg.get("ndr_agg_window_sec", 600)
    ndr_aggregator = AlertAggregator(window_sec=ndr_agg_window)

    # 原型码本
    protos = load_prototypes(proto_path)
    if protos:
        print(f"原型码本: {len(protos)} 技术")

    # NDR
    ndr = NDRDetector(window=100)
    ndr_score_threshold = enforce_cfg.get("ndr_score_min", 0.5)

    # 执法器
    if args.no_enforce:
        enforcer = None
        print("执法: 已禁用")
    else:
        enforcer = Enforcer(dry_run=not args.enforce)
        print(f"执法: {'dry-run' if not args.enforce else 'ENFORCE'}")

    # 告警 sink（传入 alerts_path 覆盖配置文件中的路径）
    cfg.setdefault("daemon", {})["alerts"] = alerts_path
    alert_sink = AlertSink(cfg)
    print(f"告警输出: {alerts_path}")

    # 执法阈值
    p2_threshold = enforce_cfg.get("p2_threshold", 3)
    p0_ttl = enforce_cfg.get("p0_ttl", 3600)
    p1_ttl = enforce_cfg.get("p1_ttl", 3600)
    p2_ttl = enforce_cfg.get("p2_ttl", 1800)

    # ── 连接状态机 ──
    tracker = ConnectionStateTracker()

    # ── Prometheus metrics ──
    metrics_port = cfg.get("metrics", {}).get("port", 9101)
    metrics = MetricsExporter(port=metrics_port)
    metrics.start()
    metrics.update_model_info(tau=tau)

    # ── 自适应阈值 ──
    adaptive = AdaptiveTau(tau)

    # ── 状态 ──
    state_path = os.path.join(model_dir, "daemon_state.json")
    state = {"offset": 0, "window": [], "ewma": 0.0, "prev_ts_epoch": None,
             "consecutive_p2": 0}
    if os.path.exists(state_path):
        try:
            state.update(json.load(open(state_path)))
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] daemon_state.json 损坏 ({e})，重置为默认值")
            state = {"offset": 0, "window": [], "ewma": 0.0,
                     "prev_ts_epoch": None, "consecutive_p2": 0}

    os.makedirs(os.path.dirname(alerts_path), exist_ok=True)

    scan_cycle = 0

    def scan_once():
        """扫描新增流量一次，返回告警统计"""
        nonlocal tau, scan_cycle
        scan_cycle += 1
        t_start = time.time()

        n_scored = 0
        n_whitelist = 0
        n_alert = {"P0": 0, "P1": 0, "P_NDR": 0, "P2": 0, "P3": 0}
        recent_scores = []

        if not os.path.exists(src_path):
            duration = time.time() - t_start
            metrics.record_scan(duration, n_scored)
            return n_scored, n_alert, n_whitelist

        batch_buf = []

        def flush_batch(buf):
            """批量前向推理：一次 model() 处理整个 batch，再逐条判定"""
            nonlocal n_scored, n_whitelist
            if not buf:
                return

            # 右填充到同一长度后一次前向传播（因果掩码保证实 token 不受 padding 影响）
            windows = [item["window"] for item in buf]
            max_len = max(len(w) for w in windows)
            padded = torch.zeros((len(buf), max_len), dtype=torch.long,
                                 device=DEVICE)
            for i, w in enumerate(windows):
                padded[i, :len(w)] = torch.tensor(w, device=DEVICE)

            with torch.no_grad():
                logits = model(padded)
                lp = torch.log_softmax(logits, dim=-1)

            for i, item in enumerate(buf):
                window = item["window"]
                start = item["start"]
                w_len = len(window)
                tokens = item["tokens"]
                flow = item["flow"]
                ts_epoch = item["ts_epoch"]
                ndr_hit = item["ndr_hit"]
                conn_info = item["conn_info"]

                tgt = torch.tensor(window[start:w_len], device=DEVICE)
                nll = -lp[i, start - 1:w_len - 1].gather(
                    -1, tgt.unsqueeze(-1)).squeeze(-1)
                s_ev = nll.max().item()

                state["ewma"] = alpha * s_ev + (1 - alpha) * state["ewma"]
                recent_scores.append(s_ev)
                metrics.record_flow(s_ev)

                # 分维度
                fired = []
                for t_name, v in zip(tokens, nll.tolist()):
                    s = slot_of(t_name)
                    if v > slot_tau.get(s, 99):
                        fired.append((s, t_name, round(v, 2)))
                n_unk = sum(1 for t_name in tokens if t_name not in stoi)
                ctx_fired = [x for x in fired if x[0] in CTX_SLOTS]
                rare_fired = [x for x in fired if x[0] in RARE_SLOTS]

                # P0 原型
                proto_hit = None
                if protos:
                    embedding = embed_event(model, stoi, tokens)
                    proto_hit = proto_match(protos, embedding)

                # P1 模式
                hits = db.match(tokens)
                patterns = [{"id": h["id"], "technique": h["technique"],
                             "severity": h["severity"], "name": h["name"],
                             "action": h.get("action", "ALERT")} for h in hits]

                # 连接状态机
                conn_state = conn_info["state"]
                is_scan = conn_info.get("is_scan_pattern", False)

                # 白名单匹配（按方向检查远端 IP）
                wl_hit = False
                if "DIR:OUT" in tokens:
                    # 出站: 检查 dst_ip 是否为可信目标
                    wl_hit = whitelist.matches(flow.get("dst_ip", ""))
                else:
                    # 入站/其他: 检查 src_ip
                    wl_hit = whitelist.matches(flow.get("src_ip", ""))

                # 五网判定
                if proto_hit:
                    prio = "P0"
                    state["consecutive_p2"] = 0
                elif patterns:
                    prio = "P1"
                    state["consecutive_p2"] = 0
                elif (ndr_hit and ndr_hit["beacon"]
                      and ndr_hit.get("score", 0) >= ndr_score_threshold
                      and not wl_hit):
                    # NDR 信标: 白名单 IP 跳过（min_samples 已在检测器侧控制）
                    prio = "P_NDR"
                    state["consecutive_p2"] = 0
                elif conn_state == "INVALID":
                    # 无效连接 → 自动 P2
                    prio = "P2"
                    state["consecutive_p2"] += 1
                elif ctx_fired:
                    prio = "P2"
                    state["consecutive_p2"] += 1
                    if is_scan:
                        # 扫描模式 → P2 升级 P1
                        prio = "P1"
                        state["consecutive_p2"] = 0
                elif rare_fired or n_unk > 0:
                    prio = "P3"
                    state["consecutive_p2"] = 0
                else:
                    state["consecutive_p2"] = 0
                    n_scored += 1
                    continue

                n_scored += 1

                # 白名单: 跳过 P1-P3，只保留 P0
                if wl_hit and prio != "P0":
                    n_whitelist += 1
                    continue

                # P3 弱信号: 不独立告警，只做 P0-P2 加分项
                if prio == "P3":
                    n_alert["P3"] += 1
                    continue

                # NDR 信标: 同 beacon target 在窗口内只报一次
                if prio == "P_NDR" and ndr_hit:
                    agg = ndr_aggregator.check(ts_epoch, tokens, prio,
                                               key_override=ndr_hit.get("target", ""))
                    if agg["suppressed"]:
                        continue
                else:
                    # 其他优先级: 同 SRCNET→DSTNET→PORTCLS 组合 60s 内只报一次
                    agg = aggregator.check(ts_epoch, tokens, prio)
                    if agg["suppressed"]:
                        continue

                alert = {
                    "ts": flow.get("ts", ""), "ts_epoch": ts_epoch,
                    "prio": prio,
                    "proto_hit": {"technique": proto_hit[0],
                                  "distance": round(proto_hit[1], 3)} if proto_hit else None,
                    "ndr_hit": ndr_hit if (ndr_hit and ndr_hit["beacon"]) else None,
                    "patterns": patterns or None,
                    "fired_dims": [f"{t}:{v}" for _, t, v in fired] or None,
                    "s_ev": round(s_ev, 3), "ewma": round(state["ewma"], 3),
                    "n_unk": n_unk, "tokens": tokens,
                    "src_ip": flow.get("src_ip", ""),
                    "dst_ip": flow.get("dst_ip", ""),
                    "dst_port": flow.get("dst_port", 0),
                    "proto": flow.get("proto", ""),
                    "bytes": flow.get("bytes", 0),
                    "conn_state": conn_state,
                    "is_scan_pattern": is_scan,
                    "hit_count": agg["hit_count"],
                    "wl_hit": wl_hit,
                }

                n_alert[prio] += 1
                alert_sink.emit(alert)
                metrics.record_alert(prio)

                # 执法
                if enforcer:
                    src = flow.get("src_ip", "")
                    dst = flow.get("dst_ip", "")
                    if prio == "P0":
                        target = dst if "DIR:OUT" in tokens else src
                        enforcer.block(target, reason=f"P0: {proto_hit[0]}", ttl=p0_ttl)
                    elif prio == "P1" and patterns:
                        if s_ev > tau:
                            action = patterns[0].get("action", "ALERT")
                            target = dst if "DIR:OUT" in tokens else src
                            if action == "DROP":
                                enforcer.block(target, reason=f"P1: {patterns[0]['name']}", ttl=p1_ttl)
                            elif action == "THROTTLE":
                                enforcer.throttle(target, reason=f"P1: {patterns[0]['name']}", ttl=p1_ttl)
                    elif prio == "P_NDR" and ndr_hit:
                        target = dst if "DIR:OUT" in tokens else src
                        enforcer.block(target,
                                       reason=f"NDR: period={ndr_hit['period']:.0f}s cv={ndr_hit['cv']:.3f}",
                                       ttl=p0_ttl)
                    elif prio == "P2" and state["consecutive_p2"] >= p2_threshold:
                        target = dst if "DIR:OUT" in tokens else src
                        enforcer.block(target, reason="P2: 连续上下文异常", ttl=p2_ttl)

        with open(src_path, errors="replace") as f:
            f.seek(state["offset"])
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    flow = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts_epoch = flow.get("ts_epoch", 0)
                delta = 0
                if state["prev_ts_epoch"] and ts_epoch:
                    delta = max(0, int((ts_epoch - state["prev_ts_epoch"]) * 1000))
                state["prev_ts_epoch"] = ts_epoch or None

                tokens = flow_to_tokens(flow, delta)
                ids = [stoi.get(t, 0) for t in tokens]

                # 窗口管理：MAX_WINDOW 上限防膨胀，CTX 截取作为模型输入
                state["window"] = (state["window"] + ids)[-MAX_WINDOW:]
                window = state["window"][-CTX:]
                n_tok, L_tok = len(ids), len(window)
                start = max(L_tok - n_tok, 1)

                # NDR 观察（不依赖模型，可在累积阶段完成）
                ndr_hit = ndr.observe(ts_epoch, flow.get("dst_ip", ""),
                                      flow.get("dst_port", 0), flow.get("proto", ""))

                # 连接状态机跟踪
                conn_info = tracker.update(
                    flow.get("proto", ""),
                    flow.get("src_ip", ""),
                    flow.get("src_port", 0),
                    flow.get("dst_ip", ""),
                    flow.get("dst_port", 0),
                    flow.get("state", ""),
                    flow.get("event", ""),
                )

                batch_buf.append({
                    "window": window, "start": start, "tokens": tokens,
                    "flow": flow, "ts_epoch": ts_epoch,
                    "ndr_hit": ndr_hit, "conn_info": conn_info,
                })

                # 缓冲区满 → 批量推理 + 自适应阈值更新
                if len(batch_buf) >= BATCH_SIZE:
                    flush_batch(batch_buf)
                    batch_buf = []
                    if recent_scores:
                        tau = adaptive.update(recent_scores)
                        recent_scores = []

            # EOF：刷出残余
            flush_batch(batch_buf)
            batch_buf = []
            if recent_scores:
                tau = adaptive.update(recent_scores)
                recent_scores = []

            state["offset"] = f.tell()

        json.dump(state, open(state_path, "w"))
        if enforcer:
            enforcer.cleanup_expired()

        # ── 篡改检测：每 N 轮一次 ──
        if enforcer and scan_cycle % INTEGRITY_CHECK_INTERVAL == 0:
            result = enforcer.verify_integrity()
            if result["tampered"]:
                ts = time.strftime("%F %T")
                print(f"[{ts}][WARN] iptables 篡改检测: "
                      f"missing={result['missing']} extra={result['extra']}")

        # ── metrics 收尾 ──
        duration = time.time() - t_start
        metrics.record_scan(duration, n_scored)
        metrics.update_model_info(tau=tau)
        if enforcer:
            metrics.update_blocks(enforcer.stats()["active"])

        return n_scored, n_alert, n_whitelist

    # ── 主循环 ──
    print(f"模式: {'持续监控' if args.loop else '单次扫描'}"
          f"{'（间隔 ' + str(args.interval) + 's）' if args.loop else ''}")
    print(f"数据源: {src_path}")

    while True:
        n_scored, n_alert, n_wl = scan_once()
        ts = time.strftime("%F %T")
        total = sum(n_alert.values())
        if total or args.loop:
            print(f"[{ts}] 扫描 {n_scored} 流 | P0={n_alert['P0']} "
                  f"P1={n_alert['P1']} NDR={n_alert['P_NDR']} "
                  f"P2={n_alert['P2']} P3={n_alert['P3']} (共 {total})"
                  f"{' 白名单跳过=' + str(n_wl) if n_wl else ''}")

        if not args.loop:
            break
        time.sleep(args.interval)

    alert_sink.close()


if __name__ == "__main__":
    main()
