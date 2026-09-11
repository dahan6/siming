#!/usr/bin/env python3
"""墨城防火墙: 规则管理 CLI

用法:
  fw_ctl.py pattern list                    # 列出所有模式
  fw_ctl.py pattern add <json>              # 添加模式（JSON 字符串或文件路径）
  fw_ctl.py pattern remove <id>             # 删除模式
  fw_ctl.py proto rebuild [--k 5]           # 重建原型码本
  fw_ctl.py proto list                      # 列出原型码本
  fw_ctl.py drift status                    # 漂移检测状态
  fw_ctl.py drift reset                     # 重置漂移基线
  fw_ctl.py model info                      # 模型信息
  fw_ctl.py enforce list                    # 列出当前执法规则
  fw_ctl.py enforce release <ip>            # 释放 IP
  fw_ctl.py config show                     # 显示当前配置
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    """加载 fw_config.toml"""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return {}
    cfg_path = os.path.join(BASE, "fw_config.toml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "rb") as f:
            return tomllib.load(f)
    return {}


def cmd_pattern(args):
    from fw_patterns import FwPatternDB
    db = FwPatternDB(os.path.join(BASE, "fw_patterns.jsonl"))

    if args.action == "list":
        if not db.entries:
            print("模式库为空")
            return
        print(f"{'ID':<10}{'严重度':>6}{'动作':<10}{'名称':<24}{'技术':<10}")
        print("-" * 65)
        for e in db.entries:
            print(f"{e['id']:<10}{e.get('severity',0):>6}"
                  f"{e.get('action','?'):<10}{e.get('name','?'):<24}"
                  f"{e.get('technique','?'):<10}")
        print(f"\n共 {len(db.entries)} 条")

    elif args.action == "add":
        src = args.json_str
        if os.path.exists(src):
            entry = json.load(open(src))
        else:
            entry = json.loads(src)
        entry.setdefault("id", f"FW-{len(db.entries)+1:03d}")
        entry.setdefault("type", "pattern")
        db.add(entry)
        print(f"已添加: [{entry['id']}] {entry.get('name','?')}")

    elif args.action == "remove":
        before = len(db.entries)
        db.entries = [e for e in db.entries if e.get("id") != args.id]
        removed = before - len(db.entries)
        if removed:
            with open(db.path, "w") as f:
                for e in db.entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            print(f"已删除 {removed} 条 (id={args.id})")
        else:
            print(f"未找到 id={args.id}")


def cmd_proto(args):
    proto_path = os.path.join(BASE, "fw_prototypes.jsonl")

    if args.action == "list":
        if not os.path.exists(proto_path):
            print("原型码本不存在（运行 fw_proto.py 生成）")
            return
        meta = json.load(open(proto_path))
        print(f"版本: {meta.get('version','?')}  k={meta.get('k','?')}  "
              f"margin={meta.get('margin','?')}")
        print(f"\n{'技术':<14}{'原型数':>6}{'半径':>8}{'样本数':>8}")
        print("-" * 40)
        for tid, info in meta.get("techniques", {}).items():
            n_proto = len(info.get("prototypes", []))
            r = info.get("radii", [0])[0]
            n_samp = info.get("n_samples", 0)
            print(f"{tid:<14}{n_proto:>6}{r:>8.3f}{n_samp:>8}")

    elif args.action == "rebuild":
        import subprocess
        py = sys.executable
        cmd = [py, os.path.join(BASE, "fw_proto.py"), "model",
               "--gen-samples", "--k", str(args.k), "--epochs", "200"]
        print(f"重建原型码本: {' '.join(cmd)}")
        subprocess.run(cmd)


def cmd_drift(args):
    model_dir = os.path.join(BASE, "model")

    if args.action == "status":
        snap_path = os.path.join(model_dir, "drift_baseline.json")
        trig_path = os.path.join(model_dir, "RETRAIN_TRIGGER")
        if os.path.exists(trig_path):
            trig = json.load(open(trig_path))
            print(f"!! 重训已触发: {trig.get('ts','?')}")
            for r in trig.get("reasons", []):
                print(f"   - {r}")
        elif os.path.exists(snap_path):
            snap = json.load(open(snap_path))
            print(f"基线正常: vocab={snap.get('vocab','?')} "
                  f"rate={snap.get('rate','?'):.1f}/s "
                  f"subnets={snap.get('n_subnets','?')}")
        else:
            print("无漂移基线（运行 fw_drift.py 建立）")

    elif args.action == "reset":
        snap_path = os.path.join(model_dir, "drift_baseline.json")
        trig_path = os.path.join(model_dir, "RETRAIN_TRIGGER")
        for p in [snap_path, trig_path]:
            if os.path.exists(p):
                os.remove(p)
        print("漂移基线已重置")


def cmd_model(args):
    import torch
    model_dir = os.path.join(BASE, "model")
    prior_path = os.path.join(model_dir, "prior.pt")
    if not os.path.exists(prior_path):
        print("模型不存在")
        return
    ckpt = torch.load(prior_path, map_location="cpu", weights_only=False)
    stoi = ckpt["stoi"]
    cfg = ckpt.get("config", {})
    nll = ckpt.get("baseline_nll", {})
    print(f"词表: {len(stoi)} token")
    print(f"架构: d_model={cfg.get('d_model')} n_layer={cfg.get('n_layer')} "
          f"n_head={cfg.get('n_head')} ctx={cfg.get('ctx')}")
    print(f"基线 NLL: mean={nll.get('mean',0):.3f} p50={nll.get('p50',0):.3f} "
          f"p95={nll.get('p95',0):.3f} p995={nll.get('p995',0):.3f}")
    # slot_tau
    st_path = os.path.join(model_dir, "slot_tau.json")
    if os.path.exists(st_path):
        st = json.load(open(st_path))
        print(f"\n分维度 τ (校准集 {st.get('n_calib','?')} 事件):")
        for slot, tau in st.get("slot_tau", {}).items():
            print(f"  {slot:<12} τ={tau:.3f}")
    # validation
    val_path = os.path.join(model_dir, "last_validation.json")
    if os.path.exists(val_path):
        val = json.load(open(val_path))
        print(f"\n最近验证: FPR(EWMA)={val.get('fpr',0):.2%} "
              f"margin={val.get('margin',0):.1f}x "
              f"pass={'PASS' if val.get('pass') else 'FAIL'}")


def cmd_enforce(args):
    from fw_enforce import Enforcer
    enforcer = Enforcer(dry_run=True)

    if args.action == "list":
        stats = enforcer.stats()
        print(f"规则总数: {stats['total_rules']}  活跃: {stats['active']}")
        for ip, r in enforcer.state.get("rules", {}).items():
            remaining = max(0, r["expires"] - time.time())
            print(f"  {ip:<18} {r['action']:<10} hits={r['hit_count']} "
                  f"TTL={remaining:.0f}s  reason={r['reason']}")

    elif args.action == "release":
        enforcer.release(args.ip)
        print(f"已释放 {args.ip}")


def cmd_status(args):
    """系统一键体检"""
    import subprocess
    from fw_paths import (MODEL_DIR, PRIOR_PT, SLOT_TAU_FILE, PATTERNS_FILE,
                          PROTOTYPES_FILE, DRIFT_BASELINE, RETRAIN_TRIGGER,
                          ALERTS_FILE, FLOWS_FILE, EVENT_FLOWS_FILE, LIVE_FLOWS_FILE,
                          ENFORCE_STATE)
    print("=" * 50)
    print("  墨城防火墙 · 系统体检")
    print("=" * 50)

    # 模型
    print("\n[模型]")
    if os.path.exists(PRIOR_PT):
        import torch
        ckpt = torch.load(PRIOR_PT, map_location="cpu", weights_only=False)
        stoi = ckpt["stoi"]
        nll = ckpt.get("baseline_nll", {})
        val_path = os.path.join(MODEL_DIR, "last_validation.json")
        val_info = ""
        if os.path.exists(val_path):
            val = json.load(open(val_path))
            val_info = f" FPR={val.get('fpr',0):.2%} {'PASS' if val.get('pass') else 'FAIL'}"
        print(f"  词表 {len(stoi)} | τ={nll.get('p995',0):.3f}{val_info}")
        int8_path = os.path.join(MODEL_DIR, "prior-int8.pt")
        if os.path.exists(int8_path):
            print(f"  INT8: {os.path.getsize(int8_path)//1024}KB")
    else:
        print("  ⚠️ 无模型（运行 setup.sh 或手动训练）")

    # 模式 + 原型
    print("\n[规则]")
    from fw_patterns import FwPatternDB
    db = FwPatternDB(PATTERNS_FILE)
    print(f"  模式: {len(db.entries)} 条")
    if os.path.exists(PROTOTYPES_FILE):
        meta = json.load(open(PROTOTYPES_FILE))
        print(f"  原型: {len(meta.get('techniques',{}))} 技术")

    # 采集器
    print("\n[采集]")
    for name, path in [("事件流", EVENT_FLOWS_FILE), ("快照", LIVE_FLOWS_FILE),
                        ("通用", FLOWS_FILE)]:
        if os.path.exists(path):
            n = sum(1 for _ in open(path))
            mtime = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
            print(f"  {name}: {n} 条 (更新 {mtime})")

    # daemon
    print("\n[守护进程]")
    try:
        r = subprocess.run(["systemctl", "is-active", "mocheng-firewall"],
                           capture_output=True, text=True, timeout=3)
        status = r.stdout.strip()
        print(f"  systemd: {status}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  systemd: 不可用")
    # 最新告警
    if os.path.exists(ALERTS_FILE):
        lines = open(ALERTS_FILE).readlines()
        n_alerts = len(lines)
        recent = lines[-100:] if len(lines) > 100 else lines
        from collections import Counter
        prios = Counter()
        for l in recent:
            try:
                prios[json.loads(l)["prio"]] += 1
            except (json.JSONDecodeError, KeyError):
                pass
        breakdown = " ".join(f"{p}={n}" for p, n in sorted(prios.items()))
        print(f"  告警: {n_alerts} 总计 ({breakdown} 近 100)")

    # 执法
    print("\n[执法]")
    if os.path.exists(ENFORCE_STATE):
        enforcer_state = json.load(open(ENFORCE_STATE))
        rules = enforcer_state.get("rules", {})
        now = time.time()
        active = sum(1 for r in rules.values() if r.get("expires", 0) > now)
        print(f"  规则: {active} 活跃 / {len(rules)} 总计")
    else:
        print("  无执法规则")

    # 漂移
    print("\n[漂移]")
    if os.path.exists(RETRAIN_TRIGGER):
        trig = json.load(open(RETRAIN_TRIGGER))
        print(f"  ⚠️ 重训已触发: {trig.get('ts','?')}")
    elif os.path.exists(DRIFT_BASELINE):
        print("  正常（基线已建立）")
    else:
        print("  无基线（未运行 drift 检测）")

    print("\n" + "=" * 50)


def cmd_config(args):
    cfg = load_config()
    if args.action == "show":
        print(json.dumps(cfg, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description="墨城防火墙管理 CLI")
    sub = ap.add_subparsers(dest="cmd")

    # pattern
    p_pat = sub.add_parser("pattern", help="模式库管理")
    p_pat.add_argument("action", choices=["list", "add", "remove"])
    p_pat.add_argument("json_str", nargs="?", help="JSON 字符串或文件路径（add）")
    p_pat.add_argument("id", nargs="?", help="模式 ID（remove）")

    # proto
    p_proto = sub.add_parser("proto", help="原型码本管理")
    p_proto.add_argument("action", choices=["list", "rebuild"])
    p_proto.add_argument("--k", type=int, default=5)

    # drift
    p_drift = sub.add_parser("drift", help="漂移检测管理")
    p_drift.add_argument("action", choices=["status", "reset"])

    # model
    p_model = sub.add_parser("model", help="模型信息")
    p_model.add_argument("action", choices=["info"])

    # enforce
    p_enf = sub.add_parser("enforce", help="执法规则管理")
    p_enf.add_argument("action", choices=["list", "release"])
    p_enf.add_argument("ip", nargs="?", help="IP 地址（release）")

    # config
    p_cfg = sub.add_parser("config", help="配置管理")
    p_cfg.add_argument("action", choices=["show"])

    # status
    p_st = sub.add_parser("status", help="系统一键体检")
    p_st.add_argument("action", choices=["all"])

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return

    handlers = {"pattern": cmd_pattern, "proto": cmd_proto, "drift": cmd_drift,
                "model": cmd_model, "enforce": cmd_enforce, "config": cmd_config,
                "status": cmd_status}
    handlers[args.cmd](args)


if __name__ == "__main__":
    main()
