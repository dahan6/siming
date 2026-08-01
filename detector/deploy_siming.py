#!/usr/bin/env python3
"""Siming one-click deployment script (司命一键部署脚本)

Full deployment flow on a new machine:
1. Check dependencies (Python3, torch, numpy)
2. Select/create model directory
3. Collect benign baseline (or use the universal model)
4. Calibrate slot_tau
5. Start deploy_scorer

Usage:
  ./deploy_siming.py install          # install dependencies
  ./deploy_siming.py init <tracee_src>  # initialize (extract + calibrate)
  ./deploy_siming.py start [--src <tracee.jsonl>] [--alerts <alerts.jsonl>]
                                       # start detection daemon
  ./deploy_siming.py stop              # stop
  ./deploy_siming.py status            # show status
  ./deploy_siming.py test <jsonl>      # offline test
"""
import json
import os
import shutil
import subprocess
import sys
import time

DET = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")


def cmd(*args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {' '.join(args)}\n{r.stderr}")
        sys.exit(1)
    return r.stdout.strip()


REPO_ROOT = os.path.dirname(DET)


def get_model_dir():
    """Priority: <repo>/models/vm-universal > model-vm-universal > model-current > model-host-r3-clean"""
    candidates = [
        os.path.join(REPO_ROOT, "models", "vm-universal"),
        os.path.join(DET, "model-vm-universal"),
        os.path.join(DET, "model-current"),
        os.path.join(DET, "model-host-r3-clean"),
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "prior.pt")):
            return d
    return None


def do_install():
    """安装依赖"""
    print("=== 安装依赖 ===")
    deps = ["torch", "numpy"]
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✅ {dep} 已安装")
        except ImportError:
            print(f"  安装 {dep}...")
            cmd("pip3", "install", dep)
            print(f"  ✅ {dep} 安装完成")

    # 检查 tracee/falco
    for tool in ["falco", "tracee"]:
        r = subprocess.run(["which", tool], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  ✅ {tool}: {r.stdout.strip()}")
        else:
            print(f"  ⚠️  {tool} 未安装（采集需要）")

    print("\n=== 文件检查 ===")
    required = [
        "deploy_scorer.py", "parse_raw_tracee.py", "pattern_db.py",
        "temporal_analyzer.py", "adaptive_detector.py",
        "train_prior.py", "patterns.jsonl", "prototypes.jsonl",
    ]
    for f in required:
        path = os.path.join(DET, f)
        ok = os.path.exists(path)
        print(f"  {'✅' if ok else '❌'} {f}")

    model_dir = get_model_dir()
    if model_dir:
        print(f"\n  ✅ 模型: {model_dir}")
    else:
        print(f"\n  ❌ 无可用模型（需要训练或下载通用模型）")


def do_init(tracee_src=None):
    """初始化：采集基线 + 标定"""
    model_dir = get_model_dir()
    if not model_dir:
        print("ERROR: 无可用模型。先训练：python3 train_prior.py data/vm_train.jsonl model-vm-universal")
        sys.exit(1)
    print(f"模型: {model_dir}")

    benign_path = os.path.join(DET, "data", "onboard_benign.jsonl")
    os.makedirs(os.path.dirname(benign_path), exist_ok=True)

    if tracee_src and os.path.exists(tracee_src):
        # 从 tracee 日志提取 tokens
        print(f"从 {tracee_src} 提取 tokens...")
        tokens_path = os.path.join(DET, "data", "onboard_tokens.jsonl")
        with open(tokens_path, "w") as out:
            for line in open(tracee_src, errors="replace"):
                if not line.strip().startswith("{"):
                    continue
                try:
                    from parse_raw_tracee import event_to_tokens
                    ev = json.loads(line)
                    ts = ev.get("timestamp", 0)
                    delta = 0
                    tokens = event_to_tokens(ev, delta)
                    out.write(json.dumps({"tokens": tokens}) + "\n")
                except Exception:
                    continue
        benign_path = tokens_path
        print(f"  提取完成 → {benign_path}")
    elif os.path.exists(benign_path):
        print(f"使用已有基线: {benign_path}")
    else:
        print("需要提供 tracee 日志路径，或预放 data/onboard_benign.jsonl")
        sys.exit(1)

    # 标定
    print("\n=== 标定 slot_tau ===")
    cmd = f"cd {DET} && python3 onboard_v2.py {model_dir} {benign_path}"
    print(f"  $ {cmd}")
    os.system(cmd)


def default_alerts_path():
    p = os.path.join(HOME, "siming", "data")
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, "alerts.jsonl")


def do_start(src=None, alerts=None):
    """启动 deploy_scorer 守护"""
    model_dir = get_model_dir()
    if not model_dir:
        print("ERROR: 无可用模型")
        sys.exit(1)

    if not src:
        # default telemetry locations (tracee JSONL stream)
        for cand in [
            os.path.join(HOME, "siming", "telemetry", "tracee.jsonl"),
            os.path.join(REPO_ROOT, "data", "tracee.jsonl"),
        ]:
            if os.path.exists(cand):
                src = cand
                break
    if not src:
        print("ERROR: 未找到遥测数据源。请用 --src 指定 tracee JSONL 文件：")
        print("  ./deploy_siming.py start --src /path/to/tracee.jsonl")
        sys.exit(1)

    if not alerts:
        alerts = default_alerts_path()
    state = os.path.join(model_dir, "scorer_state.json")

    print(f"模型: {model_dir}")
    print(f"数据源: {src}")
    print(f"告警: {alerts}")

    cmd = (
        f"cd {DET} && nohup python3 deploy_scorer.py {model_dir} "
        f"--src {src} --alerts {alerts} --state {state} "
        f"> {model_dir}/scorer.log 2>&1 &"
    )
    os.system(cmd)
    time.sleep(2)
    print(f"\n启动完成。日志: {model_dir}/scorer.log")
    print(f"停止: ./deploy_siming.py stop")


def do_stop():
    """停止"""
    os.system("pkill -f deploy_scorer.py 2>/dev/null")
    print("已停止")


def do_status():
    """查看状态"""
    # 检查进程
    r = subprocess.run(["pgrep", "-f", "deploy_scorer.py"], capture_output=True, text=True)
    if r.returncode == 0:
        print("✅ deploy_scorer 运行中")
        # 读最近日志
        model_dir = get_model_dir()
        if model_dir:
            log = os.path.join(model_dir, "scorer.log")
            if os.path.exists(log):
                lines = open(log).readlines()
                if lines:
                    print(f"  最近: {lines[-1].strip()}")
    else:
        print("❌ deploy_scorer 未运行")

    # 模型
    model_dir = get_model_dir()
    if model_dir:
        print(f"\n模型: {model_dir}")
        for f in ["prior.pt", "slot_tau_local.json", "slot_tau_vm.json", "slot_tau.json"]:
            p = os.path.join(model_dir, f)
            if os.path.exists(p):
                size = os.path.getsize(p)
                mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
                print(f"  ✅ {f} ({size} bytes, {mtime})")

    # 告警统计
    alerts = default_alerts_path()
    if os.path.exists(alerts):
        n = sum(1 for _ in open(alerts))
        print(f"\n告警总数: {n}")


def do_test(jsonl_path):
    """离线测试"""
    model_dir = get_model_dir()
    if not model_dir:
        print("ERROR: 无可用模型")
        sys.exit(1)
    os.system(f"cd {DET} && python3 eval_universal.py {model_dir}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    action = sys.argv[1]
    if action == "install":
        do_install()
    elif action == "init":
        tracee_src = sys.argv[2] if len(sys.argv) > 2 else None
        do_init(tracee_src)
    elif action == "start":
        src = alerts = None
        args = sys.argv[2:]
        for i, a in enumerate(args):
            if a == "--src" and i + 1 < len(args):
                src = args[i + 1]
            elif a == "--alerts" and i + 1 < len(args):
                alerts = args[i + 1]
        do_start(src=src, alerts=alerts)
    elif action == "stop":
        do_stop()
    elif action == "status":
        do_status()
    elif action == "test":
        do_test(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(f"未知命令: {action}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
