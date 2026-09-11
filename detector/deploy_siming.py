#!/usr/bin/env python3
"""司命一键部署脚本

在新机器上的完整部署流程：
1. 检查依赖（Python3, torch, numpy）
2. 选择/创建模型目录
3. 采集良性基线（或使用通用模型）
4. 标定 slot_tau
5. 启动 deploy_scorer

用法:
  ./deploy_siming.py install          # 安装依赖
  ./deploy_siming.py init <tracee_src>  # 初始化（采集+标定）
  ./deploy_siming.py start             # 启动检测守护
  ./deploy_siming.py stop              # 停止
  ./deploy_siming.py status            # 查看状态
  ./deploy_siming.py test <jsonl>      # 离线测试
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


def get_model_dir():
    """优先级：model-vm-universal > model-current > model-host-r3-clean"""
    candidates = [
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


def do_start():
    """启动 deploy_scorer 守护"""
    model_dir = get_model_dir()
    if not model_dir:
        print("ERROR: 无可用模型")
        sys.exit(1)

    src = os.path.join(HOME, "defense-lab", "data", "host_tracee.jsonl")
    if not os.path.exists(src):
        # 尝试 lado-range 路径
        src = os.path.join(HOME, "lado-range", "telemetry", "tracee.jsonl")

    alerts = os.path.join(HOME, "defense-lab", "data", "alerts.jsonl")
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
    alerts = os.path.join(HOME, "defense-lab", "data", "alerts.jsonl")
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
        do_start()
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
