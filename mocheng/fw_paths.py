#!/usr/bin/env python3
"""墨城防火墙: 统一路径管理

所有模块通过此文件获取路径，不再硬编码 ~/mocheng-firewall 或 /home/lab。
项目可放在任意位置，只要文件结构不变即可工作。
"""
import os

# 项目根目录 = 本文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 子目录
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_CURRENT = os.path.join(BASE_DIR, "model-current")
RULES_DIR = os.path.join(BASE_DIR, "rules")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
TESTS_DIR = os.path.join(BASE_DIR, "tests")

# 关键文件路径
CONFIG_FILE = os.path.join(BASE_DIR, "fw_config.toml")
PATTERNS_FILE = os.path.join(BASE_DIR, "fw_patterns.jsonl")
PROTOTYPES_FILE = os.path.join(BASE_DIR, "fw_prototypes.jsonl")

# 数据文件路径（便捷别名）
FLOWS_FILE = os.path.join(DATA_DIR, "flows.jsonl")
LIVE_FLOWS_FILE = os.path.join(DATA_DIR, "live_flows.jsonl")
EVENT_FLOWS_FILE = os.path.join(DATA_DIR, "event_flows.jsonl")
TOKENS_FILE = os.path.join(DATA_DIR, "tokens.jsonl")
ALERTS_FILE = os.path.join(DATA_DIR, "alerts.jsonl")

# 模型文件路径
PRIOR_PT = os.path.join(MODEL_DIR, "prior.pt")
PRIOR_INT8_PT = os.path.join(MODEL_DIR, "prior-int8.pt")
SLOT_TAU_FILE = os.path.join(MODEL_DIR, "slot_tau.json")
DRIFT_BASELINE = os.path.join(MODEL_DIR, "drift_baseline.json")
RETRAIN_TRIGGER = os.path.join(MODEL_DIR, "RETRAIN_TRIGGER")
DAEMON_STATE = os.path.join(MODEL_DIR, "daemon_state.json")

# 执法状态
ENFORCE_STATE = os.path.join(RULES_DIR, "state.json")

# 确保目录存在
for d in [DATA_DIR, MODEL_DIR, RULES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


def model_dir(name=None):
    """获取模型目录。name=None 返回默认 model/，否则返回 model-{name}/"""
    if name is None or name == "model":
        return MODEL_DIR
    return os.path.join(BASE_DIR, f"model-{name}")
