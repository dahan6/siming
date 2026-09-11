# Behavior Grammar Range（行为语法靶场）

防御研究项目：**梯度防御靶场（LADO Range）+ 行为语法异常检测器**。

- **靶场侧**：基于 libvirt/KVM 的隔离网络靶场，支持按"防御层级"（tier）克隆 VM，逐层叠加防御手段，用于测量"每层防御买来多少可见性"。
- **检测器侧**：把 tracee 遥测事件按字段离散化为"行为语法" token 流（每事件 7 个 token：事件类型/进程/参数骨架/父进程/UID/目标/时间间隔），用小型因果 Transformer 学习正常行为的序列先验，以 token 惊讶度（NLL）+ EWMA 滑窗 + 基线 p995 分位阈值判定异常，误报率有数学上界；词表外（`<UNK>`）事件必报。

## 架构

```
                 物理宿主机 (192.0.2.1, virbr-lado)
        ┌──────────────────────────────────────────────┐
        │  隔离网段 192.0.2.0/24（lockdown.sh 双向切断转发）│
        │                                              │
        │   VM-A (tier0)   VM-B (tier1)   VM-C (tier2) │
        │      │ tracee + rsyslog 转发 (tcp/514)       │
        └──────┼───────────────────────────────────────┘
               ▼
        /var/log/lado-range/<host>/*.log   ← 遥测汇聚
               │
               ▼
   detector/parse_events.py     事件 → 离散 token 流 (JSONL)
               │
               ▼
   detector/train_prior.py      训练序列先验 → model/prior.pt
               │                （含基线 NLL 分位数：p50/p95/p99/p995）
               ▼
   detector/score_events.py     在线打分：超阈或含 UNK → 告警
   detector/synth_anomaly_test.py  离线合成异常自测（不碰 VM）
```

## 快速开始

前置：libvirt/KVM 宿主机，VM 内已装 tracee；cloud-init 模板需注入主机名、防御层级与 SSH 公钥占位符。

```bash
# 0. 部署前必改：range/xml/user-data.template 中的 CHANGE_ME 占位口令
# 1. 建隔离网络（无出网路由），并加 iptables 兜底
sudo virsh net-define range/xml/lado-isolated-net.xml && sudo virsh net-start lado-isolated
sudo bash range/scripts/lockdown.sh          # 双向切断隔离网段转发，幂等

# 2. 克隆一台指定防御层级的 VM（脚本假定运行时布局 $HOME/lado-range/）
bash range/scripts/create-vm.sh vm-a 1 2048 2   # 名称 层级 内存MB vCPU
bash range/scripts/reset-vm.sh vm-a 1            # 重建回到干净状态

# 3. 产生行为数据（在 VM 内）
bash detector/benign_workload.sh 80              # 良性运维负载
bash detector/inject_anomaly.sh                  # 标定用异常（仅隔离网内探测）

# 4. 遥测 → token 流（rsyslog 已把 VM 日志汇聚到宿主机）
python3 detector/parse_events.py "/var/log/lado-range/*/*.log" data/tokens.jsonl

# 5. 训练先验（输出 model/prior.pt + 基线分位数）
python3 detector/train_prior.py data/tokens.jsonl model

# 6. 打分与验证
python3 detector/score_events.py data/tokens.jsonl --top 15
python3 detector/synth_anomaly_test.py           # 离线合成异常自测
```

## 目录说明

```
range/scripts/   靶场 VM 生命周期：create-vm.sh / reset-vm.sh / lockdown.sh
range/xml/       libvirt 隔离网络定义 + cloud-init 模板（含占位符，部署前必改口令）
detector/        行为语法检测器：解析、训练、打分、负载/异常生成与离线自测
docs/            设计文档与方法论讨论
```

约定：`data/`（遥测与 token 流）、`model/`（权重）、`figures/` 均为本地产物，由 `.gitignore` 排除，不入仓。

## Dual-use 声明

本仓库仅用于**防御研究与检测能力测量**：

- 不包含任何攻击代码、漏洞利用或武器化工具；`inject_anomaly.sh` 仅为检测标定用的隔离网内行为探针，不具攻击性载荷。
- 靶场默认无出网路由，并通过 `lockdown.sh` 在宿主机双向阻断隔离网段转发，实验流量不出实验室。
- 遥测数据（`data/`）与模型权重（`model/`）不入仓；使用真实遥测时请自行确保符合所在机构的数据合规与隐私要求。
- 请勿将本项目的脚本与流程用于任何未授权的环境。

## 引用

如本项目对你的研究有帮助，请引用（见 `CITATION.cff`，正式条目待补）：

> synbasin. *Behavior Grammar Range: 梯度防御靶场与行为语法异常检测器*. 2026.

## 许可

Apache-2.0，见 `LICENSE`。
