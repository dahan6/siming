# 墨城防火墙（MoCheng Firewall）

行为语法网络防火墙——用序列学习检测异常流量模式，而非静态规则。

## 架构

```
网络流事件 → 7-token 离散化 → TinyGPT 先验（惊讶度）
                                    ↓
                          四网融合判定（优先级递减）
                          ┌─────────────────────────┐
                          │ P0 原型匹配（攻击 DNA）  │ → DROP
                          │ P1 模式匹配（ATT&CK）   │ → DROP/THROTTLE（需模型佐证）
                          │ P2 上下文异常（组合 τ）  │ → DROP（连续 ≥3 次）
                          │ P3 稀有度异常（UNK）    │ → ALERT
                          └─────────────────────────┘
                                    ↓
                          iptables 执法（TTL 自动过期）
```

**核心洞察**：传统防火墙按 IP/端口做静态过滤，无法检测"用合法端口干非法事"。
墨城学习正常流量的**序列分布**——DNS→HTTPS→NTP 是正常节奏，
连续高端口外联/周期信标/突发大流量是异常节奏。

### 7-token 事件 schema

| 槽位 | 含义 | 示例值 |
|---|---|---|
| PROTO | 协议 | TCP / UDP / ICMP |
| DIR | 方向 | IN / OUT / FWD |
| SRCNET | 源网络 | LAN / VPN / EXT / LOOPBACK |
| DSTNET | 目的网络 | LAN / VPN / EXT |
| PORTCLS | 端口/服务 | HTTPS / SSH / DNS / HIGHPORT / SMB |
| SIZECLS | 流量大小 | S(<200B) / M(<5KB) / L(<100KB) / XL |
| DT | 时间间隔 | DT0(<1ms) ~ DT6(>60s) |

## 快速上手

```bash
# 1. 生成合成数据 + 训练
python synth_traffic.py 8000 1
python fw_tokens.py data/benign_flows.jsonl data/benign_tokens.jsonl
python train_prior.py data/benign_tokens.jsonl model/
python fw_calibrate.py model data/benign_tokens.jsonl

# 2. 训练原型码本
python fw_proto.py model --gen-samples --k 5

# 3. 验证
python fw_validate.py model data/benign_tokens.jsonl

# 4. 跑检测（dry-run）
python fw_daemon.py model --src data/mixed_flows.jsonl --dry-run --once

# 5. 管理 CLI
python fw_ctl.py pattern list
python fw_ctl.py model info
python fw_ctl.py drift status
```

## 部署

```bash
# systemd 服务
sudo cp mocheng-firewall.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mocheng-firewall

# 日志轮转
sudo cp logrotate-mocheng /etc/logrotate.d/mocheng

# 新主机上线
bash onboard_fw.sh ubuntu@198.51.100.88 host1 1800

# 影子发布
bash shadow_release.sh compare model-new/
bash shadow_release.sh promote model-new/
bash shadow_release.sh rollback
```

## 模块清单

| 文件 | 职责 |
|---|---|
| `fw_tokens.py` | 网络流 → 7-token 离散化 |
| `train_prior.py` | TinyGPT 先验训练（0.82M 参数） |
| `fw_score.py` | 批量打分 + EWMA 阈值 |
| `fw_calibrate.py` | 分维度 τ 校准（7 槽位） |
| `fw_patterns.py` | ATT&CK 模式库（6 条默认规则） |
| `fw_proto.py` | 原型学习头（对比学习，5 技术） |
| `fw_validate.py` | 模型验证（FPR + 检出率 + 分离余量） |
| `fw_collect.py` | conntrack 实时采集器 |
| `fw_drift.py` | 漂移守护（词表/分布/速率/子网） |
| `fw_enforce.py` | iptables 执法层（TTL/状态管理） |
| `fw_alert.py` | 告警外发（file/syslog/webhook） |
| `fw_daemon.py` | 四网融合实时守护（核心） |
| `fw_ctl.py` | 规则管理 CLI |
| `fw_config.toml` | 全局配置 |
| `quantize_fw.py` | INT8 量化（torchao PTQ） |
| `synth_traffic.py` | 合成流量生成器 |
| `onboard_fw.sh` | 新主机自适应上线 |
| `shadow_release.sh` | 影子发布 |

## 验证结果（合成数据）

- 模型：0.82M 参数，val ppl 1.5
- EWMA 误报率：0.10%
- 攻击检出：85/85 = 100%（端口扫描/C2信标/数据外泄/横向移动/DNS隧道）
- 原型留一接住率：90.6%，良性误报 0.2%
- 分离余量：2.4x

## 与司命的关系

墨城复用司命（`~/defense-lab/`）的 TinyGPT 架构和三层检测理念，
将事件域从进程/syscall 迁移到网络流。核心引擎不变，token schema 和执法层是新增。
两个项目完全独立，互不影响。
