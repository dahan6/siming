# 脱敏审计记录（SANITIZE_LOG）

开源前对全部入仓文件执行敏感信息审计（密码、私钥、API token、个人身份信息、内网地址、私人路径）。

## 审计方法

- 正则检索：`password|passwd|secret|token|api_key|private key|ssh-rsa|ssh-ed25519|authorized_keys`
- IPv4 地址全量枚举
- 邮箱、手机号、`/home/<user>` 路径、个人用户名检索

## 发现并修复

| # | 文件 | 问题 | 处理 |
|---|------|------|------|
| 1 | `range/xml/user-data.template` | cloud-init 模板含示例明文口令 `lado-range-2026` | 改为 `CHANGE_ME`，并加注释"部署时必须修改，禁止使用默认值上线" |
| 2 | `detector/parse_events.py` | 默认输出路径指向私人工作区 `~/lado-range/detector/data/tokens.jsonl` | 改为仓库相对路径 `data/tokens.jsonl` |
| 3 | `detector/train_prior.py` | 默认模型输出目录指向 `~/lado-range/detector/model` | 改为仓库相对路径 `model` |
| 4 | `detector/score_events.py` | 默认模型路径指向 `~/lado-range/detector/model/prior.pt` | 改为仓库相对路径 `model/prior.pt` |
| 5 | `detector/synth_anomaly_test.py` | 默认模型路径指向 `~/defense-lab/detector/model/prior.pt` | 改为仓库相对路径 `model/prior.pt` |

## 审计后确认无需处理项

- **内网地址**：所有 IPv4 地址均属靶场隔离网段 `192.0.2.0/24` 的规划地址（网关 `192.0.2.1`、DHCP 池 `192.0.2.100–199`），是靶场设计内容，予以保留。
- **无私钥/凭据**：`user-data.template` 中 SSH 公钥为 `SSH_PUBKEY_PLACEHOLDER` 占位符；脚本仅引用密钥文件名 `~/.ssh/lado_range`，不含任何私钥材料。
- **无个人敏感信息**：未发现邮箱、手机号、真实姓名、个人用户名等 PII。`/home/range` 为 VM 内实验账户路径，属设计内容。
- **范围外路径**：`range/scripts/` 中 `$HOME/lado-range/...` 为靶场运行时的约定部署布局（镜像、种子盘、串口日志目录），属操作文档性质，非敏感信息，保留原样。

## 未入仓内容（按设计排除）

- `detector/data/`（遥测数据）、`detector/model/`（训练权重）、`__pycache__/` —— 由 `.gitignore` 保证不入仓。
