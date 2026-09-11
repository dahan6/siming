# Security Policy / 安全政策

## Scope / 范围

Siming (司命) is a **defensive** security research project: a host-based behavioral anomaly detection engine, plus MoCheng (墨城), its network-side counterpart. It contains no offensive tooling.

司命是一个**防御性**安全研究项目：主机行为异常检测引擎，以及其网络侧对应项目墨城防火墙。本项目不包含任何攻击性工具。

## Reporting a Vulnerability / 报告漏洞

If you discover a security vulnerability in this project (for example, in the iptables enforcement layer, the onboarding scripts, or the calibration pipeline), please report it responsibly:

如果你在本项目中发现安全漏洞（例如 iptables 执法层、上线脚本或标定流水线中的问题），请负责任地披露：

- Email / 邮箱：**dahanxinshen@outlook.com**
- Please do **not** open a public issue for unpatched vulnerabilities. / 未修复的漏洞请**不要**直接开公开 issue。
- We aim to acknowledge reports within 7 days. / 我们会在 7 天内确认收到报告。

## Operational Safety Notes / 运行安全提示

- Scripts under `detector/` and `mocheng/` that invoke `auditd`, `tracee`, `conntrack`, or `iptables` require root privileges and modify system state. Run them only on dedicated lab machines.
  `detector/` 与 `mocheng/` 中调用 auditd、tracee、conntrack、iptables 的脚本需要 root 权限并会修改系统状态，请只在专用实验机上运行。
- `detector/collect_atomic.py` executes public Atomic Red Team *test* commands to collect labeled telemetry. Run it only in isolated, dedicated test environments — never on production or shared machines.
  `detector/collect_atomic.py` 会执行公开的 Atomic Red Team 测试命令以采集标注遥测，只可在隔离的专用测试环境中运行，切勿在生产或共享机器上执行。
- `mocheng/fw_enforce.py` installs `iptables` rules (fail-closed with TTL). Misconfiguration can disrupt network connectivity; use `--dry-run` first.
  `mocheng/fw_enforce.py` 会安装 iptables 规则（fail-closed + TTL），配置不当可能中断网络连通性，请先使用 `--dry-run`。
- The model files (`*.pt`) are PyTorch checkpoints. Load them only with `torch.load(..., weights_only=True)` or equivalent safe loading practices on untrusted copies.
  模型文件（`*.pt`）为 PyTorch checkpoint。对来源不可信的副本，请使用 `torch.load(..., weights_only=True)` 等安全加载方式。
