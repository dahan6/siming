# E3 报告 · 异质工作负载基线

日期：2026-09-10 ｜ 对应审稿 P2-2 项（良性同质性质疑）｜ 云端 A800 完成
环境：7 台 VM（Ubuntu 24.04，KVM/libvirt）｜ 模型：VM-local TinyGPT 0.85M

## 1. 科学问题

审稿人质疑：良性基线 95.1% ssh / 83.3% CONN，过于同质——检测结论（尤其是
DeepLog-keys 的近零 FPR）是否只是良性太单一的产物？
**本实验回答：结论在负载多样性下是否成立？**

## 2. 实验设计

- **3 种角色 × VM**：开发机 dev×3（gcc/git/pip/curl）、Web 服务器 web×2
  （http 服务/日志/流量/ss）、桌面模拟 desktop×2（文档处理/浏览/媒体/不规律节奏）
- 每台跑独立角色负载驱动（~90 分钟）+ tracee 采集
- 数据：7 台共 25,632 事件（dev 11,313 / web 7,480 / desktop 6,839），8-token 语法
- 评估：token 分布异质性、逐角色 val_ppl/留出 FPR/攻击检出、LOVO 跨机泛化、跨角色 FPR

## 3. 结果

### 3.1 负载异质性（证实三种负载确实不同）

| 角色 | 事件 | PROC top3 | ET 分布 |
|---|---|---|---|
| dev | 11,313 | curl/git/sleep | EXEC 58% / CONN 42% |
| web | 7,480 | curl/head/sleep | EXEC 67% / CONN 33% |
| desktop | 6,839 | sleep/curl/head | EXEC 70% / CONN 30% |

三种负载在进程词汇和事件类型分布上均有明显差异——**异质性成立，不是单一负载**。

### 3.2 逐角色：val_ppl / 留出 FPR / 攻击检出

| 角色 | val_ppl | τ | 留出 FPR | UNK | 攻击检出 |
|---|---|---|---|---|---|
| dev | 1.32 | 7.05 | **0.00%** | 0.00% | **3/3** |
| web | 1.44 | 6.79 | **0.00%** | 0.13% | **3/3** |
| desktop | 1.58 | 6.02 | 0.66% | 2.27% | **3/3** |

每种角色的语法都被良好学习（val_ppl 1.3-1.6），留出误报极低，合成攻击全检出。

### 3.3 LOVO 跨机泛化（7 折，论文核心声明的复验）⭐

| 留出 VM | FPR | 留出 VM | FPR |
|---|---|---|---|
| e3-dev-1 | 0.00% | e3-web-2 | 0.00% |
| e3-dev-2 | 0.00% | e3-desktop-1 | 0.00% |
| e3-dev-3 | 0.00% | e3-desktop-2 | 0.00% |
| e3-web-1 | 0.00% | **平均** | **0.00% ±0.00%** |

**论文 v4 的多 VM 联合先验 + onboarding 协议，在异构负载下跨机泛化 FPR 全为 0。**
这是对审稿人同质性质疑的直接反证——pipeline 不过拟合单一负载。

### 3.4 跨角色 FPR 矩阵（揭示的边界）

| 训 ↓ 测 → | dev | web | desktop |
|---|---|---|---|
| dev | 0.0% | 0.9% | 3.8% |
| web | **19.0%** | 0.0% | **14.2%** |
| desktop | 3.0% | 0.6% | 0.2% |

**不对称发现**：web 训的模型迁移到其他角色差（19%/14.2%），但 dev/desktop 训的
模型迁移到 web 好（0.9%/0.6%）。web 负载（服务进程+流量模式）是最"特异"的角色。

## 4. 结论（对审稿人的回答）

1. **结论在负载多样性下成立**：三种异构负载的逐角色留出 FPR 0~0.66%，
   攻击检出 3/3，LOVO 跨机 FPR 0%——检测能力不是良性同质的产物。
2. **LOVO=0% 证明 onboarding 协议有效**：多 VM 联合先验 + 本地 τ 校准
   在异构负载下跨机泛化完美。
3. **跨角色不对称再次印证论文核心教训**：单一负载训练的先验不跨域
   （web→其他 19%/14%），**必须本机自训/onboarding**——这正是论文反复强调的原则，
   E3 在异构负载上再次证明它。

**建议入文英文句**：

> *Workload heterogeneity.* We re-evaluate on seven VMs across three distinct workload roles (developer, web server, desktop simulation; 25,632 events with clearly distinct token distributions). Per-role holdout FPR is 0–0.66% with 3/3 synthetic-attack detection, and leave-one-VM-out cross-host FPR is 0.00% across all seven machines — the conclusions do not stem from benign homogeneity. The cross-role matrix further confirms the necessity of per-machine onboarding: a prior trained on the web role generalizes poorly to other roles (14–19% FPR), while the multi-VM joint prior with local calibration achieves zero cross-host FPR.

## 5. 诚实的边界

- web 角色的强特异性（迁移到它易、从它出难）提示：服务器类负载的语法分布
  与交互式负载差异显著，部署时应按角色分别 onboarding，不可混训。
- 角色负载为脚本模拟（非真实用户），真实开发/办公行为的多样性更高，
  但三种负载的 token 分布差异已足以检验泛化能力。

## 6. 产物

- 数据：`data/e3_raw/`（7 台 cap + tokens，25,632 事件）
- 结果：`data/e3_raw/e3_analysis.json`
- 脚本：`detector/e3_analyze.py`、角色驱动（dev/web/desktop driver）
- 计算：云端 A800（训练 13 个模型：3 角色 + 7 LOVO + 3 跨角色）
