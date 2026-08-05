# 司命（Siming）— 行为语法检测引擎 v4

**六网融合管道**：统计 + 语义 + 时序 + 自适应四层独立检测，交叉确认降误报。

## 快速开始

```bash
# 安装
pip install torch numpy scikit-learn
sudo apt install auditd

# 启用 auditd execve 监控
sudo auditctl -a always,exit -F arch=b64 -S execve -k exec_log
sudo auditctl -a always,exit -F arch=b32 -S execve -k exec_log

# 运行六网融合管道
python detector/fusion_pipeline.py --eval data/audit_all.jsonl

# 新机器标定
python detector/onboard_v2.py models/model-stat-v3 data/onboard_benign.jsonl
```

## 最终效果

| 指标 | 值 |
|------|-----|
| 良性 FPR | 0.3% |
| 攻击 TPR | 92.8% |
| exfil/lateral | 100% |
| recon | 99.8% |
| persist | 87.7% |
| privesc | 80.8% |
| FFT C2 检测 | SNR=25.9 |
| 自适应测试 | 7/7 |

## 项目结构

```
siming-full/
├── docs/                     # 文档
│   ├── 司命-系统文档-v4.md     # 完整系统文档
│   ├── 司命-全层升级报告.md     # 升级对比
│   ├── 司命-反自适应升级报告.md  # 早期报告
│   ├── paper_*.md             # 论文
│   ├── figures/               # 8 张论文图表
│   └── 对抗与平衡_蓝队对话纪要.md
├── detector/                 # 核心代码（42 个 .py）
│   ├── fusion_pipeline.py     # 六网融合管道
│   ├── stat_layer_upgrade.py  # 统计层（PREV+EWMA）
│   ├── semantic_layer_upgrade.py # 语义层（窗口分类+focal）
│   ├── temporal_fft.py        # 时序层（FFT+多尺度）
│   ├── adaptive_detector.py   # 自适应层（变体容忍）
│   ├── train_semantic.py      # 对比学习+分类头训练
│   ├── auto_labeler.py        # 自动弱标注器
│   ├── collect_auditd.py      # auditd 采集器
│   ├── deploy_siming.py       # 一键部署 CLI
│   ├── patterns.jsonl         # 模式库（99 条）
│   └── ...                    # 其他工具脚本
├── models/                   # 预训练模型
│   ├── model-stat-v3/         # 统计层（333词表, 3.6MB）
│   ├── model-semantic-v5/     # 语义层（3.5MB）
│   └── model-semantic-embed/  # 语义嵌入（532KB）
├── data/                     # 数据
│   ├── audit_all.jsonl        # 真实 auditd 事件（7598条）
│   ├── synth_attacks_v4.jsonl # 合成攻击（4387条）
│   ├── classifier_train_v5.jsonl # 分类头训练集
│   └── ...
└── README.md
```

## 技术栈

- TinyGPT（4 层 Transformer, 0.90M 参数, 128 维）
- 对比学习（InfoNCE）+ 分类头（6 类行为意图）
- FFT 周期检测 + CV 变异系数
- auditd execve 实时采集
- Python 3.12 / PyTorch 2.5

## License

Apache 2.0 — Zhiyan Security Lab
