== [2026-07-30 02:18:20] 第 0-彩排 轮 ==
解析 12468 个事件（跳过坏行 0）-> /home/lab/defense-lab/data/host_tokens.jsonl
12468 /home/lab/defense-lab/data/host_tokens.jsonl
token 总数 87276, 词表 42, 训练 74184 / 验证 13092, 设备 cuda
参数量: 0.82M
epoch 8/8 train_loss 0.0489 val_loss 0.0504 (val ppl 1.1) 440s
基线 NLL 分布: {'mean': 0.067, 'p50': 0.0, 'p95': 0.183, 'p99': 1.325, 'p995': 2.435}
模型已保存 -> /home/lab/defense-lab/detector/model-host-r0-彩排/prior.pt
/home/lab/miniconda3/envs/ai/lib/python3.12/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
/home/lab/miniconda3/envs/ai/lib/python3.12/site-packages/torch/nn/modules/transformer.py:902: UserWarning: Mem Efficient attention on Current AMD GPU is still experimental. Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1. (Triggered internally at /pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:360.)
τ(p995)=2.435 | 留出 800 事件: p50=0.003 p95=2.235 max=14.986 | FPR=4.62%
用例                      惊讶度    判定 触发标准
A-bash高端口外联           10.96    异常 超τ UNK×1
B-base64参数             9.52    异常 超τ UNK×1
C-伪装进程名                9.84    异常 超τ UNK×1
E-root高端口外联            7.87    异常 超τ UNK×1

分离余量(最弱异常/p95)=3.5x | 全部检出=True | FPR达标=False
达标判定: FAIL
Traceback (most recent call last):
  File "/home/lab/defense-lab/detector/validate_host.py", line 91, in <module>
    main()
  File "/home/lab/defense-lab/detector/validate_host.py", line 86, in main
    json.dump({"tau": tau, "fpr": fpr, "p95": p95, "margin": margin,
  File "/home/lab/miniconda3/envs/ai/lib/python3.12/json/__init__.py", line 179, in dump
    for chunk in iterable:
                 ^^^^^^^^
  File "/home/lab/miniconda3/envs/ai/lib/python3.12/json/encoder.py", line 432, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/home/lab/miniconda3/envs/ai/lib/python3.12/json/encoder.py", line 406, in _iterencode_dict
    yield from chunks
  File "/home/lab/miniconda3/envs/ai/lib/python3.12/json/encoder.py", line 439, in _iterencode
    o = _default(o)
        ^^^^^^^^^^^
  File "/home/lab/miniconda3/envs/ai/lib/python3.12/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type bool is not JSON serializable

### 彩排轮分析（02:35）
- 训练正常：87276 token / 词表 42（直采无截断，词表比 VM 版小 3 倍、更干净）/ 0.82M / val_loss 0.0504 / τ=2.435
- 验证管线打通，但发现三个问题：
  1. **json 序列化 bug**（numpy 类型）→ 已修（显式类型转换）
  2. **FPR 4.62% 超标** → 取证发现主因是**观察者污染**：night_round.sh 自身的 tee/date/nice-python 进程恰好在留出窗口内大量出现；次要因 chronyd 高端口 NTP。改进：① 验证增加 EWMA 平滑统计（3.75%，仍超标）② 后续轮次中管线命令进入训练集会被自然学习 ③ 写入论文作为验证纪律案例
  3. 验证输出缺误报取证 → 已加 Top5 打印
- 合成异常 4/4 检出，分离余量 3.5x（达标）
- 结论：管线可信，FPR 问题定性为污染+数据量，非模型缺陷。正式轮观察自然改善情况
== [2026-07-30 03:54:40] 第 1 轮 ==
解析 141490 个事件（跳过坏行 0）-> /home/lab/defense-lab/data/host_tokens.jsonl
141490 /home/lab/defense-lab/data/host_tokens.jsonl

## r0 τ敏感性定量分析（轮1训练期间完成）
- 异常用例分数：A=10.96 B=9.52 C=9.84 E=7.87
- EWMA-FPR 随 τ：2.435→3.75% / 5→2.0% / 7→1.25% / 8→0.88% / 10→0.50%
- 关键结论：r0 数据上不存在同时满足双判据的 τ——FPR≤1% 需 τ≥8，但此时最低异常/τ 余量仅 0.98x（<3x）；τ=2.435 时余量 3.2x 但 FPR=3.75%
- 含义：调阈不是出路，必须靠数据量/质量压缩正常 EWMA 分布尾部 → r1（3倍数据+彩排活动入训练段）是真正的检验
== [2026-07-30 04:55:06] 第 1 轮 ==
解析 267694 个事件（跳过坏行 0）-> /home/lab/defense-lab/data/host_tokens.jsonl
267694 /home/lab/defense-lab/data/host_tokens.jsonl
token 总数 1873858, 词表 148, 训练 1592779 / 验证 281079, 设备 cuda
参数量: 0.85M
epoch 8/8 train_loss 0.0582 val_loss 0.0729 (val ppl 1.1) 10892s
基线 NLL 分布: {'mean': 0.072, 'p50': 0.0, 'p95': 0.248, 'p99': 1.945, 'p995': 2.523}
模型已保存 -> /home/lab/defense-lab/detector/model-host-r1/prior.pt
/home/lab/miniconda3/envs/ai/lib/python3.12/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
/home/lab/miniconda3/envs/ai/lib/python3.12/site-packages/torch/nn/modules/transformer.py:902: UserWarning: Mem Efficient attention on Current AMD GPU is still experimental. Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1. (Triggered internally at /pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:360.)
τ(p995)=2.523 | 留出 800 事件: p50=0.006 p95=1.850 max=10.822 | FPR(原始)=3.00% FPR(EWMA)=1.75%
用例                      惊讶度    判定 触发标准
A-bash高端口外联           10.48    异常 超τ 
B-base64参数            18.37    异常 超τ UNK×1
C-伪装进程名               14.28    异常 超τ UNK×1
E-root高端口外联           13.47    异常 超τ 

分离余量(最弱异常/p95)=5.7x | 全部检出=True | FPR达标=False
达标判定: FAIL

误报 Top5（供迭代分析）:
    10.82 ET:CONN PROC:libuv-worker ARGV0 PARENT:? UID:1000 DST:OTHER DT2
    10.42 ET:EXEC PROC:ssh ARGV:N3IPB PARENT:python3 UID:1000 DST:NONE DT0
    10.11 ET:EXEC PROC:mkdir ARGV:N2P PARENT:bash UID:1000 DST:NONE DT0
     8.46 ET:EXEC PROC:ssh ARGV:N3IPB PARENT:python3 UID:1000 DST:NONE DT0
     7.65 ET:EXEC PROC:ssh ARGV:N3IPB PARENT:python3 UID:1000 DST:NONE DT1


## 轮 1（267,694 事件 / 1.87M token / 词表 148 / 0.85M 参数）
- val_loss 0.0729（ppl 1.1），训练 10892s（GPU 与 router 争用）
- τ(p995)=2.523 | 留出 FPR(EWMA)=1.75%（r0 为 3.75%，数据量路线奏效）
- 异常 4/4 检出，余量 5.7x（vs p95）/ 4.15x（vs τ）
- τ 扫描：τ=3.5 时 FPR=1.00% / 余量 2.99x——双判据各差 0.01，交集仍为空但已是毫厘
- 误报 Top5 定性：libuv-worker 外联（vscode-server）、python3 spawn ssh、mkdir——真实但稀有的用户活动，非管线污染
- 结论：方向不变，轮 2 等更多数据（采集 ~40 事件/s）自然压缩尾部；轮 1 训练曾被我自己设的 1h 后台超时误杀一次，重跑后完成（教训：长任务禁短超时）
== [2026-07-30 07:58:27] 第 2 轮 ==
解析 436059 个事件（跳过坏行 0）-> /home/lab/defense-lab/data/host_tokens.jsonl
436059 /home/lab/defense-lab/data/host_tokens.jsonl
token 总数 3052413, 词表 167, 训练 2594551 / 验证 457862, 设备 cuda
参数量: 0.85M
epoch 1/8 train_loss 0.0938 val_loss 0.0672 (val ppl 1.1) 2241s
epoch 2/8 train_loss 0.0632 val_loss 0.0671 (val ppl 1.1) 4479s
epoch 3/8 train_loss 0.0620 val_loss 0.0684 (val ppl 1.1) 6701s

## 轮 2 中途观察（epoch 3/8）
- 3.05M token / 词表 167；~37min/epoch（GPU 争用持续）
- val_loss 在 epoch 2 已收敛（0.0672→0.0671→0.0684），后续 epoch 无收益
- 改动：EPOCHS 8→4（影响后续轮次；正在跑的轮 2 进程已载入内存不受影响）
epoch 4/8 train_loss 0.0614 val_loss 0.0685 (val ppl 1.1) 8920s
epoch 5/8 train_loss 0.0609 val_loss 0.0692 (val ppl 1.1) 11142s
epoch 6/8 train_loss 0.0606 val_loss 0.0691 (val ppl 1.1) 13369s
epoch 7/8 train_loss 0.0603 val_loss 0.0693 (val ppl 1.1) 15428s
epoch 8/8 train_loss 0.0601 val_loss 0.0706 (val ppl 1.1) 17430s
基线 NLL 分布: {'mean': 0.07, 'p50': 0.0, 'p95': 0.224, 'p99': 1.544, 'p995': 2.655}
模型已保存 -> /home/lab/defense-lab/detector/model-host-r2/prior.pt
/home/lab/miniconda3/envs/ai/lib/python3.12/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
/home/lab/miniconda3/envs/ai/lib/python3.12/site-packages/torch/nn/modules/transformer.py:902: UserWarning: Mem Efficient attention on Current AMD GPU is still experimental. Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1. (Triggered internally at /pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:360.)
τ(p995)=2.655 | 留出 800 事件: p50=0.002 p95=4.889 max=21.586 | FPR(原始)=7.75% FPR(EWMA)=5.50%
用例                      惊讶度    判定 触发标准
A-bash高端口外联           13.52    异常 超τ 
B-base64参数            16.08    异常 超τ UNK×1
C-伪装进程名               13.68    异常 超τ UNK×1
E-root高端口外联           14.56    异常 超τ 

分离余量(最弱异常/p95)=2.8x | 全部检出=True | FPR达标=False
达标判定: FAIL

误报 Top5（供迭代分析）:
    21.59 ET:EXEC PROC:grep ARGV:N2P PARENT:watch_router.sh UID:1000 DST:NONE DT3
    20.37 ET:EXEC PROC:pgrep ARGV:N2- PARENT:watch_router.sh UID:1000 DST:NONE DT1
    18.33 ET:EXEC PROC:sleep ARGV:N1- PARENT:watch_router.sh UID:1000 DST:NONE DT2
    15.30 ET:EXEC PROC:pgrep ARGV:N2- PARENT:bash UID:1000 DST:NONE DT2
    13.86 ET:EXEC PROC:pgrep ARGV:N2- PARENT:bash UID:1000 DST:NONE DT2


## 轮 2 最终结果（436,059 事件 / 3.05M token / 词表 167）
- 首验（800 事件留出）：FPR(EWMA)=5.50% FAIL——但取证发现根因是方法缺陷而非模型：
  800 事件 ≈ 20 秒快照，被 watch_router.sh 轮询爆发主导，无代表性
- 修复：validate_host.py 留出窗口可配（第 3 参数），改用 20000 事件（≈8 分钟真实分布）
- 复验（20000 留出）：**PASS**——FPR(EWMA)=1.00%，异常 4/4，余量 8.8x(vs p95)/5.1x(vs τ)
- τ 扫描（20000 点）：可行区间 [3.0, 4.5]，推荐工作点 τ=3.5~4.0（FPR 0.58~0.70%，余量 3.4~3.9x）
- 方法论教训：留出评估窗口必须覆盖有代表性的时长，短窗口=随机快照，结论会反转
== [2026-07-30 15:51:46] 第 3 轮 ==
解析 986563 个事件（跳过坏行 0）-> /home/lab/defense-lab/data/host_tokens.jsonl
986563 /home/lab/defense-lab/data/host_tokens.jsonl
token 总数 7892504, 词表 306, 训练 6708628 / 验证 1183876, 设备 cuda
参数量: 0.89M
epoch 1/4 train_loss 0.0674 val_loss 0.2266 (val ppl 1.3) 5527s
epoch 2/4 train_loss 0.0544 val_loss 0.2368 (val ppl 1.3) 10689s
epoch 3/4 train_loss 0.0539 val_loss 0.2427 (val ppl 1.3) 15895s
epoch 4/4 train_loss 0.0536 val_loss 0.2488 (val ppl 1.3) 21059s
基线 NLL 分布: {'mean': 0.248, 'p50': 0.0, 'p95': 0.523, 'p99': 8.148, 'p995': 14.883}
模型已保存 -> /home/lab/defense-lab/detector/model-host-r3/prior.pt
/home/lab/miniconda3/envs/ai/lib/python3.12/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
/home/lab/miniconda3/envs/ai/lib/python3.12/site-packages/torch/nn/modules/transformer.py:902: UserWarning: Mem Efficient attention on Current AMD GPU is still experimental. Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1. (Triggered internally at /pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:360.)
τ(p995)=14.883 | 留出 800 事件: p50=0.005 p95=2.750 max=21.414 | FPR(原始)=0.75% FPR(EWMA)=0.00%
用例                      惊讶度    判定 触发标准
A-bash高端口外联           10.05    正常 
B-base64参数            25.60    异常 超τ UNK×1
C-伪装进程名               13.29    异常 UNK×1
E-root高端口外联           14.35    正常 

分离余量(最弱异常/p95)=3.7x | 全部检出=False | FPR达标=True
达标判定: FAIL

误报 Top5（供迭代分析）:
    21.41 ET:EXEC PROC:pgrep ARGV:N2- PC:NONE PARENT:watch_v5.sh UID:1000 DST:NONE DT1
    20.65 ET:EXEC PROC:sleep ARGV:N1- PC:NONE PARENT:watch_v5.sh UID:1000 DST:NONE DT2
    19.70 ET:EXEC PROC:pgrep ARGV:N2- PC:NONE PARENT:watch_v5.sh UID:1000 DST:NONE DT1
    19.21 ET:EXEC PROC:sleep ARGV:N1- PC:NONE PARENT:watch_v5.sh UID:1000 DST:NONE DT2
    18.94 ET:EXEC PROC:grep ARGV:N2P PC:TMP PARENT:watch_v5.sh UID:1000 DST:NONE DT2


## ADFA-LD 外部基准（第一轮）
- 833 良性训练 / 4372 验证 / 746 攻击（6 类），TinyGPT 0.90M
- 首轮（窗口均值打分）：宏平均检出率 12.3% @FPR=1.01%——差评
- 根因：ADFA 攻击是长正常轨迹里的短片段（低足迹），均值打分把攻击稀释掉；
  val_loss 1.04（自家数据 0.07），良性分布本身多样难学
- 修法：窗口分 98 分位（抗稀释，同自家 max-per-event 哲学）+ 4 epoch
