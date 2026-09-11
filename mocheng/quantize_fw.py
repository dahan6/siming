#!/usr/bin/env python3
"""墨城防火墙: INT8 量化（PTQ）

用 torchao 对 TinyGPT 做训练后量化（Post-Training Quantization）。
司命验证过：PTQ 直接可用，不需要 QAT，零掉点。

用法: quantize_fw.py <model_dir> [--eval-tokens tokens.jsonl]
输出: <model_dir>/prior-int8.pt
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import TinyGPT, CTX

DEVICE = "cpu"  # 量化在 CPU 上做


def main():
    model_dir = sys.argv[1]
    eval_tokens = None
    if "--eval-tokens" in sys.argv:
        eval_tokens = sys.argv[sys.argv.index("--eval-tokens") + 1]

    prior_path = os.path.join(model_dir, "prior.pt")
    ckpt = torch.load(prior_path, map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi))
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    orig_size = os.path.getsize(prior_path)
    print(f"原始模型: {n_params/1e6:.2f}M 参数, {orig_size/1024:.0f}KB")

    # torchao INT8 量化
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(model, Int8WeightOnlyConfig())
        print("torchao INT8 量化完成")
    except (ImportError, Exception) as e:
        print(f"[WARN] torchao 量化失败: {e}，跳过")
        return

    # 保存
    out_path = os.path.join(model_dir, "prior-int8.pt")
    torch.save({"model": model.state_dict(), "stoi": stoi,
                "config": ckpt["config"], "baseline_nll": ckpt["baseline_nll"],
                "quantized": True},
               out_path)
    int8_size = os.path.getsize(out_path)
    print(f"INT8 模型: {int8_size/1024:.0f}KB (压缩比 {orig_size/int8_size:.1f}x)")

    # 零掉点验证
    if eval_tokens:
        print(f"\n验证（{eval_tokens}）:")
        events = [json.loads(l) for l in open(eval_tokens)][:2000]

        # 原始模型分数
        orig_model = TinyGPT(len(stoi))
        orig_model.load_state_dict(ckpt["model"])
        orig_model.eval()

        def score_batch(mdl, evts):
            scores = []
            window = []
            for ev in evts:
                ids = [stoi.get(t, 0) for t in ev["tokens"]]
                window = (window + ids)[-CTX:]
                n, L = len(ids), len(window)
                start = max(L - n, 1)
                with torch.no_grad():
                    x = torch.tensor(window).unsqueeze(0)
                    lp = torch.log_softmax(mdl(x), dim=-1)[0]
                    tgt = torch.tensor(window[start:L])
                    nll = -lp[start-1:L-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                scores.append(nll.max().item())
            return scores

        orig_scores = score_batch(orig_model, events)
        int8_scores = score_batch(model, events)

        import numpy as np
        diff = np.array(int8_scores) - np.array(orig_scores)
        print(f"  分数差异: mean={diff.mean():.6f} max={np.abs(diff).max():.6f}")
        print(f"  原始 p995={np.percentile(orig_scores, 99.5):.3f} "
              f"INT8 p995={np.percentile(int8_scores, 99.5):.3f}")
        if np.abs(diff).max() < 0.02:
            print("  零掉点验证: PASS")
        else:
            print(f"  零掉点验证: WARN (最大差异 {np.abs(diff).max():.4f})")


if __name__ == "__main__":
    main()
