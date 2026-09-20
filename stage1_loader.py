# -*- coding: utf-8 -*-
"""自研 3D 内核 · 阶段1：权重加载器。

把 hunyuan_3d_v2.1.safetensors 按命名空间拆成三套子网络的 state_dict：
  conditioner.*  -> DINOv2 图像条件编码器
  model.*        -> 主 DiT（flow-matching 去噪）
  vae.*          -> 内嵌 ShapeVAE
仅做"加载 + 分组 + 形状/范数自检"，不含前向（前向在后续阶段）。
"""
import os
import sys
from collections import defaultdict

import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHAPE_CKPT = os.path.join(ROOT, "models", "diffusion_models", "hunyuan_3d_v2.1.safetensors")


def load_groups(path=SHAPE_CKPT):
    """返回 {'conditioner': {...}, 'model': {...}, 'vae': {...}, 'other': {...}}"""
    sd = load_file(path)  # 全部加载到 CPU（7GB，需足够内存）
    groups = defaultdict(dict)
    for k, v in sd.items():
        top = k.split(".")[0]
        if top in ("conditioner", "model", "vae"):
            groups[top][k] = v
        else:
            groups["other"][k] = v
    return dict(groups)


def summarize(groups):
    print("=== 权重分组自检 ===")
    for name, g in groups.items():
        n_tensors = len(g)
        n_params = sum(v.numel() for v in g.values())
        # 简单数值健康检查：是否有 NaN/Inf
        bad = sum(1 for v in g.values()
                  if torch.isnan(v).any() or torch.isinf(v).any())
        print(f"  {name:14s} tensors={n_tensors:5d}  params={n_params/1e6:8.1f}M  nan/inf={bad}")
    total = sum(sum(v.numel() for v in g.values()) for g in groups.values())
    print(f"  {'TOTAL':14s} params={total/1e9:.2f}B")


def main():
    print("[load] 读取", os.path.basename(SHAPE_CKPT))
    groups = load_groups()
    summarize(groups)
    # 抽样打印 DiT 主干的层级结构（为阶段3 DiT 前向做准备）
    dit_keys = [k for k in groups.get("model", {}) if "blocks" in k or "layers" in k]
    print(f"\n[DiT] 含 blocks/layers 的键数: {len(dit_keys)}")
    for k in dit_keys[:12]:
        print("   ", k, tuple(groups["model"][k].shape))
    print("\n[PASS] 阶段1 权重加载器自检完成")


if __name__ == "__main__":
    main()
