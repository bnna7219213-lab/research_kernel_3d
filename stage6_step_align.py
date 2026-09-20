# -*- coding: utf-8 -*-
"""阶段 6 · DiT 单步对齐（真实 cond，官方 vs 自研，GPU fp16）。

v2 结论：官方 cond 驱动自研全链路 → 正体素 295/F=688（从 5/36 提升），
但官方 F=11148，仍差一个数量级；cond 全局 cos≈1 但逐 token maxdiff=0.148。

本脚本：同一 x0 [1,4096,64]、同一官方 cond、同一 t=0.5，分别跑
  - 官方 DiT（pipeline.model, fp16, GPU）单步
  - 自研 DiT（stage3_dit, fp16, GPU）单步
比较输出。分两段加载避免同时占显存。

若单步一致 → 差异在采样循环（CFG/更新式）；
若单步不一致 → DiT 前向仍有差异（长 context / fp16 下的结构细节）。

运行：python research_kernel_3d/stage6_step_align.py
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from stage2_conditioner import load_conditioner
from stage3_dit import build_and_load as build_dit, LATENT_DIM
from stage6_self_e2e_v1 import official_preprocess, dino_transform

IMG = os.path.join(ROOT, "output", "aigc_img_00002_.png")
CONFIG_YAML = os.path.join(ROOT, "models", "hy3dgen", "tencent",
                           "Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", "config.yaml")
CKPT = os.path.join(ROOT, "models", "diffusion_models",
                    "hunyuan_3d_v2.1.safetensors")
DEV = "cuda"


def _vendor_path():
    vendor = os.path.join(ROOT, "vendor", "Hunyuan3D-2.1-main", "hy3dshape")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    return vendor


def main():
    torch.backends.cuda.matmul.allow_tf32 = True

    # 1) 固定输入：x0 / cond / t
    torch.manual_seed(42)
    x0 = torch.randn(1, 4096, LATENT_DIM, device=DEV, dtype=torch.float16)
    t = torch.tensor([0.5], device=DEV, dtype=torch.float16)

    img_pm1 = official_preprocess(IMG)
    px = dino_transform(img_pm1.cpu())
    self_cond_model, _ = load_conditioner(device="cpu")
    with torch.no_grad():
        cond = self_cond_model(px)                    # [1,1370,1024] fp32
    del self_cond_model
    cond16 = cond.to(DEV, torch.float16)
    print(f"[input] x0 {tuple(x0.shape)}  cond {tuple(cond16.shape)}  t={t.item()}")

    # 2) 官方 DiT 单步（从 pipeline 拿 model，加载后释放其余组件）
    _vendor_path()
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    print("[1] 官方 DiT 单步 ...")
    t0 = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        CKPT, CONFIG_YAML, device=DEV, dtype=torch.float16,
        use_safetensors=True)
    off_model = pipe.model
    with torch.no_grad():
        out_off = off_model(x0, t, {"main": cond16})
    print(f"    官方: {time.time() - t0:.1f}s  out mean={out_off.mean():.5f} "
          f"std={out_off.std():.5f}")
    np.save(os.path.join(HERE, "stage6_step_off.npy"),
            out_off[0].float().cpu().numpy())
    del pipe, off_model, out_off
    torch.cuda.empty_cache()

    # 3) 自研 DiT 单步
    print("[2] 自研 DiT 单步 ...")
    t0 = time.time()
    dit = build_dit().to(DEV, dtype=torch.float16)
    with torch.no_grad():
        out_self = dit(x0, t, cond16)
    print(f"    自研: {time.time() - t0:.1f}s  out mean={out_self.mean():.5f} "
          f"std={out_self.std():.5f}")
    np.save(os.path.join(HERE, "stage6_step_self.npy"),
            out_self[0].float().cpu().numpy())

    # 4) 对比
    a = out_self.flatten().float()
    b = out_off.float().flatten()
    cos = F.cosine_similarity(a[None], b[None]).item()
    d = (a - b).abs().max().item()
    rel = ((a - b).abs() / (b.abs() + 1e-3)).mean().item()
    # token 级 cosine 分布
    tok_cos = F.cosine_similarity(out_self[0].float(), out_off[0].float(), dim=-1)
    print(f"\n[align] cosine={cos:.6f}  maxdiff={d:.4f}  rel_err={rel:.4f}")
    print(f"[align] token-cos: min={tok_cos.min():.4f} mean={tok_cos.mean():.6f} "
          f"max={tok_cos.max():.6f}")
    ok = cos > 0.99
    print(f"[verdict] {'PASS（单步一致，差异在采样循环/解码）' if ok else 'FAIL（DiT 前向差异，需逐层定位）'}")


if __name__ == "__main__":
    main()
