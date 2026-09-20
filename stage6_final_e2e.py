# -*- coding: utf-8 -*-
"""阶段 6 · 终局验证：自研内核 50 步端到端（对齐官方默认步数）。

证据链（已完成）：
  A. 采样调度数学：自研 _sigmas/更新式 vs 官方 step() → maxdiff=0.00e+00
  B. CFG 组合数学：官方 step() 实跑 → maxdiff=0.00e+00
  C. 体积解码：自研 vs 官方 VolumeDecoder → maxdiff=0.00e+00
  D. 表面提取：skimage MC vs 官方 MCSurfaceExtractor → 逐位一致
  E1. 官方 pipeline 同配置（fp16/res32/10步/GS5）→ V=5466 F=11148 watertight
  E2. 官方 cond 驱动自研全链路 10 步 → F=688 watertight（真实但小）
  S. 真实 cond 单步对齐（fp16 GPU）→ cosine=0.999982 maxdiff=0.035

结论：内核逐步等价；E2 与 E1 的 F 差距最可能来自 10 步种子方差
（官方默认 50 步）。本脚本以 50 步重跑自研全链路做终局判定。

运行：python research_kernel_3d/stage6_final_e2e.py
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from stage2_conditioner import load_conditioner
from stage3_dit import build_and_load as build_dit, LATENT_DIM
from stage4_sampler import sample_loop
from stage5_vae import build_and_load as build_vae, decode_to_grid
from stage6_surface import extract_surface_mc, mesh_stats
from stage6_self_e2e_v1 import official_preprocess, dino_transform

RES = 32
STEPS = 50          # 官方默认
GS = 5.0
SEED = 42
IMG = os.path.join(ROOT, "output", "aigc_img_00002_.png")
DEV = "cuda"


def main():
    torch.backends.cuda.matmul.allow_tf32 = True

    # 条件（官方预处理 + 自研 conditioner，fp32）
    img_pm1 = official_preprocess(IMG)
    px = dino_transform(img_pm1.cpu())
    cond_model, _ = load_conditioner(device="cpu")
    with torch.no_grad():
        cond = cond_model(px)
    del cond_model

    dit = build_dit().to(DEV, dtype=torch.float16)
    torch.manual_seed(SEED)
    x0 = torch.randn(1, 4096, LATENT_DIM, device=DEV, dtype=torch.float16)
    t0 = time.time()
    lat = sample_loop(dit, x0, cond.to(DEV, torch.float16),
                      num_steps=STEPS, shift=1.0, guidance_scale=GS)
    print(f"[sample] {STEPS} 步 {time.time() - t0:.1f}s  "
          f"latent mean={lat.mean():.4f} std={lat.std():.4f}")
    np.savez(os.path.join(HERE, "stage6_final_latent.npz"),
             latent=lat[0].float().cpu().numpy())
    del dit, x0
    torch.cuda.empty_cache()

    print("[decode] CPU fp32 ...")
    t0 = time.time()
    vae = build_vae().cpu()
    grid = decode_to_grid(vae, lat[0].float().cpu().unsqueeze(0),
                          octree_resolution=RES, num_chunks=20000, verbose=False)
    n_pos = int((grid[0] > 0).sum())
    print(f"[decode] {time.time() - t0:.1f}s  grid mean={grid.mean():.4f} "
          f"std={grid.std():.4f} 正体素={n_pos}")

    v, f = extract_surface_mc(grid[0], mc_level=0.0, bounds=1.01,
                              octree_resolution=RES)
    s = mesh_stats(v, f)
    print(f"[mesh] V={s['n_vertices']} F={s['n_faces']} "
          f"watertight={s.get('watertight')} euler={s.get('euler')}")

    npz = os.path.join(HERE, "stage6_final_e2e.npz")
    np.savez(npz, grid=grid[0].numpy(), vertices=v, faces=f)
    import trimesh
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    glb = os.path.join(ROOT, "output", "stage6_final_e2e.glb")
    mesh.export(glb)
    print(f"[export] {glb}")

    ok = s["n_faces"] > 1000 and s.get("watertight") is True
    print(f"[verdict] {'PASS' if ok else 'FAIL'}  "
          f"(参考：官方 10 步 F=11148)")


if __name__ == "__main__":
    main()
