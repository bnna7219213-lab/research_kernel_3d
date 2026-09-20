# -*- coding: utf-8 -*-
"""阶段 6 · 根因隔离：官方 cond 驱动自研采样链。

背景：
  - E2（自研 cond，10 步）：F=688 watertight
  - E-final（自研 cond，50 步）：F=192 —— 步数越多越差，
    指向每步系统性偏差累积，而非种子方差
  - S 段单步 DiT 对齐 cosine=0.999982，但两 DiT 都由同一自研 cond 驱动，
    未隔离 cond 差异（自研 vs 官方逐 token maxdiff=0.15）

实验：官方 conditioner 生成 cond（保存）→ 自研 DiT 采样 10 步
→ 自研 VAE/MC → 与 E2（F=688）对比。
若 F 跃升至官方量级（≈11000）→ 根因 = conditioner 数值差异；
若仍小 → 根因 = 采样循环 fp16/累积。

运行：python research_kernel_3d/stage6_cond_isolate.py
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from stage3_dit import build_and_load as build_dit, LATENT_DIM
from stage4_sampler import sample_loop
from stage5_vae import build_and_load as build_vae, decode_to_grid
from stage6_surface import extract_surface_mc, mesh_stats
from stage6_self_e2e_v1 import official_preprocess

IMG = os.path.join(ROOT, "output", "aigc_img_00002_.png")

RES = 32
STEPS = 10
GS = 5.0
SEED = 42
CONFIG_YAML = os.path.join(ROOT, "models", "hy3dgen", "tencent",
                           "Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", "config.yaml")
CKPT = os.path.join(ROOT, "models", "diffusion_models",
                    "hunyuan_3d_v2.1.safetensors")
COND_NPZ = os.path.join(HERE, "stage6_official_cond.npz")
DEV = "cuda"


def get_official_cond():
    """官方 conditioner 生成 cond 并保存（若已有则复用）。"""
    if os.path.exists(COND_NPZ):
        d = np.load(COND_NPZ)
        return torch.from_numpy(d["cond"])
    from stage6_step_align import _vendor_path
    _vendor_path()
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        CKPT, CONFIG_YAML, device=DEV, dtype=torch.float16,
        use_safetensors=True)
    cond_inputs = pipe.prepare_image(IMG)          # 官方 ImageProcessorV2
    image = cond_inputs.pop("image").to(DEV)
    with torch.no_grad():
        cond = pipe.conditioner(image=image)["main"]   # [1,1370,1024] fp16
    np.savez(COND_NPZ, cond=cond.float().cpu().numpy())
    print(f"[cond] 官方 cond 已保存 {COND_NPZ}")
    del pipe
    torch.cuda.empty_cache()
    return cond.float().cpu()


def main():
    torch.backends.cuda.matmul.allow_tf32 = True

    print("[1] 官方 cond ...")
    t0 = time.time()
    cond_off = get_official_cond()
    print(f"    {time.time() - t0:.1f}s  shape={tuple(cond_off.shape)} "
          f"tok_norm={cond_off.norm(dim=-1).mean():.3f}")

    print("[2] 自研 DiT 采样（官方 cond，10 步）...")
    dit = build_dit().to(DEV, dtype=torch.float16)
    torch.manual_seed(SEED)
    x0 = torch.randn(1, 4096, LATENT_DIM, device=DEV, dtype=torch.float16)
    t0 = time.time()
    lat = sample_loop(dit, x0, cond_off.to(DEV, torch.float16),
                      num_steps=STEPS, shift=1.0, guidance_scale=GS)
    print(f"    {time.time() - t0:.1f}s  latent mean={lat.mean():.4f} "
          f"std={lat.std():.4f}")
    np.savez(os.path.join(HERE, "stage6_condiso_latent.npz"),
             latent=lat[0].float().cpu().numpy())
    del dit, x0
    torch.cuda.empty_cache()

    print("[3] 自研 VAE/MC ...")
    vae = build_vae().cpu()
    grid = decode_to_grid(vae, lat[0].float().cpu().unsqueeze(0),
                          octree_resolution=RES, num_chunks=20000, verbose=False)
    n_pos = int((grid[0] > 0).sum())
    v, f = extract_surface_mc(grid[0], mc_level=0.0, bounds=1.01,
                              octree_resolution=RES)
    s = mesh_stats(v, f)
    print(f"    grid mean={grid.mean():.4f} std={grid.std():.4f} 正体素={n_pos}")
    print(f"[mesh] V={s['n_vertices']} F={s['n_faces']} "
          f"watertight={s.get('watertight')}")

    import trimesh
    glb = os.path.join(ROOT, "output", "stage6_cond_isolate.glb")
    trimesh.Trimesh(vertices=v, faces=f, process=False).export(glb)
    print(f"[export] {glb}")
    print(f"\n[compare] 官方cond+自研链: F={s['n_faces']}")
    print(f"          自研cond+自研链(E2): F=688")
    print(f"          官方全链(E1): F=11148")


if __name__ == "__main__":
    main()
