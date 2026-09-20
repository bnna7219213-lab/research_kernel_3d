# -*- coding: utf-8 -*-
"""阶段 6 · 轨迹对比：官方 vs 自研，同 x0，逐步 latent 比对。

已排除的假设：
  - 硬件/配置（E1 官方同配置 F=11148）
  - 预处理（E3→E2 跨越）
  - cond 差异（cond-isolate：官方 cond 驱动自研链 F=720 ≈ 688）
  - 单步 DiT 前向（S：cosine=0.999982）
  - 调度/CFG 数学（A/B：0.00e+00）

本实验：monkey-patch 官方 prepare_latents 注入同一 x0，callback 捕获
官方每步 latent；自研 inline 循环记录每步 x；逐步对比 cosine/maxdiff。
另将官方最终 latent 交给自研 VAE/MC 解码（交叉验证解码链）。

运行：python research_kernel_3d/stage6_traj_compare.py
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

from stage3_dit import build_and_load as build_dit, LATENT_DIM
from stage4_sampler import _sigmas
from stage5_vae import build_and_load as build_vae, decode_to_grid
from stage6_surface import extract_surface_mc, mesh_stats

RES = 32
STEPS = 10
GS = 5.0
SEED = 42
IMG = os.path.join(ROOT, "output", "aigc_img_00002_.png")
CONFIG_YAML = os.path.join(ROOT, "models", "hy3dgen", "tencent",
                           "Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", "config.yaml")
CKPT = os.path.join(ROOT, "models", "diffusion_models",
                    "hunyuan_3d_v2.1.safetensors")
DEV = "cuda"
OUT_NPZ = os.path.join(HERE, "stage6_traj_compare.npz")


def _vendor_path():
    vendor = os.path.join(ROOT, "vendor", "Hunyuan3D-2.1-main", "hy3dshape")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    return vendor


def run_official_with_fixed_x0(x0):
    """官方 pipeline + 注入 x0 + callback 捕获每步 latent。"""
    _vendor_path()
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    print("[1] 官方 pipeline（注入固定 x0，逐步捕获）...")
    t0 = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        CKPT, CONFIG_YAML, device=DEV, dtype=torch.float16,
        use_safetensors=True)
    print(f"    加载 {time.time() - t0:.1f}s")

    # 注入固定 x0：替换 prepare_latents
    def _prepare_latents(batch_size, dtype, device, generator, latents=None):
        return x0.clone()
    pipe.prepare_latents = _prepare_latents

    traj = []  # 每步 prev_sample

    def cb(step_idx, t, outputs):
        traj.append(outputs.prev_sample[0].float().cpu().numpy())
        print(f"    [official step {step_idx}] captured")

    t0 = time.time()
    out = pipe(
        image=IMG,
        num_inference_steps=STEPS,
        guidance_scale=GS,
        octree_resolution=RES,
        mc_level=0.0,
        num_chunks=20000,
        box_v=1.01,
        output_type="trimesh",
        enable_pbar=False,
        callback=cb,
        callback_steps=1,
    )
    print(f"    采样 {time.time() - t0:.1f}s  捕获 {len(traj)} 步")
    mesh = out[0][0] if isinstance(out[0], list) else out[0]
    print(f"    官方 mesh: V={len(mesh.vertices)} F={len(mesh.faces)} "
          f"watertight={mesh.is_watertight}")
    del pipe
    torch.cuda.empty_cache()
    return traj, len(mesh.faces)


def run_self_with_fixed_x0(x0, cond16):
    """自研 inline 采样循环（逐步记录，与 sample_loop 相同数学）。"""
    print("\n[2] 自研循环（同一 x0，逐步记录）...")
    dit = build_dit().to(DEV, dtype=torch.float16)
    sigmas = _sigmas(STEPS, 1.0)
    uncond = torch.zeros_like(cond16)
    x = x0.clone()
    x2 = torch.cat([x, x], dim=0)
    cond2 = torch.cat([cond16, uncond], dim=0)
    traj = [x[0].float().cpu().numpy()]
    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            t = torch.tensor([sigma.item()], device=DEV, dtype=x.dtype)
            t2 = t.expand(2)
            out = dit(x2, t2, cond2)
            out_cond, out_uncond = out.chunk(2)
            model_out = out_uncond + GS * (out_cond - out_uncond)
            x = x + (sigma_next - sigma) * model_out
            x2 = torch.cat([x, x], dim=0)
            traj.append(x[0].float().cpu().numpy())
    print(f"    采样 {time.time() - t0:.1f}s")
    del dit
    torch.cuda.empty_cache()
    return traj


def main():
    torch.backends.cuda.matmul.allow_tf32 = True

    # 固定 x0（两边完全相同）
    torch.manual_seed(SEED)
    x0 = torch.randn(1, 4096, LATENT_DIM, device=DEV, dtype=torch.float16)

    # 官方轨�迹 + mesh
    traj_off, f_off = run_official_with_fixed_x0(x0)

    # 官方最终 latent（traj 最后一步）
    lat_off_final = torch.from_numpy(traj_off[-1])

    # 自研轨迹：cond 用官方（stage6_official_cond.npz）
    d = np.load(os.path.join(HERE, "stage6_official_cond.npz"))
    cond16 = torch.from_numpy(d["cond"]).to(DEV, torch.float16)
    traj_self = run_self_with_fixed_x0(x0, cond16)

    # 逐步对比（traj_off[i] 是第 i 步后的 latent；traj_self[0] 是 x0）
    print("\n[3] 逐步对比（同 x0、同官方 cond）")
    n = min(len(traj_off), len(traj_self) - 1)
    print(f"    {'step':>4} {'cos':>10} {'maxdiff':>10} "
          f"{'off_std':>8} {'self_std':>8}")
    for i in range(n):
        a = torch.from_numpy(traj_self[i + 1]).flatten()
        b = torch.from_numpy(traj_off[i]).flatten()
        cos = F.cosine_similarity(a[None], b[None]).item()
        dmax = (a - b).abs().max().item()
        print(f"    {i:>4} {cos:>10.6f} {dmax:>10.4f} "
              f"{b.std():>8.4f} {a.std():>8.4f}")

    np.savez(OUT_NPZ,
             **{f"off_{i}": traj_off[i] for i in range(len(traj_off))},
             **{f"self_{i}": traj_self[i] for i in range(len(traj_self))})
    print(f"\n    轨迹已保存 {OUT_NPZ}")

    # 交叉验证：官方最终 latent → 自研 VAE/MC
    print("\n[4] 官方 latent → 自研 VAE/MC（交叉验证）")
    vae = build_vae().cpu()
    grid = decode_to_grid(vae, lat_off_final.unsqueeze(0),
                          octree_resolution=RES, num_chunks=20000, verbose=False)
    v, f = extract_surface_mc(grid[0], mc_level=0.0, bounds=1.01,
                              octree_resolution=RES)
    s = mesh_stats(v, f)
    print(f"    mesh: V={s['n_vertices']} F={s['n_faces']} "
          f"watertight={s.get('watertight')}")
    print(f"\n[summary] 官方 mesh F={f_off}；官方latent经自研VAE F={s['n_faces']}")


if __name__ == "__main__":
    main()
