# -*- coding: utf-8 -*-
"""自研 3D 内核 · 阶段 6 端到端验证（严格对齐官方 pipeline）。

四段逐级对照，任何一段不一致都能定位：
  A. conditioner：自研 DINOv2 vs 官方 pipe.conditioner（同输入图）
  B. 采样：自研 sample_loop vs 官方 scheduler+model 循环（同 x0/cond/步数/CFG）
  C. 体积解码：自研 decode_to_grid vs 官方 volume_decoder
  D. 表面提取：自研 skimage MC vs 官方 MCSurfaceExtractor

关键修正（本脚本验证）：
  官方 sigmas = linspace(0, 1, steps)，时间轴 **从 0（噪声）升到 1**，
  且默认 CFG guidance_scale=5.0，cond 为 cat([cond, uncond])。
  阶段 4 旧实现方向相反 → 解码出全负场（无形状），已修正。
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from stage2_conditioner import load_conditioner, preprocess
from stage3_dit import build_and_load as build_dit, LATENT_DIM
from stage4_sampler import sample_loop, _sigmas
from stage5_vae import build_and_load as build_vae, decode_to_grid
from stage6_surface import extract_surface_mc, mesh_stats

CONFIG_YAML = os.path.join(ROOT, "models", "hy3dgen", "tencent",
                           "Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", "config.yaml")
SHAPE_CKPT = os.path.join(ROOT, "models", "diffusion_models",
                          "hunyuan_3d_v2.1.safetensors")
IMG = os.path.join(ROOT, "output", "aigc_img_00002_.png")

RES = 64
STEPS = 10
SEED = 42
GS = 5.0


def load_image_518(path):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((518, 518), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def official_pipe():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    vendor = os.path.join(ROOT, "vendor", "Hunyuan3D-2.1-main", "hy3dshape")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt_path=SHAPE_CKPT, config_path=CONFIG_YAML,
        device="cpu", dtype=torch.float32, use_safetensors=True)
    pipe.model.eval()
    pipe.vae.eval()
    pipe.conditioner.eval()
    return pipe


def official_sample(pipe, x0, cond, num_steps=STEPS, gs=GS):
    """复刻官方 pipeline.__call__ 的采样循环（CPU）。"""
    import numpy as np
    sched = pipe.scheduler
    sigmas_np = np.linspace(0, 1, num_steps)
    sched.set_timesteps(num_inference_steps=None, sigmas=sigmas_np, device="cpu")
    timesteps = sched.timesteps
    uncond = torch.zeros_like(cond)
    x2 = torch.cat([x0, x0], dim=0)
    cond2 = torch.cat([cond, uncond], dim=0)
    lat = x0.clone()
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            tin = (t.expand(x2.shape[0]) / sched.config.num_train_timesteps).to(lat.dtype)
            npred = pipe.model(x2, tin, {'main': cond2})
            nc, nu = npred.chunk(2)
            npred = nu + gs * (nc - nu)
            lat = sched.step(npred, t, lat).prev_sample
    return lat


def main():
    print(f"[cfg] res={RES} steps={STEPS} gs={GS} seed={SEED}")
    img = load_image_518(IMG)
    px = preprocess(img)                       # [1,3,518,518] ImageNet 归一化
    print(f"[0] image {img.shape} mean={img.mean():.3f}")

    # ---- A. conditioner ----
    print("[A] conditioner 对齐 ...")
    cond_model, _ = load_conditioner(device="cuda")
    with torch.no_grad():
        cond_mine = cond_model(px.cuda()).cpu()
    pipe = official_pipe()
    raw = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)   # [1,3,518,518] in [0,1]
    with torch.no_grad():
        cond_off = pipe.conditioner(image=raw, value_range=None)['main']
    dA = (cond_mine - cond_off).abs().max().item()
    print(f"    cond {tuple(cond_mine.shape)} maxdiff={dA:.2e}")

    # ---- B. 采样 ----
    print(f"[B] 采样对齐（{STEPS} 步, CFG={GS}）...")
    torch.manual_seed(SEED)
    x0 = torch.randn(1, 4096, LATENT_DIM)
    dit = build_dit().cuda()
    lat_mine = sample_loop(dit, x0.cuda(), cond_mine.cuda(), num_steps=STEPS,
                           shift=1.0, guidance_scale=GS).cpu()
    print(f"    mine: mean={lat_mine.mean():.4f} std={lat_mine.std():.4f}")
    lat_off = official_sample(pipe, x0, cond_off, num_steps=STEPS, gs=GS)
    print(f"    off:  mean={lat_off.mean():.4f} std={lat_off.std():.4f}")
    dB = (lat_mine - lat_off).abs().max().item()
    cosB = torch.nn.functional.cosine_similarity(
        lat_mine.flatten()[None], lat_off.flatten()[None]).item()
    print(f"    latent maxdiff={dB:.4e} cos={cosB:.6f}")

    # ---- C. 体积解码 ----
    print(f"[C] 体积解码 r={RES} ...")
    vae_mine = build_vae().cuda()
    grid_mine = decode_to_grid(vae_mine, lat_mine.cuda(), octree_resolution=RES,
                               num_chunks=20000, verbose=False).cpu()
    with torch.no_grad():
        grid_off = pipe.vae.volume_decoder(
            pipe.vae(lat_off / pipe.vae.scale_factor), pipe.vae.geo_decoder,
            bounds=1.01, octree_resolution=RES, num_chunks=20000,
            enable_pbar=False)
    dC = (grid_mine - grid_off).abs().max().item()
    n_pos = int((grid_mine[0] > 0).sum())
    print(f"    grid {tuple(grid_mine.shape)} mean={grid_mine.mean():.4f} "
          f"std={grid_mine.std():.4f} 正体素={n_pos}")
    print(f"    grid maxdiff={dC:.2e}")

    # ---- D. 表面提取 ----
    print("[D] 表面提取 ...")
    v, f = extract_surface_mc(grid_mine[0], mc_level=0.0, bounds=1.01,
                              octree_resolution=RES)
    s = mesh_stats(v, f)
    print(f"    mine mesh: V={s['n_vertices']} F={s['n_faces']} "
          f"watertight={s.get('watertight')} euler={s.get('euler')} "
          f"volume={s.get('volume')}")

    from hy3dshape.models.autoencoders.surface_extractors import MCSurfaceExtractor
    out_off = MCSurfaceExtractor()(grid_off, mc_level=0.0, bounds=1.01,
                                   octree_resolution=RES)[0]
    v_off, f_off = out_off.mesh_v, out_off.mesh_f
    same_v = v.shape == v_off.shape
    dD = float(np.abs(v - v_off).max()) if same_v else float("nan")
    same_f = (f.shape == f_off.shape) and bool((f == f_off).all())
    print(f"    off  mesh: V={v_off.shape[0]} F={f_off.shape[0]}")
    print(f"    vertices maxdiff={dD:.2e} faces identical={same_f}")

    # ---- 保存 ----
    np.savez(os.path.join(HERE, "stage6_e2e.npz"),
             grid=grid_mine[0].numpy(), vertices=v, faces=f,
             latent=lat_mine[0].numpy())
    print("[saved] stage6_e2e.npz")

    print("\n[summary] A=%.2e B=%.2e C=%.2e D=%.2e" % (dA, dB, dC, dD))
    ok = (dA < 1e-4) and (dB < 1e-3) and (dC < 1e-4) and same_v and dD < 1e-4 and same_f
    print(f"[verdict] {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
