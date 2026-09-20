# -*- coding: utf-8 -*-
"""阶段 6 · conditioner 严格对齐 + 自研内核全链路（E 段修正版 v2）。

v1 结论：
  - 官方 pipeline 同配置产出 V=5466 F=11148 watertight（硬件/配置非瓶颈）
  - [A] 官方(fp16) vs 自研(fp32) cond: cosine=0.999946 maxdiff=0.768
    —— 混杂了 dtype 差异，无法定位预处理是否精确复刻
  - [B] 自研采样 1413.7s 完成，但 VAE 解码因 frequencies 设备 bug 崩溃，
    latent 未保存，全部丢失

v2 改动：
  - [A] 直接构建官方 conditioner（fp32）与自研 fp32 逐位对比，
    排除 dtype 干扰，验证预处理复刻精确性
  - [B] 采样后立即落盘 latent npz（防丢失），解码改 CPU fp32
    （C/D 检查已验证 CPU 解码路径与官方一致）
  - [C] GLB 导出

运行：python research_kernel_3d/stage6_self_e2e.py
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
from stage4_sampler import sample_loop
from stage5_vae import build_and_load as build_vae, decode_to_grid
from stage6_surface import extract_surface_mc, mesh_stats
from stage6_self_e2e_v1 import official_preprocess, dino_transform

RES = 32
STEPS = 10
GS = 5.0
SEED = 42
IMG = os.path.join(ROOT, "output", "aigc_img_00002_.png")
CONFIG_YAML = os.path.join(ROOT, "models", "hy3dgen", "tencent",
                           "Hunyuan3D-2.1", "hunyuan3d-dit-v2-1",
                           "config.yaml")
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


def build_official_conditioner_fp32():
    """直接从 config + ckpt 构建官方 conditioner（fp32，不走 pipeline）。"""
    _vendor_path()
    import yaml
    from hy3dshape.utils import instantiate_from_config
    from safetensors.torch import load_file

    with open(CONFIG_YAML) as f:
        cfg = yaml.safe_load(f)
    model = instantiate_from_config(cfg["conditioner"])
    sd = load_file(CKPT, device="cpu")
    cond_sd = {k[len("conditioner."):]: v for k, v in sd.items()
               if k.startswith("conditioner.")}
    missing, unexpected = model.load_state_dict(cond_sd, strict=True)
    model.eval()
    return model


def check_cond_align(img_pm1):
    print("[A] conditioner 严格对齐（官方 fp32 vs 自研 fp32，同预处理）")
    off_model = build_official_conditioner_fp32()
    with torch.no_grad():
        cond_off = off_model(image=img_pm1)["main"]        # [1,1370,1024] fp32

    px = dino_transform(img_pm1.cpu())                    # [1,3,518,518]
    self_model, _ = load_conditioner(device="cpu")
    with torch.no_grad():
        cond_self = self_model(px)                        # [1,1370,1024] fp32

    cos = F.cosine_similarity(cond_off.flatten()[None],
                              cond_self.flatten()[None]).item()
    d = (cond_off - cond_self).abs().max().item()
    print(f"    官方: {tuple(cond_off.shape)} tok_norm={cond_off.norm(dim=-1).mean():.3f}")
    print(f"    自研: {tuple(cond_self.shape)} tok_norm={cond_self.norm(dim=-1).mean():.3f}")
    print(f"    cosine={cos:.8f}  maxdiff={d:.2e}")
    ok = cos > 0.999999 and d < 1e-3
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    del off_model, self_model
    return ok, cond_off


def self_e2e(cond):
    print("\n[B] 自研内核全链路（官方 cond → 自研 DiT 采样 → 自研 VAE/MC）")
    dit = build_dit().to(DEV, dtype=torch.float16)
    torch.manual_seed(SEED)
    x0 = torch.randn(1, 4096, LATENT_DIM, device=DEV, dtype=torch.float16)
    t0 = time.time()
    lat = sample_loop(dit, x0, cond.to(DEV, torch.float16),
                      num_steps=STEPS, shift=1.0, guidance_scale=GS)
    print(f"    采样 {time.time() - t0:.1f}s  latent mean={lat.mean():.4f} "
          f"std={lat.std():.4f}")
    # 立即落盘，防丢失
    lat_npz = os.path.join(HERE, "stage6_self_latent.npz")
    np.savez(lat_npz, latent=lat[0].float().cpu().numpy())
    print(f"    latent 已保存 {lat_npz}")
    del dit, x0
    torch.cuda.empty_cache()

    # 解码走 CPU fp32（避开 frequencies 设备 bug；C/D 已验证 CPU 路径与官方一致）
    print("    [decode] CPU fp32 ...")
    t0 = time.time()
    vae = build_vae().cpu()
    lat_cpu = lat[0].float().cpu().unsqueeze(0)
    grid = decode_to_grid(vae, lat_cpu, octree_resolution=RES,
                          num_chunks=20000, verbose=False)
    n_pos = int((grid[0] > 0).sum())
    print(f"    解码 {time.time() - t0:.1f}s  grid mean={grid.mean():.4f} "
          f"std={grid.std():.4f} 正体素={n_pos}")

    v, f = extract_surface_mc(grid[0], mc_level=0.0, bounds=1.01,
                              octree_resolution=RES)
    s = mesh_stats(v, f)
    print(f"    mesh: V={s['n_vertices']} F={s['n_faces']} "
          f"watertight={s.get('watertight')} euler={s.get('euler')}")

    npz = os.path.join(HERE, "stage6_self_e2e.npz")
    np.savez(npz, grid=grid[0].numpy(), vertices=v, faces=f)
    print(f"[C] 已保存 {npz}")
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        glb = os.path.join(ROOT, "output", "stage6_self_e2e.glb")
        mesh.export(glb)
        print(f"[C] 已导出 {glb}")
    except Exception as exc:
        print(f"[C] GLB 导出失败: {exc}")
    ok = s["n_faces"] > 1000 and s.get("watertight") is True
    print(f"\n[verdict] {'PASS（自研内核端到端出真实形状）' if ok else 'FAIL'}")
    return ok


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    img_pm1 = official_preprocess(IMG)
    ok_align, cond_off = check_cond_align(img_pm1)
    ok_e2e = self_e2e(cond_off)
    print(f"\n[summary] cond_align={ok_align}  self_e2e={ok_e2e}")


if __name__ == "__main__":
    main()
