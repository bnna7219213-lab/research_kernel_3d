# -*- coding: utf-8 -*-
"""阶段 6 · conditioner 对齐 + 自研内核全链路（E 段修正版）。

背景（stage6_official_e2e.py 实测）：
  官方 pipeline 同配置（fp16/res32/10步/GS=5）产出 V=5466 F=11148
  watertight 形状 → 硬件与配置不是瓶颈，自研内核存在细节差异。

主嫌疑：条件准备。自研 E 段原先用 PIL LANCZOS 直接缩 518；官方是
  recenter(border 0.15, 白底) → 512 cubic → [-1,1] →
  DinoImageEncoder 内部 (x+1)/2 → Resize(518 bilinear) + ImageNet norm。

本脚本：
  [A] 复刻官方预处理 → 自研 conditioner vs 官方 conditioner 逐位对齐
  [B] 用官方 cond 驱动自研 DiT+采样器+VAE+MC 全链路（GPU fp16）
  [C] 保存 npz + GLB（供 Blender 无头 QA）

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

RES = 32
STEPS = 10
GS = 5.0
SEED = 42
IMG = os.path.join(ROOT, "output", "aigc_img_00002_.png")

DEV = "cuda"


def _vendor_path():
    vendor = os.path.join(ROOT, "vendor", "Hunyuan3D-2.1-main", "hy3dshape")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    return vendor


# ───────────── 官方预处理复刻（preprocessors.py ImageProcessorV2） ─────────────

def official_preprocess(img_path, border_ratio=0.15, size=512):
    """recenter + resize512 + [-1,1] tensor [1,3,512,512]。"""
    import cv2
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # recenter（与 ImageProcessorV2.recenter 相同逻辑）
    if image.shape[-1] == 4:
        mask = image[..., 3]
    else:
        mask = np.ones_like(image[..., 0:1]) * 255
        image = np.concatenate([image, mask], axis=-1)
        mask = mask[..., 0]
    H, W, C = image.shape
    s = max(H, W)
    result = np.zeros((s, s, C), dtype=np.uint8)
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h, w = x_max - x_min, y_max - y_min
    desired = int(s * (1 - border_ratio))
    scale = desired / max(h, w)
    h2, w2 = int(h * scale), int(w * scale)
    x2_min = (s - h2) // 2
    y2_min = (s - w2) // 2
    result[x2_min:x2_min + h2, y2_min:y2_min + w2] = cv2.resize(
        image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
    bg = np.ones((s, s, 3), dtype=np.uint8) * 255
    m = result[..., 3:].astype(np.float32) / 255
    result = (result[..., :3] * m + bg * (1 - m)).clip(0, 255).astype(np.uint8)
    # BGR→RGB + resize 512 cubic
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = cv2.resize(result, (size, size), interpolation=cv2.INTER_CUBIC)
    # [-1,1] tensor
    t = torch.tensor(result).float() / 255 * 2 - 1      # [512,512,3]
    return t.permute(2, 0, 1).unsqueeze(0)              # [1,3,512,512]


def dino_transform(img_pm1, image_size=518):
    """DinoImageEncoder.forward 的 value_range + transforms 复刻。

    value_range=(-1,1): image = (image - low) / (high - low) → [0,1]
    transform: Resize(518, bilinear, antialias) + CenterCrop(518) + Normalize
    """
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    img01 = (img_pm1 + 1) / 2
    tf = T.Compose([
        T.Resize(image_size, InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop(image_size),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img01)


# ───────────── A. conditioner 对齐 ─────────────

def check_cond_align():
    print("[A] conditioner 对齐（官方预处理 → 自研 DINOv2 vs 官方）")
    _vendor_path()
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    config_yaml = os.path.join(ROOT, "models", "hy3dgen", "tencent",
                               "Hunyuan3D-2.1", "hunyuan3d-dit-v2-1",
                               "config.yaml")
    ckpt = os.path.join(ROOT, "models", "diffusion_models",
                        "hunyuan_3d_v2.1.safetensors")
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt, config_yaml, device=DEV, dtype=torch.float16,
        use_safetensors=True)

    img_pm1 = official_preprocess(IMG).to(DEV)          # [1,3,512,512] in [-1,1]
    with torch.no_grad():
        cond_off = pipe.conditioner(image=img_pm1)["main"].float()  # [1,1370,1024]
    # 释放官方 pipeline（DiT 占显存）
    del pipe
    torch.cuda.empty_cache()

    px = dino_transform(img_pm1.cpu())                  # [1,3,518,518]
    cond_self_model, _ = load_conditioner(device=DEV)
    with torch.no_grad():
        cond_self = cond_self_model(px.to(DEV)).float()

    cos = F.cosine_similarity(cond_off.flatten()[None],
                              cond_self.flatten()[None]).item()
    d = (cond_off - cond_self).abs().max().item()
    print(f"    官方 cond: {tuple(cond_off.shape)} norm={cond_off.norm(dim=-1).mean():.2f}")
    print(f"    自研 cond: {tuple(cond_self.shape)} norm={cond_self.norm(dim=-1).mean():.2f}")
    print(f"    cosine={cos:.6f}  maxdiff={d:.2e}")
    ok = cos > 0.9999 and d < 1e-2
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    del cond_self_model
    torch.cuda.empty_cache()
    return ok, cond_off


# ───────────── B/C. 自研内核全链路 ─────────────

def self_e2e(cond_off):
    print("\n[B] 自研内核全链路（官方 cond → 自研 DiT/采样/VAE/MC）")
    t0 = time.time()
    dit = build_dit().to(DEV, dtype=torch.float16)
    vae = build_vae().to(DEV, dtype=torch.float16)
    print(f"    模型加载 {time.time() - t0:.1f}s")

    torch.manual_seed(SEED)
    x0 = torch.randn(1, 4096, LATENT_DIM, device=DEV, dtype=torch.float16)
    t0 = time.time()
    lat = sample_loop(dit, x0, cond_off.to(torch.float16),
                      num_steps=STEPS, shift=1.0, guidance_scale=GS)
    print(f"    采样 {time.time() - t0:.1f}s  latent mean={lat.mean():.4f} "
          f"std={lat.std():.4f}")

    t0 = time.time()
    grid = decode_to_grid(vae, lat, octree_resolution=RES,
                          num_chunks=20000, verbose=False)
    n_pos = int((grid[0] > 0).sum())
    print(f"    解码 {time.time() - t0:.1f}s  grid mean={grid.mean():.4f} "
          f"std={grid.std():.4f} 正体素={n_pos}")

    v, f = extract_surface_mc(grid[0].float().cpu(), mc_level=0.0,
                              bounds=1.01, octree_resolution=RES)
    s = mesh_stats(v, f)
    print(f"    mesh: V={s['n_vertices']} F={s['n_faces']} "
          f"watertight={s.get('watertight')} euler={s.get('euler')}")

    # C. 落盘
    npz = os.path.join(HERE, "stage6_self_e2e.npz")
    np.savez(npz, grid=grid[0].float().cpu().numpy(),
             vertices=v, faces=f, latent=lat[0].float().cpu().numpy())
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


def _main_v1():
    """v1 主流程（已弃用：VAE GPU 解码有设备 bug、cond 对比混入 fp16 噪声），
    保留仅供追溯，见 stage6_self_e2e_v2.py。"""
    torch.backends.cuda.matmul.allow_tf32 = True
    ok_align, cond_off = check_cond_align()
    ok_e2e = self_e2e(cond_off)
    print(f"\n[summary] cond_align={ok_align}  self_e2e={ok_e2e}")


if __name__ == "__main__":
    _main_v1()
