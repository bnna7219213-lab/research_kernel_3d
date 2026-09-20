# -*- coding: utf-8 -*-
"""自研 3D 内核 · 阶段 4：flow-matching 采样循环 + 与官方库单步对齐。

目标：从"自洽"走向"正确"——用官方库（vendor/Hunyuan3D-2.1）做参照，
验证自研 DiT 单步输出与官方一致，再跑完整采样循环。

官方采样器行为（vendor/Hunyuan3D-2.1/hy3dshape/schedulers.py，已读源码）：
  FlowMatchEulerDiscreteScheduler，shift=1.0 时 sigmas 不变（恒等变换）。
  欧拉步：prev_sample = sample + (sigma_next - sigma) * model_output
  sigmas 从 sigma_max 线性降到 sigma_min，末尾补 1.0。
  注意：官方 __call__ 里 timestep = t / num_train_timesteps（归一化到 [0,1]）。

对齐策略：
  1. 用官方库 from_single_file 加载同一权重，跑单步（t=0.5，随机 latent+cond）
  2. 用自研 stage3_dit 跑同输入单步
  3. 对比输出（cosine / 相对误差），定位差异来源（时间注入/位置编码/归一化）
  4. 修正后跑完整采样循环（10 步），输出 latent 统计

红线：不 import hy3dshape 到自研内核；官方库仅作对照（单独进程/函数）。
"""
import os
import sys
import json

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from stage3_dit import ShapeDiT, build_and_load, load_model_sd, HIDDEN, LATENT_DIM

CKPT = os.path.join(ROOT, "models", "diffusion_models",
                    "hunyuan_3d_v2.1.safetensors")
CONFIG_YAML = os.path.join(ROOT, "models", "hy3dgen", "tencent",
                           "Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", "config.yaml")


# ───────────── 官方库对照（单独函数，不混入自研内核） ─────────────

def _official_single_step(x, t, cond):
    """官方库单步前向（仅作对照）。返回 [B,N,64]。"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    vendor = os.path.join(ROOT, "vendor", "Hunyuan3D-2.1-main", "hy3dshape")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt_path=CKPT, config_path=CONFIG_YAML,
        device="cpu", dtype=torch.float32, use_safetensors=True)
    model = pipe.model
    model.eval()

    # 官方 forward 期望 contexts 是 dict（contexts['main']）
    # 且 t 是 [B] 标量（内部 t_embedder 会 unsqueeze）
    with torch.no_grad():
        out = model(x, t, {'main': cond})
    return out


# ───────────── 自研采样循环 ─────────────

def _sigmas(num_steps, shift=1.0, num_train_timesteps=1000):
    """官方 FlowMatchEuler 的 sigma 序列。

    官方 pipeline：sigmas = np.linspace(0, 1, num_inference_steps)
    然后 set_timesteps(sigmas=...) 里 shift=1 时恒等；self.sigmas = cat([sigmas, [1.0]])。
    注意：官方时间轴是 **从 0（噪声）升到 1**，与常见 diffusion（1→0）相反。
    """
    import numpy as np
    sigmas = np.linspace(0, 1, num_steps)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    sigmas = np.concatenate([sigmas, [1.0]])
    return torch.from_numpy(sigmas).float()


def sample_loop(model, x0, cond, num_steps=10, shift=1.0, seed=42,
                guidance_scale=5.0, uncond=None):
    """自研欧拉采样循环（严格复刻官方 pipeline）。

    官方 pipeline 循环（pipelines.py __call__）：
      sigmas = linspace(0,1,steps)；timesteps = sigmas*1000（t 从 0 增到 ~990）
      每步：timestep = t / num_train_timesteps  → 模型收到 t∈[0,1)
            CFG: latent_model_input = cat([latents]*2)
                 noise_pred = model(x2, t2, cond2)
                 noise_pred = uncond + gs * (cond - uncond)
            scheduler.step: prev = sample + (sigma_next - sigma) * noise_pred
    其中 cond2 = cat([cond, uncond])（官方 encode_cond 的 cat_recursive）。

    Args:
        x0: 初始噪声 [B,N,64]（官方 prepare_latents 用 randn * init_noise_sigma=1）
        cond: 条件 [B,1370,1024]（已含 CLS token）
        guidance_scale: CFG 强度（官方默认 5.0；<0 表示关闭）
        uncond: 无条件 embedding；None 时用官方约定 zeros_like
    """
    torch.manual_seed(seed)
    sigmas = _sigmas(num_steps, shift)
    x = x0.clone()
    use_cfg = guidance_scale >= 0
    if use_cfg:
        if uncond is None:
            uncond = torch.zeros_like(cond)
        x2 = torch.cat([x, x], dim=0)
        cond2 = torch.cat([cond, uncond], dim=0)

    for i in range(num_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        # 官方：t = sigma * 1000，模型收到 t / 1000 = sigma
        t = torch.tensor([sigma.item()], device=x.device, dtype=x.dtype)
        with torch.no_grad():
            if use_cfg:
                t2 = t.expand(2).to(x.dtype)
                out = model(x2, t2, cond2)
                out_cond, out_uncond = out.chunk(2)
                model_out = out_uncond + guidance_scale * (out_cond - out_uncond)
            else:
                model_out = model(x, t, cond)
        x = x + (sigma_next - sigma) * model_out
    return x


# ───────────── 对齐诊断 ─────────────

def _align_check():
    """自研 vs 官方单步对齐。"""
    torch.manual_seed(42)
    B, N, M = 1, 64, 16
    x = torch.randn(B, N, LATENT_DIM)
    t = torch.tensor([0.5])
    cond = torch.randn(B, M, 1024)

    print("[1] 自研单步 ...")
    model = build_and_load()
    with torch.no_grad():
        out_self = model(x, t, cond)
    print(f"    self: mean={out_self.mean():.4f} std={out_self.std():.4f} "
          f"absmax={out_self.abs().max():.4f}")

    print("[2] 官方单步 ...")
    out_official = _official_single_step(x, t, cond)
    print(f"    official: mean={out_official.mean():.4f} std={out_official.std():.4f} "
          f"absmax={out_official.abs().max():.4f}")

    # 对齐指标
    a = out_self.flatten().float()
    b = out_official.flatten().float()
    cos = F.cosine_similarity(a[None], b[None]).item()
    rel_err = ((a - b).abs() / (b.abs() + 1e-6)).mean().item()
    print(f"\n[align] cosine={cos:.4f}  rel_err={rel_err:.4f}")
    print(f"[align] 判定: {'PASS' if cos > 0.95 else 'FAIL'}")

    return {
        "self": {"mean": float(out_self.mean()), "std": float(out_self.std())},
        "official": {"mean": float(out_official.mean()), "std": float(out_official.std())},
        "cosine": cos, "rel_err": rel_err,
    }


def _full_sample():
    """完整采样循环（自研）。"""
    torch.manual_seed(42)
    B, N, M = 1, 64, 16
    x0 = torch.randn(B, N, LATENT_DIM)
    cond = torch.randn(B, M, 1024)

    print("[3] 自研完整采样（10 步）...")
    model = build_and_load()
    lat = sample_loop(model, x0, cond, num_steps=10, shift=1.0)
    print(f"    latent: mean={lat.mean():.4f} std={lat.std():.4f} "
          f"absmax={lat.abs().max():.4f} NaN={torch.isnan(lat).any()}")
    return lat


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", action="store_true", help="只做单步对齐")
    ap.add_argument("--full", action="store_true", help="只做完整采样")
    args = ap.parse_args()

    if args.align or not args.full:
        res = _align_check()
        with open(os.path.join(HERE, "stage4_align.json"), "w") as f:
            json.dump(res, f, indent=2)
    if args.full or not args.align:
        _full_sample()
