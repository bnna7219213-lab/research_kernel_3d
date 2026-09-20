# -*- coding: utf-8 -*-
"""自研 3D 内核 · 阶段 5 对齐诊断：ShapeVAE 解码端 vs 官方库。

对照项（同输入 latent/查询点）：
  1. decode 通路：post_kl + transformer（16 层）输出  [B,N,1024]
     - 同时验证 scale_factor：官方 _export 先做 latents /= scale_factor 再进 vae
  2. geo_decoder 通路：query_proj(fourier(queries)) -> cross_attn -> ln_post -> output_proj
  3. 合并判定：cosine / max_abs_diff

官方库仅在本文件内作对照，不进自研内核模块。
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from stage5_vae import (build_and_load, NUM_LATENTS, EMBED_DIM, SCALE_FACTOR,
                        VAE_CKPT)

CONFIG_YAML = os.path.join(ROOT, "models", "hy3dgen", "tencent",
                           "Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", "config.yaml")
SHAPE_CKPT = os.path.join(ROOT, "models", "diffusion_models",
                          "hunyuan_3d_v2.1.safetensors")


def _cos_diff(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    cos = F.cosine_similarity(a[None], b[None]).item()
    diff = (a - b).abs().max().item()
    return cos, diff


def _load_official_vae():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    vendor = os.path.join(ROOT, "vendor", "Hunyuan3D-2.1-main", "hy3dshape")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt_path=SHAPE_CKPT, config_path=CONFIG_YAML,
        device="cpu", dtype=torch.float32, use_safetensors=True)
    pipe.vae.eval()
    return pipe.vae


def main():
    torch.manual_seed(42)
    N = NUM_LATENTS
    lat_raw = torch.randn(1, N, EMBED_DIM)
    queries = torch.rand(1, 512, 3) * 2.02 - 1.01

    print("[1] 自研 ShapeVAE 解码端 ...")
    mine = build_and_load()

    print("[2] 官方 ShapeVAE ...")
    off = _load_official_vae()

    # ---- 对齐 A: decode 通路（不做 scale 除法） ----
    with torch.no_grad():
        dec_mine = mine.decode(lat_raw)
        dec_off = off.decode(lat_raw)   # post_kl + transformer
    cos, diff = _cos_diff(dec_mine, dec_off)
    print(f"[align-A] decode(post_kl+transformer) cos={cos:.6f} maxdiff={diff:.6f}")

    # ---- 对齐 B: geo_decoder 占用查询 ----
    with torch.no_grad():
        occ_mine = mine.query_occ(queries, dec_mine)
        occ_off = off.geo_decoder(queries=queries, latents=dec_off)
    cos2, diff2 = _cos_diff(occ_mine, occ_off)
    print(f"[align-B] geo_decoder(queries)        cos={cos2:.6f} maxdiff={diff2:.6f}")

    # ---- 对齐 C: scale_factor 惯例验证 ----
    # 官方 _export: latents = latents / scale_factor 再 vae(latents)=decode
    # 即若输入是采样器输出的 latent，应先除 scale_factor。
    lat_scaled = lat_raw / SCALE_FACTOR
    with torch.no_grad():
        dec_off_scaled = off.decode(lat_scaled)
    cos3, diff3 = _cos_diff(mine.decode(lat_scaled), dec_off_scaled)
    print(f"[align-C] decode(latents/scale)       cos={cos3:.6f} maxdiff={diff3:.6f}")

    verdict = "PASS" if min(cos, cos2, cos3) > 0.999 else "FAIL"
    print(f"\n[verdict] {verdict}")


if __name__ == "__main__":
    main()
