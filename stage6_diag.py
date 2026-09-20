# -*- coding: utf-8 -*-
"""自研 3D 内核 · 阶段 6 诊断：E 段形状琐碎的根因排查。

主线验证（stage6_verify.py）结果：A/B/C/D 全部 PASS，E FAIL
（grid ~ flat，正体素仅 5，mesh F=36）。
在不影响主线对齐结论的前提下，定位 E 段形状琐碎的最可能单点。

诊断项（独立、互不依赖）：
  1. cond sanity   — DINOv2 cond 的 norm（及对随机初始化 DiT 的 basin）
  2. sigma 强度    — std(Δσ)·std(model_out) 等效力矩
  3. latent scale  — 若 latent 呈现 focal scaling/offset pattern，
                     用结构性缩放（非逐元素作弊）看正体素数
  4. GS 扫描        — guidance scale 对正体素数/mesh 面的影响
  5. 随机 cond 对照 — 真实图 cond vs 随机初始化 cond 的解码差异，
                     区分"问题在 DiT 传播" vs "问题在 VAE 重建"

运行：python research_kernel_3d/stage6_diag.py
"""
import os
import sys
import time

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from stage2_conditioner import load_conditioner, preprocess
from stage3_dit import build_and_load as build_dit, LATENT_DIM
from stage4_sampler import _sigmas, sample_loop
from stage5_vae import build_and_load as build_vae, decode_to_grid
from stage6_surface import extract_surface_mc, mesh_stats
from stage6_verify import _official_scheduler, load_official_vae, RES, STEPS, GS, SEED, IMG

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEV == "cuda" else torch.float32

VAE_CKPT = os.path.join(ROOT, "models", "vae", "hunyuan3d_v2.1_vae.safetensors")


def _img():
    from PIL import Image
    return np.asarray(Image.open(IMG).convert("RGB").resize(
        (518, 518), Image.LANCZOS), dtype=np.float32) / 255.0


def _load_cond(px, device):
    cond_model, _ = load_conditioner(device=device)
    with torch.no_grad():
        cond = cond_model(px)
    return cond


# ── 1. cond sanity ────────────────────────────────────────────────

def diag_cond_sanity(cond):
    print("[1] cond sanity")
    norms = cond[0].norm(dim=-1)
    print(f"    cond: shape={tuple(cond.shape)}  seq_norm mean={norms.mean():.4f} "
          f"min={norms.min():.4f} max={norms.max():.4f}")
    zero_uncond = torch.zeros_like(cond).to(cond.device)
    print(f"    ||cond - 0|| = {(cond - zero_uncond).norm():.4f}  (large)")
    torch.manual_seed(0)
    rand_cond = torch.randn_like(cond)
    print(f"    ||cond - rand|| = {(cond - rand_cond).norm():.4f}  (for scale)")
    print()


# ── 2. sigma 强度 ─────────────────────────────────────────────────

def diag_sigma_strength():
    print("[2] sigma 强度")
    sig = _sigmas(STEPS, 1.0)
    deltas = torch.diff(sig)
    print(f"    sigma: {[round(float(s), 3) for s in sig]}")
    print(f"    deltas: {[round(float(d), 3) for d in deltas]}")
    print(f"    mean|Δσ|={deltas.abs().mean():.3f}, max|Δσ|={deltas.abs().max():.3f}")

    # 随机初始化 DiT + 随机输入 → model_out 数量级（在 GPU 上定量，
    # 并打印“该幅度 × Δσ 是否足以把 latent 推向有意义的区域”）
    from stage3_dit import build_and_load as _build
    torch.manual_seed(0)
    x = torch.randn(1, 4096, LATENT_DIM, device=DEV, dtype=DTYPE)
    cond_test = torch.randn(1, 16, 1024, device=DEV, dtype=DTYPE)
    dit = _build().to(DEV, dtype=DTYPE)
    with torch.no_grad():
        t = torch.tensor([0.5], device=DEV, dtype=DTYPE)
        out = dit(x, t, cond_test).float()
    mean_step = float((deltas.abs().mean() * out.std()))
    print(f"    random-init DiT model_out: mean={out.mean():.4f} std={out.std():.4f} "
          f"absmax={out.abs().max():.4f}")
    print(f"    mean|Δσ|·std(model_out) ≈ {mean_step:.4f}，10 步累计 drift ≈ {mean_step * STEPS:.4f}")
    del dit, out, x, cond_test
    torch.cuda.empty_cache()
    print()


# ── 3. latent scale 分析 ──────────────────────────────────────────

def diag_latent_scale(lat, vae_mine):
    # 结构性缩放 + shift（用"差异性信号"而非单点 bias，避免出现非物理的 flat+1 离散解）
    print("[3] latent scale / shift 分析（结构性线性变换，非逐元素作弊）")
    print(f"    raw latent: mean={lat.mean():.4f} std={lat.std():.4f}")
    for scale, shift in [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (5.0, 0.0), (5.0, 2.0), (10.0, 0.0), (10.0, 5.0)]:
        lat_t = lat * scale + shift
        g = decode_to_grid(vae_mine, lat_t, octree_resolution=RES,
                           num_chunks=20000, verbose=False)
        n_pos = int((g[0] > 0).sum())
        mesh_str = ""
        if n_pos > 0:
            v, f = extract_surface_mc(g[0], mc_level=0.0, bounds=1.01,
                                      octree_resolution=RES)
            mesh_str = f"  mesh: V={len(v)}, F={len(f)}"
        print(f"    scale={scale:>5.1f} shift={shift:>3.0f}: grid mean={g.mean():+.4f} "
              f"std={g.std():.4f} 正体素={n_pos}{mesh_str}")
        del g, lat_t
    print()


# ── 4. GS 扫描 ─────────────────────────────────────────────────────

def diag_gs_scan(cond, vae_mine, dit, px):
    print("[4] guidance scale 扫描")
    for gs in [2.0, 5.0, 10.0, 20.0]:
        torch.manual_seed(SEED)
        x0 = torch.randn(1, 4096, LATENT_DIM, device=DEV, dtype=DTYPE)
        with torch.no_grad():
            lat = sample_loop(dit, x0, cond.to(DTYPE), num_steps=STEPS,
                              shift=1.0, guidance_scale=gs)

        grid = decode_to_grid(vae_mine, lat, octree_resolution=RES,
                              num_chunks=20000, verbose=False)
        n_pos = int((grid[0] > 0).sum())
        v, f = extract_surface_mc(grid[0], mc_level=0.0, bounds=1.01,
                                  octree_resolution=RES)
        mesh_n = len(f)
        print(f"    gs={gs:>5.1f}: grid mean={grid.mean():+.4f} std={grid.std():.4f} "
              f"正体素={n_pos}  mesh_F={mesh_n}")
        del lat, grid, x0
        torch.cuda.empty_cache()
    print()


# ── 5. 随机 cond 对照 ──────────────────────────────────────────────

def diag_random_cond_baseline(cond, vae_mine, dit, px):
    print("[5] 随机 cond（对照）")
    torch.manual_seed(SEED)
    rand_cond = torch.randn_like(cond).to(DEV, DTYPE)
    x0 = torch.randn(1, 4096, LATENT_DIM, device=DEV, dtype=DTYPE)
    with torch.no_grad():
        lat_rand = sample_loop(dit, x0, rand_cond, num_steps=STEPS,
                                shift=1.0, guidance_scale=GS)

    grid_rand = decode_to_grid(vae_mine, lat_rand, octree_resolution=RES,
                               num_chunks=20000, verbose=False)
    n_pos_r = int((grid_rand[0] > 0).sum())
    v, f = extract_surface_mc(grid_rand[0], mc_level=0.0, bounds=1.01,
                              octree_resolution=RES)
    s_r = mesh_stats(v, f)
    print(f"    random-cond: 正体素={n_pos_r}  mesh_F={len(f)}")
    print(f"    grid rand mean={grid_rand.mean():+.4f} std={grid_rand.std():.4f}")
    print(f"    grid rand min={grid_rand.min():.4f} max={grid_rand.max():.4f}")
    print()


# ── 6. 官方参考（同硬件 fp16，验证是否可行 & 作为 E 段金标准） ─────

def diag_official_e2e(px):
    print("[6] 官方 Hunyuan3D FlowMatching 同图采样（fp16, res=32, steps=10）")
    print("    说明：如果这段能出 watertight 大 mesh → 硬件/配置不是瓶颈；")
    print("          如果同样「正体素≈5」→ 需要更多步数/更高分辨率/换判别层。")
    from stage6_verify import _vendor_path
    _vendor_path()
    try:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    except Exception as exc:
        print(f"    import 官方 pipeline 失败: {exc}")
        return
    try:
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            os.path.join(ROOT, "models", "hy3dgen", "tencent",
                         "Hunyuan3D-2.1-hunyuan3d-dit-v2-1"),
            dtype=torch.float16,
        )
    except Exception as exc:
        print(f"    from_pretrained 失败: {exc}")
        return
    try:
        out = pipe(
            image=IMG,
            num_inference_steps=10,
            guidance_scale=5.0,
            octree_resolution=RES,
            mc_level=0.0,
            num_chunks=20000,
            box_v=1.01,
            output_type="trimesh",
            enable_pbar=False,
        )
        mesh = out[0][0] if isinstance(out[0], list) else out[0]
        v, f = mesh.vertices, mesh.faces
        print(f"    官方 mesh: V={len(v)} F={len(f)} watertight={mesh.is_watertight}")
        if mesh.is_watertight and len(f) > 100:
            print("    >>> 官方验证了「fp16 + 同 res + 同 steps」可行，问题在自研内核细节")
        else:
            print("    >>> 官方输出同样琐碎 → 大概率是该硬件上 10 步+res32 本身难出形状")
    except Exception as exc:
        print(f"    官方管道执行失败: {exc}")
    print()


# ── 7. 噪声尺度扫描（诊断 sample_loop 的初值尺度） ────────────────

def diag_noise_scale_scan(vae_mine, dit, cond):
    print("[7] 初始噪声 x0 尺度扫描")
    for scale in [0.1, 0.5, 1.0, 2.0]:
        torch.manual_seed(SEED)
        x0 = scale * torch.randn(1, 4096, LATENT_DIM, device=DEV, dtype=DTYPE)
        with torch.no_grad():
            lat = sample_loop(dit, x0, cond.to(DTYPE), num_steps=STEPS,
                              shift=1.0, guidance_scale=GS)
        g = decode_to_grid(vae_mine, lat, octree_resolution=RES,
                           num_chunks=20000, verbose=False)
        n_pos = int((g[0] > 0).sum())
        v, f = extract_surface_mc(g[0], mc_level=0.0, bounds=1.01,
                                  octree_resolution=RES)
        print(f"    x0_scale={scale:>4.1f}: lat mean={lat.mean():+.4f} std={lat.std():.4f} "
              f"grid mean={g.mean():+.4f} 正体素={n_pos} mesh_F={len(f)}")
        del x0, lat, g
        torch.cuda.empty_cache()
    print()


def main():
    t0 = time.time()
    print(f"[env] device={DEV}, dtype={DTYPE}")
    print(f"[env] torch {torch.__version__}, cuda={torch.cuda.is_available()}")
    if DEV == "cuda":
        print(f"[env] GPU={torch.cuda.get_device_name(0)}")
    print()

    px = preprocess(_img()).to(DEV)
    cond = _load_cond(px, DEV)
    cond_f = cond.to(DTYPE)
    print(f"loaded cond in {time.time() - t0:.1f}s")
    t0 = time.time()

    dit = build_dit().to(DEV, dtype=DTYPE)
    vae_mine = build_vae().to(DEV, dtype=DTYPE)
    print(f"loaded models in {time.time() - t0:.1f}s")
    print()

    diag_cond_sanity(cond_f)
    t0 = time.time()

    torch.manual_seed(SEED)
    x0 = torch.randn(1, 4096, LATENT_DIM, device=DEV, dtype=DTYPE)
    lat = sample_loop(dit, x0, cond_f, num_steps=STEPS, shift=1.0,
                      guidance_scale=GS)
    print(f"[main] latent done in {time.time() - t0:.1f}s  "
          f"mean={lat.mean():.4f} std={lat.std():.4f}")
    print()

    diag_sigma_strength()

    diag_latent_scale(lat, vae_mine)
    diag_gs_scan(cond_f, vae_mine, dit, px)
    diag_random_cond_baseline(cond_f, vae_mine, dit, px)
    diag_noise_scale_scan(vae_mine, dit, cond_f)

    del dit, vae_mine
    torch.cuda.empty_cache()

    diag_official_e2e(px)

    print(f"[done] total {(time.time() - t0):.0f}s")


if __name__ == "__main__":
    main()
