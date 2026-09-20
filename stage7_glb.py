# -*- coding: utf-8 -*-
"""自研 3D 内核 · 阶段 7：GLB 导出固化（端到端一键 CLI）。

阶段 6 结论：同 x0 下自研全链路 F=9920 watertight（官方 9890）。
本模块把「图 → cond → 采样 → VAE/MC → GLB」固化为可复用的管线：

  - prepare_latents：复刻 diffusers randn_tensor 语义
    （CPU generator → CPU 生成 → 搬移；无 generator → 全局流），
    同 seed 完全可复现，且与官方同 seed 的 x0 数值流一致。
  - run_pipeline：单函数端到端，参数齐全（steps/gs/res/seed/dtype）。
  - GLB 导出：官方 export_to_trimesh 语义（faces 翻转绕向 + 默认 process）
  - 自检：generator 确定性 + 采样循环确定性（同输入两次前向逐位一致）

用法：
  python research_kernel_3d/stage7_glb.py                 # 自检 + 默认全链路
  python research_kernel_3d/stage7_glb.py --steps 10 --seed 42 \
      --image path/to/img.png --out output/mesh.glb
"""
import argparse
import json
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

IMG_DEFAULT = os.path.join(ROOT, "output", "aigc_img_00002_.png")
OUT_DEFAULT = os.path.join(ROOT, "output", "stage7_glb.glb")


def prepare_latents(shape, seed=None, generator=None, device="cuda",
                    dtype=torch.float16):
    """diffusers randn_tensor 语义复刻。

    - generator 给定且在 CPU：CPU 生成 → 搬移（官方约定，可复现）
    - generator 在 cuda：直接在 cuda 生成
    - generator=None：用 seed 建 CPU generator（seed=None 则全局流）
    """
    if generator is None and seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    if generator is not None and generator.device.type == "cpu":
        lat = torch.randn(shape, generator=generator, device="cpu",
                          dtype=dtype)
        return lat.to(device)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def run_pipeline(image_path=IMG_DEFAULT, out_glb=OUT_DEFAULT,
                 steps=10, gs=5.0, res=32, seed=42,
                 sample_dtype=torch.float16, device="cuda",
                 save_npz=True, verbose=True):
    """自研内核端到端：图 → GLB。返回 (mesh_stats, meta)。"""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True

    log = print if verbose else (lambda *a, **k: None)

    # 1) 条件：官方预处理 + 自研 conditioner（fp32 CPU）
    img_pm1 = official_preprocess(image_path)
    px = dino_transform(img_pm1.cpu())
    cond_model, _ = load_conditioner(device="cpu")
    with torch.no_grad():
        cond = cond_model(px)
    del cond_model
    log(f"[cond] {tuple(cond.shape)} tok_norm={cond.norm(dim=-1).mean():.3f}")

    # 2) 采样（GPU，官方 randn_tensor 噪声语义）
    dit = build_dit().to(device, dtype=sample_dtype)
    x0 = prepare_latents((1, 4096, LATENT_DIM), seed=seed,
                         device=device, dtype=sample_dtype)
    t0 = time.time()
    lat = sample_loop(dit, x0, cond.to(device, sample_dtype),
                      num_steps=steps, shift=1.0, guidance_scale=gs)
    log(f"[sample] {steps} 步 {time.time() - t0:.1f}s  "
        f"latent mean={lat.mean():.4f} std={lat.std():.4f}")
    del dit
    torch.cuda.empty_cache()

    # 3) 解码 + MC（CPU fp32，阶段 6 已验证与官方逐位一致）
    log("[decode] CPU fp32 ...")
    t0 = time.time()
    vae = build_vae().cpu()
    grid = decode_to_grid(vae, lat[0].float().cpu().unsqueeze(0),
                          octree_resolution=res, num_chunks=20000,
                          verbose=False)
    v, f = extract_surface_mc(grid[0], mc_level=0.0, bounds=1.01,
                              octree_resolution=res)
    s = mesh_stats(v, f)
    log(f"[mesh] V={s['n_vertices']} F={s['n_faces']} "
        f"watertight={s.get('watertight')} euler={s.get('euler')} "
        f"({time.time() - t0:.1f}s)")

    # 4) GLB 导出（官方 export_to_trimesh 语义：faces 翻转绕向）
    import trimesh
    f_flipped = np.asarray(f)[:, ::-1]
    mesh = trimesh.Trimesh(vertices=v, faces=f_flipped)   # 默认 process=True
    out_glb = os.path.abspath(out_glb)
    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    mesh.export(out_glb)
    log(f"[export] {out_glb}  V={len(mesh.vertices)} F={len(mesh.faces)} "
        f"watertight={mesh.is_watertight}")

    # 5) 元数据 + npz
    meta = {
        "image": os.path.abspath(image_path),
        "steps": steps, "guidance_scale": gs, "octree_resolution": res,
        "seed": seed, "sample_dtype": str(sample_dtype),
        "vertices": int(s["n_vertices"]), "faces": int(s["n_faces"]),
        "watertight": bool(s.get("watertight")),
        "euler": s.get("euler"),
        "bbox_min": [float(x) for x in s["bbox_min"]],
        "bbox_max": [float(x) for x in s["bbox_max"]],
        "glb": out_glb,
    }
    with open(os.path.splitext(out_glb)[0] + "_meta.json", "w") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)
    if save_npz:
        np.savez(os.path.join(HERE, "stage7_latent.npz"),
                 latent=lat[0].float().cpu().numpy())
    return meta


def selfcheck(steps=2, device="cuda"):
    """确定性自检：generator 流 + 采样循环（同输入两次逐位一致）。"""
    print("[selfcheck A] prepare_latents generator 确定性 ...")
    g1 = torch.Generator(device="cpu").manual_seed(42)
    g2 = torch.Generator(device="cpu").manual_seed(42)
    a = prepare_latents((1, 4096, LATENT_DIM), generator=g1, device=device)
    b = prepare_latents((1, 4096, LATENT_DIM), generator=g2, device=device)
    ok_a = torch.equal(a, b)
    print(f"    同 seed 两次生成 maxdiff={(a - b).abs().max().item():.2e} "
          f"-> {'PASS' if ok_a else 'FAIL'}")

    print(f"[selfcheck B] 采样循环确定性（{steps} 步 × 2 次）...")
    dit = build_dit().to(device, dtype=torch.float16)
    cond = torch.randn(1, 1370, 1024, device=device, dtype=torch.float16)
    x0 = prepare_latents((1, 4096, LATENT_DIM), seed=42, device=device)
    l1 = sample_loop(dit, x0, cond, num_steps=steps, shift=1.0,
                     guidance_scale=5.0)
    l2 = sample_loop(dit, x0, cond, num_steps=steps, shift=1.0,
                     guidance_scale=5.0)
    ok_b = torch.equal(l1, l2)
    print(f"    两次采样 maxdiff={(l1 - l2).abs().max().item():.2e} "
          f"-> {'PASS' if ok_b else 'FAIL'}")
    del dit
    torch.cuda.empty_cache()
    return ok_a and ok_b


def main():
    ap = argparse.ArgumentParser(description="自研 3D 内核端到端 GLB 导出")
    ap.add_argument("--image", default=IMG_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--gs", type=float, default=5.0)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selfcheck-only", action="store_true")
    args = ap.parse_args()

    ok = selfcheck(steps=2)
    if args.selfcheck_only:
        return
    meta = run_pipeline(image_path=args.image, out_glb=args.out,
                        steps=args.steps, gs=args.gs, res=args.res,
                        seed=args.seed)
    meta["selfcheck"] = ok
    print(f"\n[verdict] selfcheck={'PASS' if ok else 'FAIL'}  "
          f"mesh watertight={meta['watertight']}  F={meta['faces']}")
    print(f"          GLB: {meta['glb']}")


if __name__ == "__main__":
    main()
