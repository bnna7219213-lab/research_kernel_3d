# -*- coding: utf-8 -*-
"""阶段 6 · 决定性对照：官方 pipeline 同条件实跑。

问题背景：stage6_verify.py 的 A/B/C/D 全部 PASS（调度/CFG/解码/表面提取
与官方逐位一致），但 E 段（自研全链路：真实图 + CFG + 10 步 + res32）
产出的形状琐碎（正体素 5、F=36、watertight）。

本脚本用官方 Hunyuan3DDiTFlowMatchingPipeline 以**完全相同**的参数
（fp16、res32、steps=10、GS=5.0、同一张图）实跑：
  - 官方若出大 mesh → 硬件/配置不是瓶颈，问题在自研内核的某个细节
    （候选：时间嵌入幅度、cond CLS token 处理、采样起点/方向等）
  - 官方若同样琐碎 → 10 步 + res32 在该 checkpoint 上本身不足，
    E 段判据（F>1000）需要调参（更多步数/更高分辨率）而不是改内核

运行：python research_kernel_3d/stage6_official_e2e.py
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RES = 32
STEPS = 10
GS = 5.0
IMG = os.path.join(ROOT, "output", "aigc_img_00002_.png")


def _vendor_path():
    vendor = os.path.join(ROOT, "vendor", "Hunyuan3D-2.1-main", "hy3dshape")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    return vendor


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    _vendor_path()
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    model_dir = os.path.join(ROOT, "models", "hy3dgen", "tencent",
                             "Hunyuan3D-2.1", "hunyuan3d-dit-v2-1")
    config_yaml = os.path.join(model_dir, "config.yaml")
    ckpt = os.path.join(ROOT, "models", "diffusion_models",
                        "hunyuan_3d_v2.1.safetensors")
    print(f"[load] 官方 pipeline from_single_file: {ckpt}")
    t0 = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt,
        config_yaml,
        device="cuda",
        dtype=torch.float16,
        use_safetensors=True,
    )
    print(f"[load] done in {time.time() - t0:.1f}s")

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
    )
    dt = time.time() - t0
    mesh = out[0][0] if isinstance(out[0], list) else out[0]
    print(f"[run] done in {dt:.1f}s")
    print(f"[result] V={len(mesh.vertices)} F={len(mesh.faces)} "
          f"watertight={mesh.is_watertight} "
          f"bbox={mesh.bounds.tolist() if mesh.bounds is not None else None}")

    if mesh.is_watertight and len(mesh.faces) > 1000:
        print("[verdict] 官方能出形状 → 自研内核存在细节差异，需逐项对齐")
    else:
        print("[verdict] 官方同样琐碎 → 10 步 + res32 配置本身不足，"
              "E 段判据需调参（步数/分辨率）")


if __name__ == "__main__":
    main()
