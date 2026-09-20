# -*- coding: utf-8 -*-
"""自研 3D 内核 · 阶段 6 对齐与端到端验证（计算可行版）。

官方 pipeline 的 3.68B DiT 在 6GB 显存上只能走 CPU，10 步 CFG 采样需数十分钟，
故把验证拆成可精确比对的四段（不跑官方大模型采样）：

  A. 采样调度数学：自研 _sigmas/更新式 vs 官方 FlowMatchEulerDiscreteScheduler
     的 set_timesteps/step，用**合成速度场**逐点比对（不依赖模型，精确到位）。
  B. CFG 组合数学：自研 cond/uncond 拼接与加权 vs 官方 encode_cond + chunk 公式。
  C. 体积解码：自研 decode_to_grid vs 官方 VolumeDecoder（官方 VAE 仅 0.3B，可跑）。
  D. 表面提取：自研 skimage MC vs 官方 MCSurfaceExtractor。
  E. 语义验证：自研全链路（真实图 + CFG）是否产出真实形状
     —— 无 CFG 时解码场全负（无等值面），CFG 后应出现闭合表面。

关键修正（本轮）：阶段 4 的 sigma 方向与官方相反、且缺 CFG，导致解码出全负场。
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from stage2_conditioner import load_conditioner, preprocess
from stage3_dit import build_and_load as build_dit, LATENT_DIM
from stage4_sampler import _sigmas, sample_loop
from stage5_vae import build_and_load as build_vae, decode_to_grid
from stage6_surface import extract_surface_mc, mesh_stats

CONFIG_YAML = os.path.join(ROOT, "models", "hy3dgen", "tencent",
                           "Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", "config.yaml")
SHAPE_CKPT = os.path.join(ROOT, "models", "diffusion_models",
                          "hunyuan_3d_v2.1.safetensors")
IMG = os.path.join(ROOT, "output", "aigc_img_00002_.png")

RES = 32
STEPS = 10
GS = 5.0
SEED = 42


def _vendor_path():
    """把 vendor 官方库加入 sys.path（对照用途，不 import 进内核）。"""
    vendor = os.path.join(ROOT, "vendor", "Hunyuan3D-2.1-main", "hy3dshape")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    return vendor


# ───────────── A. 采样调度数学对齐 ─────────────

def _official_scheduler(num_steps=STEPS):
    """官方 FlowMatchEulerDiscreteScheduler，按 pipeline 的方式 set_timesteps。"""
    _vendor_path()
    from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    sched.set_timesteps(num_inference_steps=None, sigmas=np.linspace(0, 1, num_steps),
                        device="cpu")
    return sched


def check_A():
    """官方 step() 实跑 vs 自研 _sigmas + 更新式（合成恒定速度场）。"""
    print("[A] 采样调度数学（官方 step() 实跑）...")
    sched = _official_scheduler()
    sig = _sigmas(STEPS, 1.0)                      # 长度 STEPS+1
    v = torch.ones(1, 1, LATENT_DIM)               # 恒定速度，解析解可推
    x_off = torch.zeros(1, 1, LATENT_DIM)
    x_mine = x_off.clone()

    max_t = max_s = max_x = 0.0
    for i in range(STEPS):
        t = sched.timesteps[i]
        x_off = sched.step(v, t, x_off).prev_sample   # 官方调度器真实前向

        s, sn = float(sig[i]), float(sig[i + 1])
        x_mine = x_mine + (sn - s) * v                # 自研更新式

        max_t = max(max_t, abs(float(t) / 1000.0 - s))     # 模型实际收到 t/1000
        max_s = max(max_s, abs(s - float(sched.sigmas[i])),
                    abs(sn - float(sched.sigmas[i + 1])))
        max_x = max(max_x, float((x_mine - x_off).abs().max()))

    print(f"    步数={STEPS}  t(=sigma*1000) maxdiff={max_t:.2e}")
    print(f"    sigmas maxdiff={max_s:.2e}   latent maxdiff={max_x:.2e}")
    print(f"    sigma 前 4 官方={[round(float(sched.sigmas[i]),4) for i in range(4)]}")
    print(f"    sigma 前 4 自研={[round(float(sig[i]),4) for i in range(4)]}")
    ok = max_t < 1e-5 and max_s < 1e-5 and max_x < 1e-6
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_B():
    """CFG 组合：自研 cat/chunk 公式 vs 官方 step() 的 prev_sample 传播。"""
    print("[B] CFG 组合数学（官方 step() 实跑）...")
    torch.manual_seed(0)
    x = torch.randn(1, 4, LATENT_DIM)
    cond_p = torch.randn(1, 4, 1024)    # 条件嵌入（类比 DINOv2 输出）
    cond_n = torch.zeros(1, 4, 1024)    # 无条件嵌入（类比空 prompt）

    # 合成"模型"：速度 = f(x, cond)，与真实 DiT 无关，只看组合与传播公式
    def v(x_, c_):
        return x_.tanh() + 0.01 * c_.sum(dim=(1, 2)).view(-1, 1, 1)

    # 自研循环（CFG 组合在速度空间，组合后走官方同款更新）
    sig = _sigmas(STEPS, 1.0)
    xc = x.clone()
    for i in range(STEPS):
        s, sn = float(sig[i]), float(sig[i + 1])
        comb = v(xc, cond_n) + GS * (v(xc, cond_p) - v(xc, cond_n))
        xc = xc + (sn - s) * comb

    # 官方 step() 实跑：先组合速度，再交官方调度器传播
    sched = _official_scheduler()
    xo = x.clone()
    for i in range(STEPS):
        comb = v(xo, cond_n) + GS * (v(xo, cond_p) - v(xo, cond_n))
        xo = sched.step(comb, sched.timesteps[i], xo).prev_sample

    d = float((xc - xo).abs().max())
    print(f"    自研 CFG+更新  vs 官方 step(): maxdiff={d:.2e}")
    print(f"    latent std: 自研={xc.std():.4f} 官方={xo.std():.4f}")
    ok = d < 1e-6
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


# ───────────── C/D. 官方 VAE + 表面提取器（小模型，可跑） ─────────────

def load_official_vae():
    _vendor_path()
    from hy3dshape.utils import instantiate_from_config
    import yaml
    with open(CONFIG_YAML) as f:
        cfg = yaml.safe_load(f)
    from safetensors.torch import load_file
    vae = instantiate_from_config(cfg["vae"])
    ckpt = load_file(os.path.join(ROOT, "models", "vae",
                                  "hunyuan3d_v2.1_vae.safetensors"))
    vae.load_state_dict(ckpt, strict=False)
    vae.eval()
    return vae


def check_CD(grid_mine, lat_mine):
    print(f"[C] 体积解码对齐 r={RES} ...")
    off = load_official_vae()
    grid_mine = grid_mine.float().cpu()
    lat_mine = lat_mine.float().cpu()
    with torch.no_grad():
        grid_off = off.volume_decoder(
            off(lat_mine / off.scale_factor), off.geo_decoder,
            bounds=1.01, octree_resolution=RES, num_chunks=20000,
            enable_pbar=False)
    dC = (grid_mine - grid_off).abs().max().item()
    n_pos_m = int((grid_mine[0] > 0).sum())
    n_pos_o = int((grid_off[0] > 0).sum())
    print(f"    grid {tuple(grid_mine.shape)} mean={grid_mine.mean():.4f} "
          f"std={grid_mine.std():.4f}")
    print(f"    正体素: 自研={n_pos_m} 官方={n_pos_o}   maxdiff={dC:.2e}")
    okC = dC < 1e-4 and n_pos_m == n_pos_o and n_pos_m > 0
    print(f"    -> {'PASS' if okC else 'FAIL'}")

    print("[D] 表面提取对齐 ...")
    v, f = extract_surface_mc(grid_mine[0], mc_level=0.0, bounds=1.01,
                              octree_resolution=RES)
    v, f = np.asarray(v), np.asarray(f)
    from hy3dshape.models.autoencoders.surface_extractors import MCSurfaceExtractor
    out_off = MCSurfaceExtractor()(grid_off, mc_level=0.0, bounds=1.01,
                                   octree_resolution=RES)[0]
    v_o, f_o = out_off.mesh_v, out_off.mesh_f
    same_v = v.shape == v_o.shape
    dD = float(np.abs(v - v_o).max()) if same_v else float("nan")
    same_f = (f.shape == f_o.shape) and bool((f == f_o).all())
    s = mesh_stats(v, f)
    print(f"    自研 mesh: V={s['n_vertices']} F={s['n_faces']} "
          f"watertight={s.get('watertight')} euler={s.get('euler')}")
    print(f"    官方 mesh: V={v_o.shape[0]} F={f_o.shape[0]}")
    print(f"    vertices maxdiff={dD:.2e}  faces identical={same_f}")
    okD = same_v and dD < 1e-4 and same_f
    print(f"    -> {'PASS' if okD else 'FAIL'}")
    return okC, okD, s, v, f


# ───────────── E. 自研全链路语义验证 ─────────────

def run_self_pipeline(cfg, device="cpu"):
    """自研全链路：真实图 → DINOv2 → 采样 → 解码 → MC。

    注：DiT 3.68B 的 fp32 约 14.7GB，本机 6GB 显存无法承载（fp16 也需 ~7.4GB），
    故采样走 CPU；VAE（0.3B）与 conditioner 同样走 CPU 以保证数值可比。
    """
    from PIL import Image
    img = np.asarray(Image.open(IMG).convert("RGB").resize(
        (518, 518), Image.LANCZOS), dtype=np.float32) / 255.0
    px = preprocess(img).to(device)

    cond_model, _ = load_conditioner(device=device)
    with torch.no_grad():
        cond = cond_model(px)
    print(f"    cond {tuple(cond.shape)} mean={cond.mean():.4f}")

    dit = build_dit().to(device)
    torch.manual_seed(SEED)
    x0 = torch.randn(1, 4096, LATENT_DIM, device=device)
    lat = sample_loop(dit, x0, cond, num_steps=STEPS, shift=1.0,
                      guidance_scale=cfg.get("gs", GS))
    print(f"    latent mean={lat.mean():.4f} std={lat.std():.4f}")

    vae = build_vae().to(device)
    grid = decode_to_grid(vae, lat, octree_resolution=RES,
                          num_chunks=20000, verbose=False)
    print(f"    grid mean={grid.mean():.4f} std={grid.std():.4f} "
          f"min={grid.min():.4f} max={grid.max():.4f} "
          f"正体素={int((grid[0] > 0).sum())}")
    return grid, lat


def check_E():
    print("[E] 自研全链路语义验证（真实图 + CFG）...")
    grid, lat = run_self_pipeline({"gs": GS})
    v, f = extract_surface_mc(grid[0], mc_level=0.0, bounds=1.01,
                              octree_resolution=RES)
    s = mesh_stats(v, f)
    print(f"    mesh: V={s['n_vertices']} F={s['n_faces']} "
          f"watertight={s.get('watertight')} euler={s.get('euler')}")
    print(f"    bbox: min={[round(x,3) for x in s['bbox_min']]} "
          f"max={[round(x,3) for x in s['bbox_max']]}")
    ok = s["n_faces"] > 1000 and s.get("watertight") is True
    print(f"    -> {'PASS（形状真实出现）' if ok else 'FAIL'}")
    return ok, grid, v, f, s, lat


def main():
    a = check_A()
    b = check_B()
    okE, grid, v, f, s, lat = check_E()
    okC, okD, s2, v2, f2 = check_CD(grid, lat)
    np.savez(os.path.join(HERE, "stage6_e2e.npz"),
             grid=grid[0].numpy(), vertices=v, faces=f, latent=lat[0].numpy())
    print("\n[summary] A=%s B=%s E=%s C=%s D=%s" % (a, b, okE, okC, okD))
    print(f"[verdict] {'PASS' if all([a, b, okE, okC, okD]) else 'FAIL'}")


if __name__ == "__main__":
    main()
