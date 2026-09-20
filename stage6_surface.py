# -*- coding: utf-8 -*-
"""自研 3D 内核 · 阶段 6：表面提取（占用场 grid_logits → mesh 顶点/面）。

官方参照（vendor/.../autoencoders/surface_extractors.py）：
  MCSurfaceExtractor.run(grid_logit, mc_level, bounds, octree_resolution):
    vertices, faces, normals, _ = skimage.measure.marching_cubes(
        grid_logit.cpu().numpy(), mc_level, method="lewiner")
    vertices = vertices / grid_size * bbox_size + bbox_min   # grid_size = res+1

即官方 MC 路径就是 skimage lewiner marching cubes + 线性坐标映射，
自研实现可与其逐点对齐（同 grid_logits 输入时顶点/面应完全一致）。

注意：
  - marching_cubes 返回的 vertices 是 [x,y,z] 顺序的网格坐标（ij 索引），
    官方直接用 xyz 顺序映射到 bbox（因为 ij 索引与 xyz 轴对齐）。
  - 分块/分层解码产生的 NaN（未查询区域）在 Vanilla 全分辨率路径不存在；
    本实现仅处理 Vanilla 网格（无 NaN），分层路径留待后续。
  - mc_level=0.0 是占用场等值面（logits>0 为内部）。

红线：不 import hy3dshape；skimage/trimesh 是公开科学计算库，允许使用。
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


# ───────────── 坐标映射（与官方 _compute_box_stat 一致） ─────────────

def compute_box_stat(bounds, octree_resolution):
    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
    bbox_min = np.array(bounds[0:3], dtype=np.float32)
    bbox_max = np.array(bounds[3:6], dtype=np.float32)
    bbox_size = bbox_max - bbox_min
    grid_size = [int(octree_resolution) + 1] * 3
    return grid_size, bbox_min, bbox_size


# ───────────── marching cubes 表面提取 ─────────────

def extract_surface_mc(grid_logit, mc_level=0.0, bounds=1.01, octree_resolution=None):
    """grid_logit [r+1,r+1,r+1] (torch 或 numpy) → (vertices [V,3], faces [F,3])。

    与官方 MCSurfaceExtractor 完全相同的算法与坐标映射。
    """
    from skimage import measure

    if isinstance(grid_logit, torch.Tensor):
        grid_logit = grid_logit.cpu().numpy()
    grid_logit = np.asarray(grid_logit, dtype=np.float32)
    if octree_resolution is None:
        octree_resolution = grid_logit.shape[0] - 1

    vertices, faces, _normals, _values = measure.marching_cubes(
        grid_logit, mc_level, method="lewiner")
    grid_size, bbox_min, bbox_size = compute_box_stat(bounds, octree_resolution)
    vertices = vertices / grid_size * bbox_size + bbox_min
    return vertices.astype(np.float32), np.ascontiguousarray(faces)


# ───────────── 网格健康检查 ─────────────

def mesh_stats(vertices, faces):
    """基本健康指标：顶点/面数、bbox、watertight（需 trimesh）。"""
    v = np.asarray(vertices)
    f = np.asarray(faces)
    stats = {
        "n_vertices": int(v.shape[0]),
        "n_faces": int(f.shape[0]),
        "bbox_min": v.min(axis=0).tolist(),
        "bbox_max": v.max(axis=0).tolist(),
    }
    try:
        import trimesh
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        stats["watertight"] = bool(m.is_watertight)
        stats["euler"] = int(m.euler_number)
        stats["volume"] = float(m.volume) if m.is_watertight else None
    except Exception as e:
        stats["trimesh_error"] = str(e)
    return stats


# ───────────── 自检：解析占用场 → 球形等值面 ─────────────

def _selfcheck():
    """构造解析球体占用场（球内 logits>0），验证提取出的 mesh 接近单位球。"""
    r = 64
    axis = np.linspace(-1.01, 1.01, r + 1, dtype=np.float32)
    xs, ys, zs = np.meshgrid(axis, axis, axis, indexing="ij")
    dist = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    grid = (0.8 - dist)  # 半径 0.8 球内为正

    v, f = extract_surface_mc(grid, mc_level=0.0, bounds=1.01, octree_resolution=r)
    s = mesh_stats(v, f)
    print(f"[selfcheck] vertices={s['n_vertices']} faces={s['n_faces']}")
    print(f"[selfcheck] bbox=[{np.array(s['bbox_min']).round(3)}]..[{np.array(s['bbox_max']).round(3)}]")
    print(f"[selfcheck] watertight={s.get('watertight')} euler={s.get('euler')} volume={s.get('volume')}")

    # 半径 0.8 的球：理论体积 4/3*pi*0.8^3 ≈ 2.144；顶点应在 ~0.8 半径附近
    radii = np.linalg.norm(v, axis=1)
    print(f"[selfcheck] vertex radius: mean={radii.mean():.4f} std={radii.std():.4f} "
          f"(expect ~0.8)")
    assert s["n_vertices"] > 100 and s["n_faces"] > 100
    assert abs(radii.mean() - 0.8) < 0.02
    print("[PASS] 阶段6 表面提取自检（解析球）")


if __name__ == "__main__":
    _selfcheck()
