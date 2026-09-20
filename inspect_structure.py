# -*- coding: utf-8 -*-
"""自研 3D 内核 · 模型结构探查工具。

读取 hunyuan_3d_v2.1.safetensors 的全部权重键与形状，
为手写 DiT 前向推理提供结构蓝图。这是自研内核的第一步。
"""
import os
import sys
import json
from collections import defaultdict

from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHAPE_CKPT = os.path.join(ROOT, "models", "diffusion_models", "hunyuan_3d_v2.1.safetensors")
VAE_CKPT = os.path.join(ROOT, "models", "vae", "hunyuan3d_v2.1_vae.safetensors")
OUT = os.path.join(HERE, "model_structure.json")


def inspect(path, label):
    print(f"\n=== {label}: {os.path.basename(path)} ===")
    if not os.path.isfile(path):
        print("  未本地化:", path)
        return None
    info = {"file": path, "keys": {}, "groups": {}}
    groups = defaultdict(int)
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        print(f"  总权重张量数: {len(keys)}")
        # 统计顶层命名空间（DiT block / VAE encoder/decoder 等）
        for k in keys:
            top = k.split(".")[0]
            groups[top] += 1
        for k in keys:
            sl = f.get_slice(k)
            info["keys"][k] = list(sl.get_shape())
        info["groups"] = dict(sorted(groups.items(), key=lambda x: -x[1]))
        # 打印顶层结构
        for g, c in info["groups"].items():
            print(f"    {g:30s} x{c}")
        # 打印前 40 个键做样本
        print("  样本键:")
        for k in keys[:40]:
            print(f"      {k}  {info['keys'][k]}")
    return info


def main():
    result = {"shape_dit": inspect(SHAPE_CKPT, "Hunyuan3D DiT (shape)"),
              "vae": inspect(VAE_CKPT, "Hunyuan3D ShapeVAE")}
    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(f"\n[save] 结构蓝图 -> {OUT}")


if __name__ == "__main__":
    main()
