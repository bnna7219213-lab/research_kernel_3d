# -*- coding: utf-8 -*-
"""阶段 4 对齐诊断：逐层对比自研 vs 官方，定位分歧点。"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from stage3_dit import ShapeDiT, build_and_load, HIDDEN, LATENT_DIM

CKPT = os.path.join(ROOT, "models", "diffusion_models",
                    "hunyuan_3d_v2.1.safetensors")
CONFIG_YAML = os.path.join(ROOT, "models", "hy3dgen", "tencent",
                           "Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", "config.yaml")


def _official_hooks():
    """官方库单步，带逐层 hook。"""
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

    feats = {}
    def hook(name):
        def fn(mod, inp, out):
            feats[name] = out.detach().clone() if isinstance(out, torch.Tensor) else out
        return fn

    # 关键节点：x_embedder、t_embedder、block 0/10/15/20 输出、final
    model.x_embedder.register_forward_hook(hook("x_embedder"))
    model.t_embedder.register_forward_hook(hook("t_embedder"))
    for i in (0, 5, 10, 11, 15, 20):
        model.blocks[i].register_forward_hook(hook(f"block{i}"))
    model.final_layer.register_forward_hook(hook("final"))
    return model, feats


def _self_hooks(model):
    feats = {}
    def hook(name):
        def fn(mod, inp, out):
            feats[name] = out.detach().clone() if isinstance(out, torch.Tensor) else out
        return fn
    model.x_embedder.register_forward_hook(hook("x_embedder"))
    model.t_embedder.register_forward_hook(hook("t_embedder"))
    for i in (0, 5, 10, 11, 15, 20):
        model.blocks[i].register_forward_hook(hook(f"block{i}"))
    # final 输出在 forward 里直接拿
    return feats


def _cmp(a, b, name):
    a = a.flatten().float()
    b = b.flatten().float()
    if a.shape != b.shape:
        print(f"  {name}: shape 不同 self={a.shape} off={b.shape}")
        return
    cos = torch.nn.functional.cosine_similarity(a[None], b[None]).item()
    diff = (a - b).abs().mean().item()
    print(f"  {name}: cos={cos:.4f} diff={diff:.4f} self_std={a.std():.4f} off_std={b.std():.4f}")


def main():
    torch.manual_seed(42)
    B, N, M = 1, 64, 16
    x = torch.randn(B, N, LATENT_DIM)
    t = torch.tensor([0.5])
    cond = torch.randn(B, M, 1024)

    print("[1] 官方（带 hook）...")
    off_model, off_feats = _official_hooks()
    with torch.no_grad():
        off_out = off_model(x, t, {"main": cond})
    off_feats["final_out"] = off_out.detach().clone()

    print("[2] 自研（带 hook）...")
    self_model = build_and_load()
    self_feats = _self_hooks(self_model)
    with torch.no_grad():
        self_out = self_model(x, t, cond)
    self_feats["final_out"] = self_out.detach().clone()

    print("\n[3] 逐层对比：")
    for name in ("x_embedder", "t_embedder", "block0", "block5", "block10",
                 "block11", "block15", "block20", "final_out"):
        if name in off_feats and name in self_feats:
            _cmp(self_feats[name], off_feats[name], name)
        else:
            print(f"  {name}: 缺 self={name in self_feats} off={name in off_feats}")


if __name__ == "__main__":
    main()
