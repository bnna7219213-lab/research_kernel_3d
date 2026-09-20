# -*- coding: utf-8 -*-
"""自研 3D 内核 · 阶段2：Conditioner（DINOv2 图像条件编码器）手写前向。

结构完全从 safetensors 键名/形状反推（未复制任何官方实现）：
  conditioner.main_image_encoder.model.*
    embeddings.cls_token                [1, 1, 1024]
    embeddings.mask_token               [1, 1024]      (掩码训练用，前向不用)
    embeddings.patch_embeddings.projection.weight [1024, 3, 14, 14]  (conv, stride=patch)
    embeddings.position_embeddings      [1, 1370, 1024]  (1 CLS + 37*37 patches @518px)
    encoder.layer.{0..23}.*
      norm1 -> attention{q,k,v + output.dense} -> layer_scale1.lambda1 -> 残差
      norm2 -> mlp{fc1 1024->4096, GELU, fc2 4096->1024} -> layer_scale2.lambda1 -> 残差
      (fc1 是单输出 4096 维，即标准 GELU MLP，不是 SwiGLU；SwiGLU 会是 2*hidden 或拆 w12)
    layernorm                           (最终 LayerNorm)

推断结论：DINOv2-large ViT，embed=1024，patch=14，24 层，预归一化，
LayerScale（lambda1 逐通道缩放），16 头（1024/64）。518 = 37*14，1370 = 1 + 37^2。

红线：不 import hy3dshape / comfy / diffusers；仅 torch + numpy + PIL。
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHAPE_CKPT = os.path.join(ROOT, "models", "diffusion_models", "hunyuan_3d_v2.1.safetensors")
COND_PREFIX = "conditioner.main_image_encoder.model."

NUM_HEADS = 16      # 1024 / 64，权重里没有存头数，按 DINOv2-large 惯例
HEAD_DIM = 64
IMG_SIZE = 518
PATCH = 14
GRID = IMG_SIZE // PATCH          # 37
NUM_PATCHES = GRID * GRID         # 1369
NUM_TOKENS = NUM_PATCHES + 1      # 1370 (CLS 在最前)


class PatchEmbed(nn.Module):
    """conv2d(3->1024, k=14, s=14) + flatten -> [B, 1369, 1024]"""

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 1024, kernel_size=PATCH, stride=PATCH)

    def forward(self, x):                       # x: [B,3,518,518]
        x = self.projection(x)                  # [B,1024,37,37]
        return x.flatten(2).transpose(1, 2)     # [B,1369,1024]


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1024))
        self.mask_token = nn.Parameter(torch.zeros(1, 1024))  # 仅占位加载，前向不用
        self.patch_embeddings = PatchEmbed()
        self.position_embeddings = nn.Parameter(torch.zeros(1, NUM_TOKENS, 1024))

    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        patch_tokens = self.patch_embeddings(pixel_values)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, patch_tokens], dim=1)   # CLS 在最前
        return tokens + self.position_embeddings


class SelfAttention(nn.Module):
    """与键名对齐：attention.{query,key,value} 内部子模块 + 外层 output.dense"""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(1024, 1024)
        self.key = nn.Linear(1024, 1024)
        self.value = nn.Linear(1024, 1024)

    def forward(self, x):                        # [B,1370,1024]
        b, n, _ = x.shape
        def heads(t):
            return t.view(b, n, NUM_HEADS, HEAD_DIM).transpose(1, 2)  # [B,H,N,64]
        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        attn = F.scaled_dot_product_attention(q, k, v)   # 全量注意力，无 mask
        return attn.transpose(1, 2).reshape(b, n, 1024)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = SelfAttention()
        self.output = nn.ModuleDict({"dense": nn.Linear(1024, 1024)})

    def forward(self, x):
        return self.output["dense"](self.attention(x))


class Mlp(nn.Module):
    """fc1(1024->4096) GELU fc2(4096->1024)：标准 ViT MLP（非 SwiGLU）"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(1024))

    def forward(self, x):
        return x * self.lambda1


class EncoderLayer(nn.Module):
    """预归一化 + LayerScale 残差（从键名 norm1/attention/layer_scale1 推断）"""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(1024)
        self.attention = Attention()
        self.layer_scale1 = LayerScale()
        self.norm2 = nn.LayerNorm(1024)
        self.mlp = Mlp()
        self.layer_scale2 = LayerScale()

    def forward(self, x):
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([EncoderLayer() for _ in range(24)])

    def forward(self, x):
        for blk in self.layer:
            x = blk(x)
        return x


class DinoConditioner(nn.Module):
    """与 safetensors 键 'conditioner.main_image_encoder.model.*' 去掉前缀后严格对齐。"""

    def __init__(self):
        super().__init__()
        self.embeddings = Embeddings()
        self.encoder = Encoder()
        self.layernorm = nn.LayerNorm(1024)

    def forward(self, pixel_values):             # [B,3,518,518]，已按 ImageNet 归一化
        hidden = self.embeddings(pixel_values)
        hidden = self.encoder(hidden)
        hidden = self.layernorm(hidden)
        return hidden                            # [B,1370,1024]

    def forward_split(self, pixel_values):
        hidden = self.forward(pixel_values)
        return hidden[:, 0], hidden[:, 1:]       # CLS [B,1024], patch tokens [B,1369,1024]


def load_conditioner(path=SHAPE_CKPT, device="cuda", dtype=torch.float32):
    from safetensors.torch import load_file
    sd = load_file(path)
    cond = {k[len(COND_PREFIX):]: v for k, v in sd.items() if k.startswith(COND_PREFIX)}
    assert len(cond) == 439, f"expect 439 tensors under prefix, got {len(cond)}"
    model = DinoConditioner()
    model.load_state_dict({k: v.to(dtype) for k, v in cond.items()}, strict=True)
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    return model, n


def make_test_image(size=IMG_SIZE, seed=0):
    """纯 numpy 生成 518x518 RGB 测试图：渐变 + 圆斑 + 噪声。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.stack([xx, yy, 0.5 * (xx + yy)], axis=-1)
    c = size // 2
    r = size // 4
    circle = ((yy * size - c) ** 2 + (xx * size - c) ** 2) < r * r
    img[circle] = img[circle] * 0.3 + 0.7
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def preprocess(img01):
    """[H,W,3] float in [0,1] -> [1,3,518,518]，ImageNet mean/std 归一化。"""
    x = torch.from_numpy(img01).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def _stats(name, t):
    t = t.float()
    print(f"  {name:14s} shape={tuple(t.shape)}  mean={t.mean():+.4f}  std={t.std():.4f}  "
          f"L2norm(tok)={t.norm(dim=-1).mean():.2f}  min={t.min():+.3f}  max={t.max():+.3f}  "
          f"nan={torch.isnan(t).any().item()}  inf={torch.isinf(t).any().item()}")


def _selfcheck():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage2] device={device}")
    model, n_params = load_conditioner(device=device)
    print(f"[load] conditioner 参数量 = {n_params/1e6:.1f}M（条件器总 304.4M 中 mask_token 占 0.001M）")

    img = make_test_image()
    px = preprocess(img).to(device)
    with torch.no_grad():
        cls, patches = model.forward_split(px)

    print("[forward] 518x518 合成测试图：")
    _stats("CLS", cls)
    _stats("patch_tokens", patches)
    full = torch.cat([cls.unsqueeze(1), patches], dim=1)
    _stats("all_tokens", full)

    # 自洽性断言
    assert cls.shape == (1, 1024) and patches.shape == (1, NUM_PATCHES, 1024)
 
    for t in (cls, patches):
        assert not torch.isnan(t).any() and not torch.isinf(t).any()
        assert t.abs().max() < 100, "输出爆界"
        assert t.std() > 1e-4, "输出塌缩为零方差"
    # 确定性：同输入两次前向一致
    with torch.no_grad():
        cls2, _ = model.forward_split(px)
    assert torch.allclose(cls, cls2, atol=1e-5), "前向不确定"
    # 区分度：不同图给出不同 embedding
    px2 = preprocess(make_test_image(seed=1)).to(device)
    with torch.no_grad():
        cls3, _ = model.forward_split(px2)
    cos = F.cosine_similarity(cls, cls3).item()
    print(f"[check] 两张不同测试图 CLS 余弦相似度 = {cos:.4f}（应 <1.0，说明输出对输入敏感）")
    assert cos < 0.9999
    print("[PASS] 阶段2 conditioner 前向自检通过")


if __name__ == "__main__":
    _selfcheck()
