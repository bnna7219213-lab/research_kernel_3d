# -*- coding: utf-8 -*-
"""自研 3D 内核 · 阶段 5：ShapeVAE 解码（latent → 占用场 grid_logits）。

官方参照（vendor/Hunyuan3D-2.1/hy3dshape/models/autoencoders/）：
  ShapeVAE.decode(latents):
    latents = post_kl(latents)          # [B,4096,64] -> [B,4096,1024]
    latents = transformer(latents)      # 16 层 ResidualAttentionBlock（自注意力）
    return latents
  VanillaVolumeDecoder(latents, geo_decoder, bounds, octree_resolution):
    在 [-1.01,1.01]^3 上生成 (res+1)^3 密集查询网格，分块送 geo_decoder：
      query_embeddings = query_proj(fourier_embedder(queries))   # 傅里叶位置编码 -> width
      x = cross_attn_decoder(query_embeddings, latents)          # 1 个 ResidualCrossAttentionBlock
      x = ln_post(x); occ = output_proj(x)                       # -> [B,P,1]
    拼回 grid_logits [B,res+1,res+1,res+1]

关键配置（config.yaml vae 节）：
  num_latents=4096, embed_dim=64, width=1024, heads=16,
  num_decoder_layers=16, qkv_bias=False, qk_norm=True（eps=1e-6）,
  num_freqs=8, include_pi=False（频率=2^i, 不乘 pi）,
  geo_decoder_downsample_ratio=1, geo_decoder_mlp_expand_ratio=4,
  geo_decoder_ln_post=True, scale_factor=1.0039506158752403

注意：FourierEmbedder(num_freqs=8, include_input=True) 的 out_dim = 3*(8*2+1) = 51。
   频率布局：frequencies=[2^0..2^7]；forward: embed=(x[...,None]*freq).view(...,-1)
   输出 cat([x, sin(embed), cos(embed)])，其中 embed 的通道顺序是
   "先坐标后频率"：dim 维展开为 [x0*f0..x0*f7, x1*f0..x1*f7, x2*f0..x2*f7]。

红线：不 import hy3dshape；官方库仅在对照函数中使用。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

VAE_CKPT = os.path.join(ROOT, "models", "vae", "hunyuan3d_v2.1_vae.safetensors")

WIDTH = 1024
HEADS = 16
NUM_LATENTS = 4096
EMBED_DIM = 64
NUM_FREQS = 8
NUM_DEC_LAYERS = 16
LN_EPS = 1e-6
SCALE_FACTOR = 1.0039506158752403  # 训练时 encode 后乘的系数；decode 前是否除见对齐结果


# ───────────── 傅里叶位置编码 ─────────────

class FourierEmbedder(nn.Module):
    """官方 FourierEmbedder(num_freqs=8, logspace=True, include_pi=False)。

    frequencies = 2^[0..7]（不乘 pi）。
    out = cat([x, sin(x*f), cos(x*f)], -1)，f 展开顺序为"先坐标后频率"。
    out_dim = 3 * (2*8 + 1) = 51。
    """

    def __init__(self, num_freqs=NUM_FREQS):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("frequencies", freqs, persistent=False)
        self.out_dim = 3 * (num_freqs * 2 + 1)

    def forward(self, x):  # [..., 3]
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((x, embed.sin(), embed.cos()), dim=-1)


# ───────────── 基础块（eps=1e-6, qkv_bias=False, qk_norm=RMS 风格的 LayerNorm affine） ─────────────

class MLP(nn.Module):
    def __init__(self, width, expand_ratio=4):
        super().__init__()
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    """官方 QKVMultiheadAttention：q_norm/k_norm 挂在这里（键名 attn.attention.q_norm.*）。"""

    def __init__(self, width=WIDTH, heads=HEADS, qk_norm=True):
        super().__init__()
        self.heads = heads
        d = width // heads
        self.q_norm = nn.LayerNorm(d, elementwise_affine=True, eps=LN_EPS) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(d, elementwise_affine=True, eps=LN_EPS) if qk_norm else nn.Identity()

    def forward(self, qkv):
        B, N, W = qkv.shape
        d = W // self.heads // 3
        qkv = qkv.view(B, N, self.heads, -1)
        q, k, v = torch.split(qkv, [d, d, d], dim=-1)   # 交错拆头（与官方一致）
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B,H,N,D]
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(B, N, -1)


class MultiheadAttention(nn.Module):
    """官方 MultiheadAttention：c_qkv 打包 -> view(B,N,H,3D) -> split(D) 交错拆头。"""

    def __init__(self, width=WIDTH, heads=HEADS, qkv_bias=False, qk_norm=True):
        super().__init__()
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(width, heads, qk_norm)

    def forward(self, x):
        return self.c_proj(self.attention(self.c_qkv(x)))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width=WIDTH, heads=HEADS, qkv_bias=False, qk_norm=True):
        super().__init__()
        self.attn = MultiheadAttention(width, heads, qkv_bias, qk_norm)
        self.ln_1 = nn.LayerNorm(width, elementwise_affine=True, eps=LN_EPS)
        self.mlp = MLP(width)
        self.ln_2 = nn.LayerNorm(width, elementwise_affine=True, eps=LN_EPS)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, layers=NUM_DEC_LAYERS, width=WIDTH, heads=HEADS,
                 qkv_bias=False, qk_norm=True):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, qkv_bias, qk_norm) for _ in range(layers)]
        )

    def forward(self, x):
        for blk in self.resblocks:
            x = blk(x)
        return x


# ───────────── 交叉注意力解码器（geo_decoder） ─────────────

class QKVCrossAttention(nn.Module):
    """官方 QKVMultiheadCrossAttention：q 独立 view，kv 打包交错 split。"""

    def __init__(self, width=WIDTH, heads=HEADS, qk_norm=True):
        super().__init__()
        self.heads = heads
        d = width // heads
        self.q_norm = nn.LayerNorm(d, elementwise_affine=True, eps=LN_EPS) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(d, elementwise_affine=True, eps=LN_EPS) if qk_norm else nn.Identity()

    def forward(self, q, kv):
        B, N, _ = q.shape
        _, M, W = kv.shape
        attn_ch = W // self.heads // 2
        q = q.view(B, N, self.heads, -1)
        kv = kv.view(B, M, self.heads, -1)
        k, v = torch.split(kv, [attn_ch, attn_ch], dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(B, N, -1)


class MultiheadCrossAttention(nn.Module):
    def __init__(self, width=WIDTH, heads=HEADS, qkv_bias=False, qk_norm=True):
        super().__init__()
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVCrossAttention(width, heads, qk_norm)

    def forward(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        return self.c_proj(self.attention(x, data))


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, width=WIDTH, heads=HEADS, expand=4, qkv_bias=False, qk_norm=True):
        super().__init__()
        self.attn = MultiheadCrossAttention(width, heads, qkv_bias, qk_norm)
        self.ln_1 = nn.LayerNorm(width, elementwise_affine=True, eps=LN_EPS)
        self.ln_2 = nn.LayerNorm(width, elementwise_affine=True, eps=LN_EPS)
        self.ln_3 = nn.LayerNorm(width, elementwise_affine=True, eps=LN_EPS)
        self.mlp = MLP(width, expand)

    def forward(self, x, data):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class CrossAttentionDecoder(nn.Module):
    """官方 CrossAttentionDecoder（downsample_ratio=1, ln_post=True）。"""

    def __init__(self, width=WIDTH, heads=HEADS, out_channels=1,
                 expand=4, qkv_bias=False, qk_norm=True):
        super().__init__()
        self.fourier_embedder = FourierEmbedder(NUM_FREQS)
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            width, heads, expand, qkv_bias, qk_norm)
        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)

    def forward(self, queries, latents):
        q = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        x = self.cross_attn_decoder(q, latents)
        x = self.ln_post(x)
        return self.output_proj(x)


# ───────────── ShapeVAE 解码端整体 ─────────────

class ShapeVAEDecoder(nn.Module):
    """只做 decode 通路：post_kl -> transformer(16 层) -> geo_decoder。"""

    def __init__(self):
        super().__init__()
        self.post_kl = nn.Linear(EMBED_DIM, WIDTH)
        self.transformer = Transformer(NUM_DEC_LAYERS, WIDTH, HEADS,
                                       qkv_bias=False, qk_norm=True)
        self.geo_decoder = CrossAttentionDecoder(WIDTH, HEADS, out_channels=1,
                                                 expand=4, qkv_bias=False,
                                                 qk_norm=True)

    def decode(self, latents):
        """[B,4096,64] -> [B,4096,1024] 解码后 latent。"""
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents

    def query_occ(self, queries, decoded_latents):
        """queries [B,P,3] -> occupancy logits [B,P,1]。"""
        return self.geo_decoder(queries, decoded_latents)


# ───────────── 权重加载 ─────────────

def load_vae_sd():
    from safetensors.torch import load_file
    return load_file(VAE_CKPT)


def build_and_load():
    model = ShapeVAEDecoder()
    sd = load_vae_sd()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # encoder/pre_kl 相关键应当 unexpected（我们没建）；其余应全中
    enc_keys = [k for k in unexpected if k.startswith(("encoder.", "pre_kl."))]
    real_unexpected = [k for k in unexpected if not k.startswith(("encoder.", "pre_kl."))]
    print(f"[load] missing={len(missing)} unexpected(non-encoder)={len(real_unexpected)} "
          f"skipped(encoder/pre_kl)={len(enc_keys)}")
    if missing:
        print("  MISSING:", missing[:10])
    if real_unexpected:
        print("  UNEXPECTED:", real_unexpected[:10])
    model.eval()
    return model


def _selfcheck():
    torch.manual_seed(0)
    model = build_and_load()
    lat = torch.randn(1, NUM_LATENTS, EMBED_DIM)
    with torch.no_grad():
        dec = model.decode(lat)
        q = torch.rand(1, 128, 3) * 2 - 1
        occ = model.query_occ(q, dec)
    print(f"[selfcheck] decode out: {tuple(dec.shape)} mean={dec.mean():.4f} std={dec.std():.4f}")
    print(f"[selfcheck] occ out:   {tuple(occ.shape)} mean={occ.mean():.4f} std={occ.std():.4f}")
    assert not torch.isnan(dec).any() and not torch.isnan(occ).any()
    print("[PASS] 阶段5 自研 ShapeVAE 解码端自检")


# ───────────── 体积分块解码（官方 VanillaVolumeDecoder 等价） ─────────────

def generate_dense_grid(bounds=1.01, octree_resolution=64):
    """在 [-bounds,bounds]^3 生成 (res+1)^3 查询网格（ij 索引），与官方一致。"""
    import numpy as np
    r = octree_resolution
    axis = np.linspace(-bounds, bounds, r + 1, dtype=np.float32)
    xs, ys, zs = np.meshgrid(axis, axis, axis, indexing="ij")
    xyz = np.stack((xs, ys, zs), axis=-1)          # [r+1,r+1,r+1,3]
    return torch.from_numpy(xyz).reshape(-1, 3)


@torch.no_grad()
def decode_to_grid(model, latents, octree_resolution=64, bounds=1.01,
                   num_chunks=10000, verbose=True):
    """latents [B,4096,64]（采样器输出，未除 scale_factor）-> grid_logits [B,r+1,r+1,r+1]。

    流程与官方 _export + VanillaVolumeDecoder 一致：
      latents = latents / scale_factor
      decoded = post_kl -> transformer
      分块查询 geo_decoder 得 logits，拼回体素网格。
    """
    B = latents.shape[0]
    latents = latents / SCALE_FACTOR
    decoded = model.decode(latents)
    xyz = generate_dense_grid(bounds, octree_resolution)
    grid_size = octree_resolution + 1
    chunks = []
    n = xyz.shape[0]
    it = range(0, n, num_chunks)
    if verbose:
        from tqdm import tqdm
        it = tqdm(it, desc=f"Volume Decoding r{octree_resolution}")
    for s in it:
        q = xyz[s:s + num_chunks].unsqueeze(0).expand(B, -1, -1)
        chunks.append(model.query_occ(q, decoded))
    grid = torch.cat(chunks, dim=1).view(B, grid_size, grid_size, grid_size).float()
    return grid


def _grid_smoke():
    """随机 latent 端到端体积解码冒烟（数值健康，不验证形状语义）。"""
    torch.manual_seed(0)
    model = build_and_load()
    lat = torch.randn(1, NUM_LATENTS, EMBED_DIM)
    grid = decode_to_grid(model, lat, octree_resolution=32, num_chunks=20000,
                          verbose=False)
    print(f"[grid] shape={tuple(grid.shape)} mean={grid.mean():.4f} "
          f"std={grid.std():.4f} min={grid.min():.4f} max={grid.max():.4f}")
    assert not torch.isnan(grid).any()
    print("[PASS] 体积分块解码冒烟")


if __name__ == "__main__":
    if "--grid" in sys.argv:
        _grid_smoke()
    else:
        _selfcheck()
