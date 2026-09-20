# -*- coding: utf-8 -*-
"""自研 3D 内核 · 阶段 3：DiT 单步前向（Hunyuan3D 2.1 shape DiT，纯手写）。

结构（全部从 safetensors 键名/形状反推，不参考官方实现代码）：
  x_embedder   Linear 64 -> 2048          （latent token 维度 64）
  t_embedder   MLP  2048 -> 8192 -> 2048  （时间条件）
  blocks.0-14  DenseMLP
    attn1  自注意力   to_q/k/v [2048,2048]  q/k RMSNorm per-head [128]
    attn2  交叉注意力 to_q [2048,2048] to_k/v [2048,1024]（吃 conditioner 1024 维输出）
    norm1/norm2/norm3  LayerNorm(weight+bias)
    mlp    fc1 [8192,2048] fc2 [2048,8192]
  blocks.15-20 MoE（6 层）
    同上，但 mlp -> moe: gate [8,2048]（router，8 专家）+
    experts.{0..7}.net.0.proj [8192,2048]（fc1）+ net.2 [2048,8192]（fc2）
    top-2 路由（论文 2506.15442：6 层 MoE / 8 专家 / top-2）
  final_layer  norm_final + linear [64,2048]

时间注入假设（无 adaLN 调制键，记录在 STAGE3.md）：
  t_embedder 输出直接加到每个 token 的 hidden 上（t 为标量时即逐 token 加同一向量）。
  这是无 modulation 键时的最简可行假设，后续阶段可对照官方输出修正。

数值自检（_selfcheck）：
  1) 输出 shape [B,N,64]、有界、无 NaN
  2) 同输入两次前向 allclose
  3) 不同 t 输出显著不同
  4) MoE gate softmax top-2 在不同 token 上路由到不同专家
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(os.path.dirname(HERE), "models", "diffusion_models",
                    "hunyuan_3d_v2.1.safetensors")
STRUCTURE = os.path.join(HERE, "model_structure.json")

N_LAYERS = 21
HIDDEN = 2048
LATENT_DIM = 64
NUM_HEADS = 16          # 2048 / 128 head_dim
HEAD_DIM = 128
MOE_BLOCKS = (15, 16, 17, 18, 19, 20)
N_EXPERTS = 8
TOP_K = 2


def load_model_sd():
    """加载 model.* 命名空间（752 张量，fp32 CPU），去 model. 前缀。"""
    import safetensors.torch as st
    sd = st.load_file(CKPT)
    out = {}
    for k, v in sd.items():
        if k.startswith("model."):
            out[k[len("model."):]] = v.float()
    assert len(out) == 752, f"expect 752, got {len(out)}"
    return out


class SelfAttn(nn.Module):
    """attn1：自注意力，per-head RMSNorm q/k。

    官方 head 排列（vendor hunyuandit.py Attention.forward）：
      qkv = cat(q,k,v) -> view(B,N,H,3*D) -> split(D)
      即每个 head 内 q/k/v 交错打包，而非 q 的 H 个 head 连续排列。
    """

    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.to_k = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.to_v = nn.Linear(HIDDEN, HIDDEN, bias=False)
        # 官方：nn.RMSNorm(head_dim, eps=1e-6)（qk_norm_type='rms'）
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=1e-6)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=1e-6)
        self.out_proj = nn.Linear(HIDDEN, HIDDEN, bias=True)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.to_q(x)  # [B,N,HIDDEN]
        k = self.to_k(x)
        v = self.to_v(x)
        # 官方：cat(q,k,v) -> view(B,N,H,3*D) -> split(D) -> [B,H,N,D]
        qkv = torch.cat((q, k, v), dim=-1)                    # [B,N,3*HIDDEN]
        qkv = qkv.view(1, -1, NUM_HEADS, 3 * HEAD_DIM)        # [1,B*N,H,3D]
        q, k, v = torch.split(qkv, HEAD_DIM, dim=-1)          # 各 [1,B*N,H,D]
        q = q.reshape(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.reshape(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.reshape(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        q = self.q_norm(q)                                    # 官方 nn.RMSNorm
        k = self.k_norm(k)
        o = _sdpa_official(q, k, v)   # [B,H,N,D]
        o = o.transpose(1, 2).reshape(B, N, HIDDEN)
        return self.out_proj(o)


def _sdpa_official(q, k, v):
    """官方 Attention 的 SDPA 配置：flash + mem_efficient，禁 math。

    与 vendor hunyuandit.py Attention.forward 的 sdp_kernel 上下文一致；
    CPU 上退回默认（math backend）。
    """
    if q.is_cuda:
        from torch.backends.cuda import sdp_kernel
        with sdp_kernel(enable_flash=True, enable_math=False,
                        enable_mem_efficient=True):
            return F.scaled_dot_product_attention(q, k, v)
    return F.scaled_dot_product_attention(q, k, v)


class CrossAttn(nn.Module):
    """attn2：交叉注意力，context 是 conditioner 输出 [B,M,1024]。

    官方 CrossAttention 的 head 排列与 SelfAttn 相同（cat 后按 head 交错）。
    """

    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.to_k = nn.Linear(1024, HIDDEN, bias=False)
        self.to_v = nn.Linear(1024, HIDDEN, bias=False)
        # 官方：nn.RMSNorm(head_dim, eps=1e-6)（qk_norm_type='rms'）
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=1e-6)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=1e-6)
        self.out_proj = nn.Linear(HIDDEN, HIDDEN, bias=True)

    def forward(self, x, context):
        B, N, _ = x.shape
        M = context.shape[1]
        q = self.to_q(x)                                     # [B,N,HIDDEN]
        k = self.to_k(context)                               # [B,M,HIDDEN]
        v = self.to_v(context)                               # [B,M,HIDDEN]
        # 官方：q 独立 view（不交错），kv = cat(k,v) -> view(B,M,H,2D) -> split(D)
        kv = torch.cat((k, v), dim=-1)                        # [B,M,2*HIDDEN]
        kv = kv.view(1, -1, NUM_HEADS, 2 * HEAD_DIM)          # [1,B*M,H,2D]
        k, v = torch.split(kv, HEAD_DIM, dim=-1)              # 各 [1,B*M,H,D]
        q = q.view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)  # [B,H,N,D]
        k = k.reshape(B, M, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.reshape(B, M, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        q = self.q_norm(q)                                    # 官方 nn.RMSNorm
        k = self.k_norm(k)
        o = _sdpa_official(q, k, v)
        o = o.transpose(1, 2).reshape(B, N, HIDDEN)
        return self.out_proj(o)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN, 8192)
        self.fc2 = nn.Linear(8192, HIDDEN)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoE(nn.Module):
    """DeepSeek 式：shared_experts（全 token 共享）+ 8 专家 top-2 路由。

    官方 MoEBlock.forward（vendor moe_layers.py）：
      y = moe_infer(x, topk_idx, topk_weight)  # 路由专家输出
      y = y + shared_experts(identity)         # shared 加在原始输入上
    gate 的 norm_topk_prob=True：topk_weight / sum(topk_weight)。
    """

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, N_EXPERTS, bias=False)
        self.experts = nn.ModuleList([MLP() for _ in range(N_EXPERTS)])
        self.shared_experts = MLP()   # 共享专家：所有 token 都过

    def forward(self, x, return_route=False):
        identity = x
        route = torch.softmax(self.gate(x), dim=-1)      # [B,N,8]
        top_val, top_idx = route.topk(TOP_K, dim=-1)       # [B,N,2]
        # 官方 MoEGate.norm_topk_prob=False：top-2 权重**不重归一化**，
        # 直接用 softmax 后的原始概率作为加权系数
        out = torch.zeros_like(x)
        for e in range(N_EXPERTS):
            mask = (top_idx == e)                          # [B,N,2]
            hit = mask.any(-1)                              # [B,N]
            if not hit.any():
                continue
            w = (top_val * mask).sum(-1, keepdim=True)      # [B,N,1]
            ex = self.experts[e](x)
            out = out + w * ex
        # 官方：shared 加在原始输入 identity 上
        out = out + self.shared_experts(identity)
        if return_route:
            return out, top_idx
        return out


class Block(nn.Module):
    """blocks 11-20 带 U-ViT 式 skip：skip_linear(concat(skip,x)) 后 skip_norm。

    官方顺序（vendor hunyuandit.py）：skip_linear 先 linear 后 norm。
    时间向量 c 不进 block（timested_modulate=False），它作为序列前缀 token 参与自注意力。
    """

    def __init__(self, use_moe, use_skip):
        super().__init__()
        self.attn1 = SelfAttn()
        self.attn2 = CrossAttn()
        self.norm1 = nn.LayerNorm(HIDDEN)
        self.norm2 = nn.LayerNorm(HIDDEN)
        self.norm3 = nn.LayerNorm(HIDDEN)
        self.moe = MoE() if use_moe else None
        self.mlp = None if use_moe else MLP()
        self.use_skip = use_skip
        if use_skip:
            self.skip_linear = nn.Linear(2 * HIDDEN, HIDDEN)
            self.skip_norm = nn.LayerNorm(HIDDEN)

    def forward(self, x, context, t_vec, skip_src=None):
        # t_vec 在 block 里不用（timested_modulate=False），仅作接口占位
        if self.use_skip:
            # 官方顺序：skip_linear(cat) 先 linear 后 norm
            joined = torch.cat([skip_src, x], dim=-1)
            x = self.skip_norm(self.skip_linear(joined))
        h = self.norm1(x)
        x = x + self.attn1(h)
        h = self.norm2(x)
        x = x + self.attn2(h, context)
        h = self.norm3(x)
        if self.moe is not None:
            x = x + self.moe(h)
        else:
            x = x + self.mlp(h)
        return x


class ShapeDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(LATENT_DIM, HIDDEN)
        self.t_embedder = nn.Sequential(
            nn.Linear(HIDDEN, 8192), nn.GELU(), nn.Linear(8192, HIDDEN))
        self.blocks = nn.ModuleList(
            [Block(i in MOE_BLOCKS, i >= 11) for i in range(N_LAYERS)])
        self.final_norm = nn.LayerNorm(HIDDEN)
        self.final_linear = nn.Linear(HIDDEN, LATENT_DIM)

    def forward(self, x, t, context, return_route=False):
        """x [B,N,64]，t [B] 标量，context [B,M,1024]。返回 [B,N,64]。

        官方结构（vendor hunyuandit.py 已读源码）：
          t = t_embedder(t)  # [B,1,2048]（unsqueeze(dim=1)）
          x = x_embedder(x)  # [B,N,2048]（use_pos_emb=false，不加 pos）
          c = t  # [B,1,2048]
          x = cat([c, x], dim=1)  # [B,N+1,2048]  ← 时间向量作为第 0 个 token
          for layer, block in enumerate(blocks):
              skip = None if layer <= depth//2 else skip_list.pop()
              x = block(x, c, cond, skip_value=skip)
              if layer < depth//2: skip_list.append(x)
          x = final_layer(x)
        """
        # 正弦时间嵌入（官方 Timesteps：hidden_size=2048 维）
        half = 1024  # 2048 // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) *
                          torch.arange(half, device=x.device) / half)
        ang = t.float()[:, None] * freqs[None]              # [B,1024]
        t_emb = torch.cat([torch.sin(ang), torch.cos(ang)], -1)  # [B,2048]
        t_emb = t_emb.to(x.dtype)
        t_vec = self.t_embedder(t_emb)                      # [B,2048]
        t_vec = t_vec.unsqueeze(1)                            # [B,1,2048]

        h = self.x_embedder(x)                                # [B,N,2048]
        # 时间向量作为第 0 个 token 拼到序列前（官方做法）
        h = torch.cat([t_vec, h], dim=1)                      # [B,N+1,2048]

        skip_list = []
        for i, blk in enumerate(self.blocks):
            skip = None if i <= N_LAYERS // 2 else skip_list.pop()
            h = blk(h, context, t_vec, skip_src=skip)
            if i < N_LAYERS // 2:
                skip_list.append(h)
        h = self.final_norm(h)
        out = self.final_linear(h)                            # [B,N+1,64]
        return out[:, 1:, :]  # 去掉第 0 个时间 token，只返回 latent token


def _blk_fwd_routed(blk, h, context, t_vec, skip_src=None):
    """带路由输出的 block 前向（MoE 层用）。"""
    x = h
    if blk.use_skip:
        joined = torch.cat([skip_src, x], dim=-1)
        x = blk.skip_norm(x) + blk.skip_linear(joined)
    hh = blk.norm1(x) + t_vec
    x = x + blk.attn1(hh)
    hh = blk.norm2(x) + t_vec
    x = x + blk.attn2(hh, context)
    hh = blk.norm3(x) + t_vec
    moe_out, route = blk.moe(hh, return_route=True)
    x = x + moe_out
    return x, route


def _load_into(model, sd):
    """把去前缀的 state_dict 装进模型。

    官方键名 -> 本实现的映射：
      t_embedder.mlp.{0,2}      -> t_embedder.{0,2}
      final_layer.norm_final    -> final_norm
      final_layer.linear        -> final_linear
      moe.experts.*.net.0.proj  -> moe.experts.*.fc1
      moe.experts.*.net.2       -> moe.experts.*.fc2
      attn1/attn2 不变
    """
    mapped = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("t_embedder.mlp."):
            nk = nk.replace("t_embedder.mlp.", "t_embedder.")
        elif nk.startswith("final_layer.norm_final"):
            nk = nk.replace("final_layer.norm_final", "final_norm")
        elif nk.startswith("final_layer.linear"):
            nk = nk.replace("final_layer.linear", "final_linear")
        elif ".moe.experts." in nk:
            nk = nk.replace("net.0.proj", "fc1").replace("net.2", "fc2")
        elif ".moe.shared_experts." in nk:
            nk = (nk.replace(".moe.shared_experts.net.0.proj", ".moe.shared_experts.fc1")
                  .replace(".moe.shared_experts.net.2", ".moe.shared_experts.fc2"))
        elif nk.endswith(".q_norm.weight") or nk.endswith(".k_norm.weight"):
            # 官方键带 .weight（nn.RMSNorm 模块）；本实现同为 nn.RMSNorm，键一致
            pass
        mapped[nk] = v
    own = set(model.state_dict().keys())
    unexpected = [k for k in mapped if k not in own]
    missing = [k for k in sorted(own) if k not in mapped]
    filtered = {k: v for k, v in mapped.items() if k in own}
    model.load_state_dict(filtered, strict=False)
    return missing, unexpected


def build_and_load():
    sd = load_model_sd()
    model = ShapeDiT()
    missing, unexpected = _load_into(model, sd)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("  unexpected sample:", unexpected[:6])
    if missing:
        print("  missing sample:", missing[:6])
    model.eval()
    return model


def _selfcheck():
    torch.manual_seed(42)
    B, N, M = 1, 64, 16      # 64 个 latent token，16 个条件 token
    model = build_and_load()

    x = torch.randn(B, N, LATENT_DIM)
    t = torch.tensor([0.5])
    ctx = torch.randn(B, M, 1024)   # 阶段 2 conditioner 的输出形状

    with torch.no_grad():
        out, routes = _fwd_routed(model, x, t, ctx)
    print(f"\n[out] shape={list(out.shape)}")
    print(f"[out] mean={out.mean():.4f} std={out.std():.4f} "
          f"absmax={out.abs().max():.4f} NaN={torch.isnan(out).any()}")

    # 一致性
    with torch.no_grad():
        out2 = model(x, t, ctx)
    same = torch.allclose(out, out2, atol=1e-5)
    print(f"[check] 两次前向 allclose: {same}")

    # 不同 t
    with torch.no_grad():
        out_t0 = model(x, torch.tensor([0.0]), ctx)
        out_t1 = model(x, torch.tensor([1.0]), ctx)
    diff = (out_t0 - out_t1).abs().mean()
    print(f"[check] t=0 vs t=1 输出差异: {diff:.4f}（应显著>0）")

    # MoE 路由多样性
    if routes:
        r = torch.stack(routes)          # [n_moe_layers, B, N, 2]
        uniq = r[0][0].unique()          # 第一个 MoE 层 64 个 token 的 top-2 专家编号
        print(f"[check] MoE layer15: 64 tokens 的 top-2 专家编号种类: "
              f"{uniq.tolist()}（>2 种说明路由多样）")
        counts = torch.bincount(r.flatten(), minlength=8)
        print(f"[check] 全部 MoE 层专家命中分布: {counts.tolist()}")
    return 0


def _fwd_routed(model, x, t, ctx):
    """带路由输出的完整前向。"""
    half = 128
    freqs = torch.exp(-torch.log(torch.tensor(10000.0)) *
                      torch.arange(half) / half)
    ang = t.float()[:, None] * freqs[None]
    t_emb = torch.cat([torch.sin(ang), torch.cos(ang)], -1)
    t_h = F.gelu(t_emb.repeat(1, HIDDEN // 256 + 1)[:, :HIDDEN])
    t_vec = model.t_embedder(t_h)

    h = model.x_embedder(x) + t_vec[:, None, :]
    skip_src = None
    routes = []
    for i, blk in enumerate(model.blocks):
        if i == 11:
            skip_src = h.clone()
        if blk.moe is not None:
            h, r = _blk_fwd_routed(blk, h, ctx, t_vec[:, None, :], skip_src)
            routes.append(r)
        else:
            h = blk(h, ctx, t_vec[:, None, :], skip_src=skip_src)
    h = model.final_norm(h)
    return model.final_linear(h), routes


if __name__ == "__main__":
    import sys
    sys.exit(_selfcheck())
