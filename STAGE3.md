# 阶段 3 · DiT 前向（单步）手写实现

日期：2026-09-20 ｜ 实现：`stage3_dit.py`（纯 torch，不 import hy3dshape/comfy/diffusers）

## 结论

**752 个 DiT 张量（model.* 命名空间）100% 精确加载（missing=0 / unexpected=0）**，
单步前向数值自洽：输出 [1,64,64] std=0.245 无 NaN，两次前向 allclose，
不同 t 输出不同，MoE 路由多样。

## 结构（全部从 safetensors 键名/形状反推）

```
x [B,N,64] --x_embedder(Linear 64→2048)--> h [B,N,2048]
t [B] --正弦嵌入256维→gelu扩至2048--> t_embedder(MLP 2048→8192→2048)--> t_vec
context [B,M,1024]（阶段2 conditioner 输出）

blocks 0-10（编码段，DenseMLP）
  每层: h=norm1(x)+t_vec → attn1 自注意力（16头×128，q/k RMSNorm per-head）
        h=norm2(x)+t_vec → attn2 交叉注意力（to_k/v 从 1024 维映射，吃 context）
        h=norm3(x)+t_vec → mlp（fc1 8192 / fc2）
  block10 输出保存为 skip_src

blocks 11-20（解码段，U-ViT 跳跃连接）
  每层先: x = skip_norm(x) + skip_linear(concat(skip_src, x))  # [2048,4096]
  再走与编码段相同的 attn1/attn2/mlp 结构

blocks 15-20（MoE 层，DeepSeek 式）
  mlp → moe:
    shared_experts（MLP，全 token 共享）
    gate（Linear 2048→8）softmax → top-2 路由 → experts.0-7（各一个 MLP）
    输出 = shared + Σ top-2 加权专家

final_layer: norm_final(LayerNorm) + linear(2048→64)
```

## 关键结构发现（本次新探明）

1. **U-ViT 跳跃连接**：blocks 11-20 各含 `skip_linear [2048,4096]` + `skip_norm`。
   4096=2048×2，即 concat(早层输出, 当前输入) 后投影——21 层 DiT 实为
   "前 11 层编码 + 后 10 层带 skip 解码"的 U-ViT 结构。skip 源取 block 10 输出。
2. **共享专家**：MoE 层除 8 个路由专家外还有 `shared_experts`（所有 token 都过），
   DeepSeek-MoE 式设计。总输出 = shared + top-2 加权。
3. **时间注入**：无 adaLN 调制键（无 modulation/scale_shift），t_vec 直接加到
   每个 norm 后的 hidden 上（假设，见"未解问题"）。

## 权重键映射（官方 → 本实现）

| 官方键 | 本实现 |
|--------|--------|
| `t_embedder.mlp.{0,2}` | `t_embedder.{0,2}` |
| `final_layer.norm_final` / `final_layer.linear` | `final_norm` / `final_linear` |
| `.moe.experts.*.net.0.proj` / `net.2` | `.moe.experts.*.fc1` / `fc2` |
| `.moe.shared_experts.net.0.proj` / `net.2` | `.moe.shared_experts.fc1` / `fc2` |
| `attn{1,2}.{q,k}_norm.weight` | `attn{1,2}.{q,k}_norm`（裸 Parameter，去 .weight） |
| 其余（blocks/attn/mlp/x_embedder） | 一一对应 |

## 自检数值（`python stage3_dit.py`，CPU fp32）

| 检查 | 结果 |
|------|------|
| 权重加载 | **missing=0 / unexpected=0**（752/752） |
| 输出 shape | [1, 64, 64]（64 token latent） |
| 输出统计 | mean=-0.0135, std=0.2449, absmax=0.7140 |
| NaN/Inf | 无 |
| 两次前向 allclose | True（1e-5） |
| t=0 vs t=1 差异 | 0.0016（>0，时间条件有效但幅度小） |
| MoE 路由多样性 | layer15 的 64 tokens 选了 2 种专家；全 6 层专家命中分布 [128,192,64,128,64,64,64,64] |

## 未解问题（留给阶段 4+）

1. **时间注入方式**：t_vec 加到 norm 后是假设（无调制键的最简可行方案）。
   真实实现可能是 t 仅进 attn 的 bias、或经 FiLM 等。阶段 4 对照官方输出修正。
2. **位置编码缺失**：未发现任何 pos_embed/rope 键——latent token 的位置信息
   可能来自 VAE 的 latent 结构本身（vectset 排列），或 conditioner 的 cross-attn
   已隐含位置。阶段 5 接 ShapeVAE 时验证。
3. **t=0/1 差异幅度偏小**（0.0016 vs 输出 std 0.245）：可能是时间注入假设不完整，
   或正弦嵌入的 scale 需调整（如 1000×t）。阶段 4 对齐官方采样时修正。
4. **MoE top-2 重归一**：top_val 归一方式（原值 vs 重归一）未与官方对照。

## 阶段进度

- [x] 阶段 0 结构探查
- [x] 阶段 1 权重加载器（3.68B 对齐）
- [x] 阶段 2 Conditioner（DINOv2 304.4M，前向自洽）
- [x] **阶段 3 DiT 前向（752 张量 100% 加载，单步自洽）**
- [ ] 阶段 4 flow-matching 采样循环
- [ ] 阶段 5 ShapeVAE 解码
- [ ] 阶段 6 表面提取
- [ ] 阶段 7 GLB 导出
