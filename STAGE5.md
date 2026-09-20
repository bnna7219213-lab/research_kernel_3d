# 阶段 5 · ShapeVAE 解码（latent → 占用场）

日期：2026-09-20 ｜ 实现：`stage5_vae.py`（解码端内核 + 体积解码）+ `stage5_diag.py`（官方对齐诊断）

## 结论

**自研 ShapeVAE 解码端与官方库完全一致**（cos=1.0000，maxdiff=0.0000），
首次实现 latent → 占用场（occupancy grid）的端到端解码。

## 解码通路（官方参照 → 自研实现）

`ShapeVAE.decode` + `VanillaVolumeDecoder` + `CrossAttentionDecoder`：

```
latents [B,4096,64]（采样器输出）
  └─ ÷ scale_factor(1.00395)        ← 官方 _export 惯例
  └─ post_kl: Linear(64→1024)
  └─ transformer: 16× ResidualAttentionBlock（自注意力，qk_norm）
       → decoded [B,4096,1024]
  └─ geo_decoder(queries):          ← 每批查询坐标
       query_proj(Fourier(xyz)) → ResidualCrossAttentionBlock → ln_post → output_proj
       → occ logits [B,P,1]
  └─ 分块拼回 grid_logits [B,r+1,r+1,r+1]
```

## 与 DiT 阶段的结构差异（易错点）

| 项 | DiT（阶段 3/4） | ShapeVAE（阶段 5） |
|----|----------------|--------------------|
| qkv_bias | False | False（同为 False，但 geo_decoder c_kv 无 bias） |
| qk_norm 类型 | RMSNorm | **LayerNorm(affine, eps=1e-6)** |
| qk_norm 挂载 | block 内 `attn.attention.q_norm` | 同样挂 `attention.q_norm`（本阶段初错挂在 `attn.q_norm`，按权重键名修正） |
| LayerNorm eps | 1e-5/1e-6 混合 | **统一 1e-6** |
| FourierEmbedder | 无 | num_freqs=8, **include_pi=False**（频率=2^i 不乘 π），out_dim=51 |
| scale_factor | — | **解码前必须 ÷1.00395**（官方 _export 在 vae(latents) 之前除） |

## 对齐结果（`stage5_diag.py`，同输入 latent/查询点，CPU fp32）

| 对照项 | cosine | max_abs_diff |
|--------|--------|--------------|
| A. decode（post_kl + transformer 16层） | 1.0000 | 0.000000 |
| B. geo_decoder（512 个随机查询点） | 1.0000 | 0.000000 |
| C. decode（latents ÷ scale_factor 后） | 1.0000 | 0.000000 |

一次通过，未经历 DiT 阶段的多轮修正——因为沿用阶段 4 已验证的
"交错拆头 + attention 内 qk_norm + eps=1e-6" 经验。

## 体积分块解码冒烟（`stage5_vae.py --grid`）

随机 latent（非采样器产出）→ r=32 占用网格：
`[1,33,33,33] mean=-0.980 std=0.164 min=-1.005 max=+1.003`，无 NaN。
（随机 latent 无形状语义，仅验证数值健康；真实形状待阶段 6 提取表面后可视化。）

## 已知边界

- 目前只做**解码端**；ShapeVAE 的 encoder/pre_kl（点云→latent）未实现
  （推理链路不需要编码，仅训练/编辑需要）。
- `FlashVDMVolumeDecoding`（官方加速路径，adaptive top-k）未实现——
  自研走 `VanillaVolumeDecoder` 等价的全分辨率分块查询，r=256 时
  (257)³≈1700 万查询点，CPU 上耗时可观；阶段 6 先在 r=64/128 验证。
- 占用网格 → mesh 的 marching cubes 属阶段 6（官方 MCSurfaceExtractor
  用 `mc_algo='mc'` 或 'dmc'）。

## 路线图状态

阶段 0 ✅ → 1 ✅ → 2 ✅ → 3 ✅ → 4 ✅ → **5 ✅（本轮）** → 6（表面提取）→ 7（GLB 导出）

阶段 6 所需输入已就绪：`decode_to_grid()` 输出即官方 `latents2mesh` 的
`grid_logits`，可直接接 marching cubes（skimage/trimesh 均有实现，
官方用自定义 CUDA marching cubes + DisoMesh，自研可用 skimage 代替）。
