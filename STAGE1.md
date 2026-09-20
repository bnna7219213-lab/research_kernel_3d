# 阶段 1 · 权重加载器收尾

日期：2026-09-18 ｜ 运行时：`comfyui_V10.3/python/python.exe`（torch 2.9.1+cu130, CUDA 可用）

## 方法

`stage1_loader.py` 用 `safetensors.torch.load_file` 全量加载
`models/diffusion_models/hunyuan_3d_v2.1.safetensors`（1601 张量），
按键名第一段拆成 `conditioner` / `model` / `vae` 三组，逐组统计张量数、参数量、NaN/Inf。

## 自检结果（实跑 `python stage1_loader.py`）

| 子网络 | 张量数 | 参数量（实测） | 目标对齐 | NaN/Inf |
|--------|--------|----------------|----------|---------|
| conditioner（DINOv2 图像编码器） | 439 | 304.4M | 0.304B ✅ | 0 |
| model（主 DiT，21 层 + MoE） | 752 | 3050.8M | 3.051B ✅ | 0 |
| vae（内嵌 ShapeVAE） | 410 | 327.7M | 0.328B ✅ | 0 |
| **合计** | 1601 | **3.68B** | 3.68B ✅ | 0 |

三套参数量与目标 0.304B / 3.051B / 0.328B 完全对齐，全部权重无 NaN/Inf。

## 顺带确认的下游结构线索（供阶段 3+）

- DiT：`model.blocks.{0..20}`，hidden=2048；`attn1` 自注意力（q/k/v/out 均 2048²），
  `attn2` 交叉注意力（to_k/to_v 输入 1024 → 2048，条件端是 conditioner 的 1024 维 token）；
  q_norm/k_norm 各 128 维（RMSNorm on head_dim，128 头维 × 16 头 = 2048）；
  mlp fc1 8192 / fc2 2048；每层 3 个 LayerNorm（norm1/2/3）。
- 嵌入：`model.x_embedder` [2048, 64]（latent 64 维 → 2048）、
  `model.t_embedder.mlp.{0,2}` 2048→8192→2048、`model.final_layer` 2048→64。
- 键数对账：752 = 21 层 × 约 34 键 + 嵌入/收尾层。

## 结论

阶段 1 完成：权重可完整加载、可正确分组、数值健康，可作为阶段 2/3/5 的统一权重来源。
