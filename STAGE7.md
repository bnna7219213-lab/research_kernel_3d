# 阶段 7 · GLB 导出固化（端到端一键 CLI）

日期：2026-09-20 ｜ 实现：`stage7_glb.py`（管线固化 + 噪声语义 + 确定性自检）

## 结论

**自研 3D 内核端到端管线固化完成**：一条命令完成
「图 → 官方预处理 → DINOv2 cond → DiT 采样 → ShapeVAE/MC → GLB」，
同 seed 逐位可复现（自检 maxdiff=0.00e+00），GLB 通过 Blender 5.2 无头 QA。

## 本阶段交付

### 1. `prepare_latents` —— 官方 randn_tensor 语义

复刻 diffusers `randn_tensor`：
- 传 **CPU generator** → CPU 生成 fp16 → 搬移设备（官方约定，可复现路径）
- generator=None + seed → 自动建 CPU generator（默认 `--seed 42`）
- 同 seed 两次生成 maxdiff=**0.00e+00**（自检 A PASS）

这解决了阶段 6 发现的「x0 噪声流差异」：官方 `randn_tensor(shape, generator,
device, dtype)` 与裸 `torch.randn` 全局流不同；微内核现在与官方同语义。

### 2. `run_pipeline` —— 单函数端到端

```
图 → official_preprocess(518) → DINOv2 cond(fp32 CPU)
  → prepare_latents(seed) → DiT 采样(GPU fp16, sdp_kernel flash)
  → ShapeVAE 解码 + MC(CPU fp32) → faces 翻转绕向
  → trimesh GLB(+ meta JSON + latent npz)
```

CLI：`--image --out --steps --gs --res --seed [--selfcheck-only]`

### 3. 官方 SDPA 配置对齐

`stage3_dit` 自研注意力改用 `_sdpa_official`：CUDA 上启用
`sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)`，
与 vendor `Attention.forward` 一致（数学结构已在阶段 4/6 验证不变）。

### 4. 确定性自检（内建）

- 自检 A：generator 噪声流 → maxdiff=0.00e+00 **PASS**
- 自检 B：采样循环同输入两次 → maxdiff=0.00e+00 **PASS**
  （`cudnn.benchmark=False` + `deterministic=True`）

## 实测结果（seed=42, steps=10, res=32, GS=5）

| 项 | 值 |
|---|---|
| 采样耗时 | 1257.7s（RTX 4050 6GB，fp16） |
| latent | mean=0.0019 std=0.9307 |
| mesh | V=594 **F=1184 watertight** euler=2 |
| Blender QA | import_ok=True, non_manifold=0, degenerate=0 |
| 尺寸 | 0.689 × 0.445 × 0.969 |

产物：`output/stage7_glb.glb` + `stage7_glb_meta.json` +
`stage7_glb_qa_blender.json` + `research_kernel_3d/stage7_latent.npz`

## 备注

- F=1184 与阶段 6 的 F=9920 不同属预期：本 run 的 x0 来自 **CPU generator
  seed=42**（官方语义），与阶段 6 traj 实验注入的 CUDA 全局流 x0 不同起点，
  扩散模型不同起点收敛到不同（都 watertight）的形状。
- 更高面数路径：`--res 64/128`、`--steps 50`（官方默认）。
- 路线图下一阶段：性能优化（fp8/量化、VAE 解码上 GPU）与 README 收尾。
