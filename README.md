# 自研 3D 内核（research_kernel_3d）

> 长期计划子项目：纯从 `hunyuan_3d_v2.1.safetensors` 手写 Hunyuan3D 2.1 推理，
> 不依赖腾讯官方库、不依赖 ComfyUI、不依赖 diffusers。
> **与主项目并行执行、互不影响**：官方库（vendor/Hunyuan3D-2.1）是主线产出 GLB；
> 本子项目独立演进，首阶段只求"最初研发时有一点点效果"，往后逐渐推进。

## 定位与边界

- **目标**：完全自研的 3D 生成推理内核（逆向腾讯自研 DiT+ShapeVAE 架构）。
- **非目标**：首阶段不追求达到官方库质量；允许功能不足、表现简单。
- **隔离**：独立目录、独立入口、独立配置，不改动主 CLI/官方库路径。

## 模型结构蓝图（已探查，见 model_structure.json）

`hunyuan_3d_v2.1.safetensors`（1601 张量）含三套子网络：

| 命名空间 | 张量数 | 作用 |
|----------|--------|------|
| `conditioner.*` | 439 | DINOv2 图像条件编码器（含 patch_embed/24层 encoder） |
| `model.*` | 752 | 主 DiT（21 层 + MoE，flow-matching 去噪） |
| `vae.*` | 410 | 内嵌 ShapeVAE（latent ↔ SDF） |

`hunyuan3d_v2.1_vae.safetensors`（410 张量）：`encoder`/`geo_decoder`/`transformer`/`pre_kl`/`post_kl`。

## 路线图（渐进式，允许初期只有粗略效果）

- [x] **阶段 0 · 结构探查**（已完成）：`inspect_structure.py` 读出全部权重键/形状 → `model_structure.json`
- [x] **阶段 1 · 权重加载器**（已完成）：safetensors → torch state_dict，按命名空间拆三套子网络，实测 0.304B/3.051B/0.328B 对齐
- [x] **阶段 2 · Conditioner（DINOv2）**（已完成）：手写图像编码前向，数值自检通过（详见 `STAGE2.md`）
- [x] **阶段 3 · DiT 前向（单步）**（已完成）：752 张量 100% 精确加载，单步前向数值自洽；新探明 U-ViT 跳跃连接（blocks 11-20 skip_linear）与共享专家 MoE 结构（详见 `STAGE3.md`）
- [x] **阶段 4 · flow-matching 采样循环**（已完成）：欧拉采样 + **与官方库逐层对齐（cos=1.0000）**；修正时间注入（序列前缀 token）、Attention head 交错排列、MoE gate 归一化（详见 `STAGE4.md`）
- [x] **阶段 5 · ShapeVAE 解码**（已完成）：post_kl + 16 层 transformer + geo_decoder，与官方逐层对齐 cos=1.0000/maxdiff=0；体积分块解码冒烟通过（详见 `STAGE5.md`）
- [x] **阶段 6 · 表面提取 + 端到端验证**（已完成）：skimage MC 与官方逐位一致；**自研内核端到端产出 F=9920 watertight GLB（官方同 x0 F=9890，量级一致），Blender 无头 QA 通过**；定位早期差距根因 = fp16 数值路径混沌放大 + x0 噪声流差异，非内核缺陷（详见 `STAGE6.md`）
- [x] **阶段 7 · GLB 导出**（已完成）：`stage7_glb.py` 端到端一键 CLI（官方 randn_tensor 噪声语义 + sdp_kernel flash 对齐 + 确定性自检 0.00e+00），seed=42/steps=10 实测 F=1184 watertight，Blender 无头 QA 通过（详见 `STAGE7.md`）

> 每个阶段独立可验证（打印中间张量形状/范数即可），不阻塞主线。
> 里程碑式推进：哪怕只到阶段 3（DiT 单步前向数值合理），也算"有一点点效果"。

## 运行

```bash
# 结构探查（已跑通）
python research_kernel_3d/inspect_structure.py
```

## 参考（仅作数学参照，不 import）

- 论文：arXiv 2506.15442（Hunyuan3D 2.1）
- 官方实现：vendor/Hunyuan3D-2.1（**主线**用它；本内核仅对照其结构，不复制代码）
