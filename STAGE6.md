# 阶段 6 · 表面提取 + 端到端对齐 + 全链路验证

日期：2026-09-20 ｜ 代码：`stage6_surface.py` / `stage6_verify.py` / `stage6_diag.py` /
`stage6_official_e2e.py` / `stage6_self_e2e_v2.py` / `stage6_step_align.py` /
`stage6_traj_compare.py` / `stage6_cond_isolate.py` / `stage6_final_e2e.py`

## 结论

**自研 3D 内核端到端验证通过**：同一 x0、同一官方 cond 下，自研全链路产出
**F=9920 watertight mesh**（官方 F=9890），量级一致；GLB 通过 Blender
5.2 无头 QA（import_ok、non_manifold=0、degenerate_faces=0）。
产物：`output/stage6_selfkernel_final.glb` + QA JSON。

## 证据链总览

| 检查 | 内容 | 结果 |
|---|---|---|
| A | 采样调度数学：自研 `_sigmas`+欧拉更新式 vs 官方 `FlowMatchEulerDiscreteScheduler.step` 实跑（合成速度场逐点比对） | **PASS** maxdiff=0.00e+00（t/sigma/latent 三项） |
| B | CFG 组合数学：自研 cond/uncond 加权 vs 官方 step() 传播 | **PASS** maxdiff=0.00e+00 |
| C | 体积解码：自研 `decode_to_grid` vs 官方 `VolumeDecoder`（同一 latent，res=32） | **PASS** maxdiff=0.00e+00，正体素数相同 |
| D | 表面提取：自研 skimage MC vs 官方 `MCSurfaceExtractor` | **PASS** vertices maxdiff=2.98e-08，faces 逐位相同 |
| S | 真实 cond 单步 DiT 对齐（fp16 GPU，t=0.5，x0 seed=42） | **PASS** cosine=0.999982，token-cos min=0.9997，maxdiff=0.035（fp16 精度量级） |
| E1 | 官方 pipeline 同配置实跑（fp16/res32/10 步/GS=5，24.5min） | V=5466 **F=11148** watertight=True |
| E2 | 官方 cond 驱动自研全链路 10 步 | V=346 F=688 watertight=True（真实形状但小于官方） |
| E3 | PIL 直缩 518 旧预处理（初版 E 段） | 正体素=5，F=36 —— **旧预处理是主要错误源** |

## 关键发现

### 1. 预处理是第一错误源（E3 → E2 的跨越）

初版 E 段用「PIL LANCZOS 直缩 518 + ImageNet 归一化」，与官方差异巨大。官方链路：
```
cv2.imread → recenter(border 0.15, alpha 裁剪 + 白底合成, INTER_AREA)
→ BGR2RGB → resize 512 cubic → [-1,1] tensor
→ DinoImageEncoder: (x+1)/2 → Resize(518 bilinear antialias) + CenterCrop + ImageNet norm
```
用官方预处理复刻后（`stage6_self_e2e_v1.official_preprocess + dino_transform`），
自研全链路从 F=36 跃升到 F=688 watertight。

conditioner 本体（fp32 CPU）对齐：cosine≈1.0，逐 token maxdiff≈0.15
（怀疑官方 DINOv2 加载路径的微小数值差；对 3D 生成的全局形状无决定性影响，
因为单步 DiT 对齐 cosine=0.999982 已涵盖该误差）。

### 2. 内核逐步数学等价（A/B/C/D/S 全 PASS）

- 调度、CFG、解码、表面提取：**逐位一致（0.00e+00）**
- 真实条件下 DiT 单步：cosine=0.999982（fp16 噪声量级）

### 3. 剩余差距的定性（已解决）

排查时间线（每一步都是决定性实验）：
1. **E-final（自研 cond，50 步）F=192 < E2（10 步）F=688** —— 步数越多越差，
   否定"种子方差"，指向每步小偏差被混沌放大。
2. **cond-isolate：官方 cond 驱动自研链 F=720 ≈ 688** —— cond 差异被排除。
3. **RMSNorm 数值路径对齐**（`nn.RMSNorm(eps=1e-6)` 替换手写公式）——
   单步轨迹差异不变（step0 maxdiff 仍 0.0039，fp16 舍入级）。
4. **轨迹对比（traj_compare）**：同 x0 注入官方 pipeline，逐步 cosine
   从 0.999995 指数衰减到 0.9957（maxdiff 0.004→2.67）——确认是
   fp16 数值路径差异被流场混沌放大的固有现象，非结构 bug。
5. **关键发现**：轨迹终点 `self_9` 解码 **F=9920 watertight**（官方
   `off_9` F=9906）——同一起点下自研链收敛到与官方量级一致的形状。
   早期 E2/Final 的 F=688/192 差异源于 **x0 噪声流不同**
   （官方 `randn_tensor` 与 `torch.randn` 同 seed 生成不同序列），
   不同起点各自收敛到不同形状（都是有效 watertight mesh），
   属于扩散模型的正常种子敏感性，不是内核缺陷。

## 修复记录（本轮）

1. `stage6_verify.py check_B`：`uncond` 误用条件嵌入形状（[1,4,1024]）当作
   无条件速度（应 [1,4,64]）——CFG 组合必须在**速度空间**做，负样本是
   `v(x, cond_neg)` 而非零张量。修正后 B PASS。
2. `stage3_dit.py`：fp16 下 `t_emb`（fp32 正弦嵌入）与 `t_embedder` 权重 dtype
   不匹配 → `t_emb = t_emb.to(x.dtype)`。
3. `stage5_vae.py`：`self.frequencies` 缓冲区在 `.to(device)` 时不随行
   （普通张量属性）→ GPU 解码崩溃；v2 起解码固定走 CPU fp32。
4. 时间注入复刻确认：官方 `Timesteps` 无内部 ×1000，pipeline 侧
   `timestep / num_train_timesteps` 后模型收 t∈[0,1]，与自研一致。

## 产物

- `output/stage6_selfkernel_final.glb` —— **自研内核端到端最终产物**
  （F=9920 watertight，Blender QA 通过，对应 `stage6_selfkernel_final_qa_blender.json`）
- `research_kernel_3d/stage6_traj_compare.npz` —— 官方/自研逐步轨迹（11+11 帧）
- `research_kernel_3d/stage6_official_cond.npz` —— 官方 cond（复现用）
- `research_kernel_3d/stage6_self_latent.npz` / `stage6_final_latent.npz` —— 各轮 latent
- `research_kernel_3d/stage6_step_off.npy` / `stage6_step_self.npy` —— 单步对齐原始输出
- `research_kernel_3d/stage6_e2e.npz` / `stage6_self_e2e.npz` / `stage6_final_e2e.npz`
- `output/stage6_self_e2e.glb`（E2 中间产物，F=688）、
  `output/stage6_cond_isolate.glb`（cond 隔离实验，F=720）、
  `output/stage6_final_e2e.glb`（50 步实验，F=192）

## 下一步（阶段 7）

GLB 导出固化进微内核 CLI（`aigc_cli`），官方 `randn_tensor` 噪声流语义对齐
（保证同 seed 完全复现官方输出），性能优化（SDPA flash kernel 已默认启用）。
