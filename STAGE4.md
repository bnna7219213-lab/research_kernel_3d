# 阶段 4 · flow-matching 采样循环 + 与官方库逐层对齐

日期：2026-09-20 ｜ 实现：`stage4_sampler.py`（采样循环）+ `stage4_diag.py`（逐层对齐诊断）

## 结论

**自研 DiT 与官方库逐层完全一致（cos=1.0000，diff=0.0000）**，flow-matching
欧拉采样循环跑通，输出 latent 有界无 NaN。

## 对齐过程（从 FAIL 到逐层一致）

初始单步对齐 cosine=0.41，经三轮修正达成完全一致：

| 轮次 | 修正 | 结果 |
|------|------|------|
| 1 | 时间注入：t 作为第 0 个 token 拼到序列前（`cat([c,x])`），而非加到每个 token | cos 0.41→0.60 |
| 2 | Attention head 排列：官方 `cat(q,k,v)`→`view(B,N,H,3D)`→`split(D)` 交错打包；CrossAttention 的 q 独立 view、kv 交错 | cos 0.60（block 0-11 已一致，MoE 层仍分歧） |
| 3 | MoE gate：`norm_topk_prob=False`（top-2 权重**不重归一化**，直接用 softmax 原始概率） | **逐层 cos=1.0000** |

## 逐层对齐结果（`stage4_diag.py`）

| 层 | cosine | diff | 说明 |
|----|--------|------|------|
| x_embedder | 1.0000 | 0.0000 | 完全一致 |
| t_embedder | 1.0000 | 0.0000 | 完全一致 |
| block 0/5/10 | 1.0000 | 0.0000 | 编码段（DenseMLP）一致 |
| block 11 | 1.0000 | 0.0000 | 第一个 skip 层一致 |
| block 15 | 1.0000 | 0.0000 | 第一个 MoE 层一致 |
| block 20 | 1.0000 | 0.0001 | 最后一层一致 |
| final_out | **1.0000** | **0.0000** | **最终输出完全一致** |

## 关键结构修正（相比阶段 3）

1. **时间注入**：t 经 `t_embedder` 后**作为第 0 个 token 拼到序列前**（`x = cat([c,x],dim=1)`），
   在 block 里**不参与计算**（`timested_modulate=False`），仅通过自注意力影响其他 token。
   最终输出**去掉第 0 个 token**（`out[:,1:,:]`）。
2. **Attention head 排列**：官方 `SelfAttn` 是 `cat(q,k,v)`→`view(B,N,H,3D)`→`split(D)`
   （每个 head 内 q/k/v 交错打包）；`CrossAttn` 的 q 独立 view、kv 交错（`cat(k,v)`→`view(B,M,H,2D)`→`split(D)`）。
3. **skip 顺序**：`skip_linear(concat(skip,x))` **先 linear 后 norm**（我最初反了）。
4. **MoE gate**：`norm_topk_prob=False`，top-2 权重直接用 softmax 原始概率（不归一化到 sum=1）。
5. **shared_experts**：加在原始输入 `identity` 上（`y = y + shared(identity)`），
   不是加在路由输出上。

## 采样循环（自研）

官方 `FlowMatchEulerDiscreteScheduler`，shift=1.0 时 sigmas 恒等变换：
```
sigmas = linspace(1.0, 1/1000, num_steps)  # 归一化到 [0,1]
sigmas = shift * sigmas / (1 + (shift-1)*sigmas)  # shift=1 时恒等
sigmas = cat([sigmas, [1.0]])
# 欧拉步：prev = sample + (sigma_next - sigma) * model_output
```

10 步采样输出：latent [1,64,64] mean=0.0139 std=0.1481 absmax=0.6015 无 NaN。

## 阶段进度

- [x] 阶段 0 结构探查
- [x] 阶段 1 权重加载器（3.68B 对齐）
- [x] 阶段 2 Conditioner（DINOv2 304.4M，前向自洽）
- [x] 阶段 3 DiT 前向（752 张量 100% 加载，单步自洽）
- [x] **阶段 4 flow-matching 采样循环 + 与官方逐层对齐（cos=1.0）**
- [ ] 阶段 5 ShapeVAE 解码
- [ ] 阶段 6 表面提取
- [ ] 阶段 7 GLB 导出

## 遗留

- 采样循环的 latent 统计与官方完整采样（50 步、真实 conditioner 输出）的数值对照
  需待阶段 5 接 ShapeVAE 后做端到端验证。
- 当前测试用随机 conditioner（[1,16,1024]），非真实 DINOv2 输出——阶段 2 的
  conditioner 前向已验证自洽，但未与官方 conditioner 对齐（属阶段 4+ 的扩展项）。
