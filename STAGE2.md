# 阶段 2 · Conditioner（DINOv2）手写前向

日期：2026-09-18 ｜ 实现：`stage2_conditioner.py`（纯 torch + numpy，不 import hy3dshape/comfy/diffusers）

## 结构推断（全部来自 safetensors 键名/形状，非抄官方代码）

键空间：`conditioner.main_image_encoder.model.*`，共 439 张量。

| 证据（权重键 / 形状） | 推断 |
|---|---|
| `embeddings.patch_embeddings.projection.weight [1024,3,14,14]` | patch embed = Conv2d(3→1024, k=14, s=14)，patch=14 |
| `embeddings.position_embeddings [1,1370,1024]` | 1370 = 1 CLS + 37² patch，输入 518×518（37×14=518），CLS 在最前 |
| `embeddings.cls_token [1,1,1024]` + `embeddings.mask_token [1,1024]` | 带 CLS token；mask_token 为 MAE 式训练遗留，推理前向不使用（仅占位加载） |
| `encoder.layer.0..23` | 24 层 encoder，hidden=1024 |
| 每层 `attention.attention.{query,key,value} [1024,1024]` + `attention.output.dense` | 标准 MHA，q/k/v 分离 + 输出投影；1024=16 头×64（头数按 DINOv2-large 惯例，权重中不存） |
| 每层 `norm1`/`norm2` 在 attention/mlp 之外 | 预归一化（pre-norm）残差结构 |
| 每层 `layer_scale1.lambda1` / `layer_scale2.lambda1 [1024]` | LayerScale：两条残差分支各乘一个逐通道标量向量 |
| `mlp.fc1 [4096,1024]` / `mlp.fc2 [1024,4096]`（fc1 单输出 4096，无 w12 拆分、无 2×hidden） | 标准 GELU MLP（4× 扩张），**不是 SwiGLU**（SwiGLU 的 fc1 会是 2/3 比例或拆成 gate+up 两个矩阵） |
| `model.layernorm [1024]` | 24 层之后还有一次最终 LayerNorm |

整体 = DINOv2-large ViT（embed 1024 / 24 层 / patch 14 / LayerScale / pre-norm）。

## 权重键映射

state_dict 键去掉前缀 `conditioner.main_image_encoder.model.` 后与本实现模块树**一一对应、strict=True 加载成功**：

```
embeddings.cls_token            -> Embeddings.cls_token
embeddings.mask_token           -> Embeddings.mask_token（加载但不参与前向）
embeddings.patch_embeddings.projection.{weight,bias} -> PatchEmbed.projection (Conv2d)
embeddings.position_embeddings  -> Embeddings.position_embeddings
encoder.layer.{i}.norm1/norm2   -> EncoderLayer.norm1/norm2 (LayerNorm)
encoder.layer.{i}.attention.attention.{query,key,value} -> SelfAttention.query/key/value
encoder.layer.{i}.attention.output.dense -> Attention.output["dense"]
encoder.layer.{i}.layer_scale1/2.lambda1 -> LayerScale.lambda1
encoder.layer.{i}.mlp.fc1/fc2   -> Mlp.fc1/fc2
layernorm                       -> 最终 LayerNorm
```

前向顺序（从键名拓扑推断）：

```
x = PatchEmbed(img) ; x = cat([CLS, x]) + pos_emb
for each of 24 layers:
    x = x + ls1( attn( norm1(x) ) )
    x = x + ls2( mlp( norm2(x) ) )
x = final_layernorm(x)   # [B, 1370, 1024]
输出：CLS = x[:,0]，patch tokens = x[:,1:]
```

## 自检数值（实跑 `python stage2_conditioner.py`，CUDA / fp32）

输入：numpy 合成的 518×518 RGB 测试图（渐变+圆斑+噪声），ImageNet mean/std 归一化。

| 输出 | shape | mean | std | 平均 token L2 | min/max | NaN/Inf |
|---|---|---|---|---|---|---|
| CLS | (1, 1024) | +0.0162 | 1.4744 | 47.16 | -6.91 / +5.70 | 无 |
| patch tokens | (1, 1369, 1024) | +0.0242 | 1.3756 | 43.98 | -29.32 / +27.54 | 无 |
| 全部 tokens | (1, 1370, 1024) | +0.0242 | 1.3756 | 43.98 | -29.32 / +27.54 | 无 |

自洽性断言（全部通过）：
- 加载：`strict=True` 无 missing/unexpected；参数量 304.4M，与阶段 1 的 conditioner 组一致。
- shape：CLS [B,1024]、patch [B,1369,1024]，与位置编码 1370=1+37² 对账。
- 数值：无 NaN/Inf、有界（|max| < 30，最终 LayerNorm 后量级正常）、std > 1 未塌缩。
- 确定性：同输入两次前向 `allclose(atol=1e-5)`。
- 区分度：两张不同测试图 CLS 余弦相似度 0.9955 < 1（对输入敏感；值偏高是因为两张合成图内容相近，属预期）。

## 与官方结构已知的差异点 / 待阶段 3+ 核对

1. **GELU 变体**：用了 `F.gelu` 默认（erf 精确版）；若官方是 tanh 近似（`approximate="tanh"`），数值会有小差异，阶段 3 对齐时再核。
2. **预处理**：mean/std 用 ImageNet 惯例（0.485/0.456/0.406、0.229/0.224/0.225）——DINOv2 标准；官方管线若用别的归一化需调整。
3. **输入尺寸**：写死 518（与 position_embeddings 1370 一致）；非 518 输入需要位置编码插值（未实现）。
4. **mask_token / attention mask**：推理路径不用 mask，全量注意力；若官方条件器对多视角图做 packing/masking，此处在阶段 4 接多图时再补。
5. **数值对齐未做**：本阶段只保证 forward 跑通且数值自洽；与官方 DINOv2 输出的逐元素对齐（cosine 相似度对照）是阶段 3+ 的事。

## 结论

阶段 2 完成：从权重键反推出 DINOv2-large 结构，手写 forward 在 strict 加载下跑通 518×518 图，
输出 CLS + patch tokens 共 [1, 1370, 1024]，数值健康、确定、对输入敏感，可作为下游 DiT 交叉注意力的条件源（`model.attn2.to_k/v` 输入 1024 维与此吻合）。
