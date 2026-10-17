# 阶段 06 报告：INT4、FP8 与 NVIDIA 精度能力矩阵

## 1. 阶段目标

本阶段把量化从独立的 W8A16 示例推进到可以真正替换 native Qwen3 decoder projection 的参考后端，同时明确区分 NVIDIA 精度的“格式存在”和“已经有 CUDA 加速”。

完成内容：

- packed INT8 weight-only reference；
- group-wise packed INT4 weight-only reference；
- FP8 E4M3FN/E5M2 scaled weight-only reference；
- native Qwen3 decoder Linear 替换；
- CLI 量化选择；
- FP8、NVFP4、MXFP8 能力探测报告；
- CPU/GPU、极值权重、状态保存和模型 decode 测试。

## 2. 新增和修改文件

```text
mini_engine/quantization/backends.py
mini_engine/quantization/linear.py
mini_engine/quantization/capabilities.py
mini_engine/quantization/__init__.py
mini_engine/config.py
mini_engine/model/native_qwen3.py
mini_engine/model/native_runner.py
mini_engine/__main__.py
tests/test_quantization.py
reports/precision_capabilities.json
reports/STAGE_06_QUANTIZATION_PRECISION_BACKENDS.md
```

## 3. 格式和实现等级

| 格式 | 权重存储 | native Linear | CUDA 低精度 GEMM | 当前状态 |
|---|---|---|---|---|
| INT8 | per-output-channel INT8 + FP32 scale | 已集成 | 未实现 | reference only |
| INT4 | 两个 signed nibble/byte + group scale | 已集成 | 未实现 | reference only |
| FP8 E4M3FN | per-output-channel FP8 + FP32 scale | 已集成 | 未实现 | reference only |
| FP8 E5M2 | per-output-channel FP8 + FP32 scale | 已集成 | 未实现 | reference only |
| MXFP8 | 未实现 | 未实现 | 未实现 | reserved |
| NVFP4 | 未实现 | 未实现 | 未实现 | reserved |
| TF32 | 不存储权重 | 尚未作为 GEMM policy 接入 | 不是量化格式 | policy reserved |

特别注意：PyTorch 暴露 `float4_e2m1fn_x2` dtype 或 GPU 属于 SM 12.0，只能说明存在候选能力，不能证明已经实现 NVIDIA NVFP4 的 block scale、额外 scale、打包布局和 Tensor Core GEMM。

## 4. QuantConfig

文件：`mini_engine/config.py`

`QuantConfig` 现在允许：

```text
fp32 / fp16 / bf16
int8 / int4
fp8 / fp8_e4m3 / fp8_e5m2
mxfp8 / nvfp4
```

`fp8` 规范化为 `fp8_e4m3`。`tf32` 不允许作为 `weight_dtype`，因为 TF32 是 FP32 GEMM 的计算策略，不是权重存储格式。

配置会校验 activation dtype、group size 和对称量化选项。当前只实现对称量化；设置 `symmetric=False` 会明确报错，避免静默使用错误的 scale 语义。

## 5. INT8 后端

文件：`mini_engine/quantization/backends.py`

`Int8WeightOnlyBackend` 使用每个输出通道一个 scale：

```text
scale[row] = max(abs(weight[row])) / 127
qweight    = round(weight / scale).clamp(-127, 127).int8
```

计算时：

```text
weight_fp32 = qweight * scale
output      = Linear(activation_fp32, weight_fp32)
```

当前是正确性 reference，不是 INT8 Tensor Core GEMM。后续应替换成 fused dequant + cuBLASLt/CUTLASS 或自研 kernel。

## 6. INT4 打包逻辑

`Int4WeightOnlyBackend` 使用 signed `[-7, 7]` 对称量化，默认 group size 128：

```text
每组 scale = max(abs(group)) / 7
每个权重 -> 4-bit signed nibble
两个 nibble -> 一个 uint8
```

低 nibble 保存更早的输入列，高 nibble 保存后一个输入列。输入特征数不是 group size 的整数倍时，最后一组用零填充，但 `in_features` 单独保存，反量化后裁剪回原形状。

例如：

```text
[-7, -1, 0, 7] -> 0xF9, 0x70
```

这样可以验证 nibble 顺序、padding 和实际显存压缩率。当前 linear 仍先反量化到 FP32，后续才是高性能 W4A16 dequant GEMM。

## 7. FP8 E4M3/E5M2

`FP8WeightOnlyBackend` 为每个输出通道保存 FP32 scale，并把缩放后的权重转换为：

```text
torch.float8_e4m3fn
torch.float8_e5m2
```

这两个格式分别适合不同动态范围/精度取舍。当前路径是：

```text
FP32 weight
  -> per-channel scale
  -> FP8 storage
  -> FP32 dequant
  -> FP32 Linear
  -> activation dtype
```

它是 weight-only reference，不是 W8A8，也没有使用 FP8 Tensor Core。报告和 CLI 都会明确标记这一点。

## 8. Native Linear 集成

文件：`mini_engine/quantization/linear.py`

`ReferenceQuantLinear` 会把 native Qwen3 decoder layer 中的 7 类 projection 替换为 packed buffer：

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
```

embedding、所有 norm、residual、lm_head 和 KV Cache 保持高精度。原始浮点 projection weight 不保留为 Parameter，避免量化后仍占用完整浮点存储。

调用链：

```text
NativeQwen3.quantize(config)
  -> quantize_model_linears
  -> create_backend
  -> pack every decoder projection
  -> ReferenceQuantLinear
  -> native forward calls module.forward
  -> dequantize + reference Linear
```

替换前先完成所有 pack，任何格式错误会在模型 mutation 前失败，避免留下半量化模型。

## 9. CLI 用法

Windows PowerShell：

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python -m mini_engine `
  --model "D:\models\Qwen3-0.6B" `
  --prompt "介绍一下 KV Cache" `
  --device cuda `
  --dtype bf16 `
  --quantization int4 `
  --quant-group-size 128 `
  --decode-backend cuda `
  --local-files-only
```

FP8 选择：

```text
--quantization fp8_e4m3
--quantization fp8_e5m2
```

`--quantization nvfp4` 和 `--quantization mxfp8` 当前会明确报 reserved，因为对应 CUDA packing、scale 语义和 GEMM 尚未实现。

## 10. 能力报告

命令：

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python -m mini_engine.quantization.capabilities `
  --device cuda --probe `
  --output reports\precision_capabilities.json
```

本机报告记录：

```text
OS                 Windows
GPU                NVIDIA GeForce RTX 5080
compute capability 12.0
PyTorch            2.9.1+cu128
FP8 E4M3/E5M2      dtype 与转换 probe 可用，reference only
NVFP4              dtype candidate 存在，但 reserved
MXFP8              未提供 dtype，reserved
BF16               supported
```

完整报告见 [precision_capabilities.json](precision_capabilities.json)。

## 11. 测试结果

新增量化测试共 31 个，覆盖：

- INT8/INT4/FP8 CPU 与 CUDA pack/dequant/linear；
- INT4 nibble 顺序与 padding；
- 极小/极大权重的 scale 和有限值；
- module state_dict、dtype/device movement、无原始 Parameter；
- native Qwen3 projection replacement 和 cached decode；
- reserved NVFP4/MXFP8 fail-before-mutation；
- FP8 dtype capability probe；
- GPU 量化 activation 和生成路径。

本阶段执行结果：

```text
quantization tests: 31 passed
```

之前阶段回归测试和本阶段测试都应在 commit 前一起运行。所有测试不下载完整模型权重。

## 12. 限制和下一步

本阶段没有实现：

- INT8/INT4/FP8 Tensor Core GEMM；
- NVFP4/MXFP8 packing 和 scale；
- 低精度 KV Cache；
- 量化后完整 Qwen3-0.6B 质量评测；
- 量化端到端 benchmark。

下一阶段按全算子 CUDA 计划，先替换所有 RMSNorm、Residual 和 SwiGLU，再继续做 Embedding、RoPE、KV 写入和 prefill。量化 CUDA GEMM 会在 Linear/LM head 阶段单独接入，避免把 reference 结果误报成性能后端。
