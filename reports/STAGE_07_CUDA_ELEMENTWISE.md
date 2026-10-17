# 阶段 07：Windows CUDA RMSNorm、Residual 和 SwiGLU

## 1. 完成内容和边界

实现 FP16/BF16 的四个 CUDA 入口，实际接入 native Qwen3 的 prefill 和 decode：

- RMSNorm：input、Q/K head、final norm；
- Residual+RMSNorm：attention 输出与原 residual 相加，再做 post-attention norm；
- Residual add：MLP 输出后的 residual 相加；
- fused SiLU×Up：MLP SwiGLU 激活部分。

保留默认 PyTorch 路径，通过 elementwise_backend="cuda" 显式启用。已运行小模型 Transformers 和 native reference 对齐、INT4/FP8 组合、CUDA Graph 与单算子 benchmark。完整 Qwen3-0.6B 权重仍未验收，本阶段不代表端到端部署或 vLLM 性能达标。

## 2. 文件逐项解读

| 文件 | 职责 |
|---|---|
| mini_engine/kernels/elementwise.py | Python torch/cuda dispatch，统一返回约定 |
| mini_engine/kernels/csrc/elementwise.cu | FP32 reduction、低精度舍入、融合和向量化 kernel |
| mini_engine/kernels/csrc/bindings.cpp | CUDA/device/dtype/shape/eps/contiguous/inference 校验与 pybind 导出 |
| mini_engine/kernels/cuda_extension.py | 同一 extension 加入 elementwise.cu，继续使用 D 盘 build 目录 |
| mini_engine/kernels/__init__.py | 导出可选择后端的算子 |
| mini_engine/model/native_qwen3.py | Norm/MLP/DecoderLayer 的实际 CUDA 调用点 |
| mini_engine/model/native_runner.py | 模型加载前准备 extension，传递 elementwise_backend |
| mini_engine/__main__.py | --elementwise-backend torch/cuda |
| tests/test_elementwise.py | CPU fallback 的融合返回值及非原地语义 |
| tests/test_cuda_elementwise.py | 数值、边界、stream、Graph、Transformers/native/量化组合 |
| benchmarks/benchmark_elementwise.py | 两种计时口径、实测 JSON |
| reports/stage07_elementwise_summary.json | 本机 FP16/BF16 紧凑实测摘要 |

## 3. 公共接口

~~~python
rms_norm(x, weight, eps=1e-6, backend="torch")
residual_rms_norm(x, residual, weight, eps=1e-6, backend="torch")
residual_add(x, residual, backend="torch")
silu_and_mul(gate, up, backend="torch")
~~~

RMSNorm 按最后一维归一化，支持 [hidden]、[tokens, hidden] 和 [tokens, heads, head_dim] 等连续形状。

residual_rms_norm 返回 (normalized, summed)，两者都是新 tensor，不修改输入。summed 保留给后续 residual 分支使用，避免模型先单独相加一次，再在 norm kernel 中重复相加。

CUDA 输入需在同一设备、连续且 dtype 一致；支持 FP16/BF16。推理算子不提供 backward；grad enabled 且输入或权重 requires_grad 时会明确拒绝。模型 native forward 已使用 inference_mode。

## 4. 数值语义为什么这样处理

### 4.1 RMSNorm

~~~text
variance = mean(float32(x)^2)
normalized_fp32 = float32(x) * rsqrt(variance + eps)
normalized_low = cast_to_input_dtype(normalized_fp32)
output = normalized_low * weight
~~~

Qwen3 会在乘 norm weight 前转回输入 dtype。因此 CUDA 也保留这个中间舍入点，而不是把所有乘法合并后只在最后舍入。

### 4.2 Residual+RMSNorm

~~~text
summed = cast_to_input_dtype(float32(x) + float32(residual))
normalized = RMSNorm(summed)
return normalized, summed
~~~

先对 residual sum 做 FP16/BF16 舍入，再计算 FP32 方差。这样 post-attention norm 的输入与原 eager 模型一致。输入不会原地改变，避免 residual aliasing 破坏后续计算。

### 4.3 SwiGLU

~~~text
silu_low = cast_to_input_dtype(SiLU(float32(gate)))
output = cast_to_input_dtype(float32(silu_low) * float32(up))
~~~

保持原 SiLU tensor 的低精度舍入。融合消除中间 global memory tensor，不改变这个舍入约定。

## 5. CUDA 内部算法

### 5.1 Norm

一个 CTA 处理一行，256 threads / 8 warps：

1. 沿最后一维分工读取，计算 FP32 平方和；
2. warp shuffle reduction；
3. 仅用 8 个 shared float 汇总 warp 部分和；
4. 求均方根倒数；
5. 写归一化输出；融合版本同时保留 residual sum。

任意正 hidden size 使用循环覆盖，包括 17、3072、12345 等维度；空 batch 直接返回空输出，不 launch。当前没有针对 16/128 维 Q/K norm 使用专门 warp-only 配置，是后续可优化项。

### 5.2 Residual / SiLU×Up

grid-stride loop 覆盖所有元素。元素数能被 4 整除且输入地址 8-byte 对齐时，采用每线程 4 个 half/BF16 的向量化读写；否则使用标量路径。

“连续 tensor”不一定地址对齐：storage_offset=1 的连续切片也必须安全，因此实测覆盖此路径。

### 5.3 Stream、内存和 Graph

所有 kernel 使用 CUDAGuard 和 PyTorch 当前 CUDA stream，分配输出后异步 launch，不在 C++ 里做设备同步。launch 后检查 CUDA 错误。输入数据准备与跨 stream 依赖由 caller 按 PyTorch 语义建立。

算子可 capture/replay，测试包含修改输入后的 Graph 重放正确性。整个 native runner 仍有 tokenizer、Python block 管理和 cache 写入同步，尚不能声称整条 decode 已 Graph 化。

## 6. Native Qwen3 调用链

~~~text
NativeQwen3.forward
  -> embedding (仍为 PyTorch)
  -> 每个 DecoderLayer
       input RMSNorm (CUDA)
       attention
         q/k projections (浮点 CUDA 库 / 可选量化 reference)
         q/k head RMSNorm (CUDA)
         RoPE + KV write (仍为 PyTorch)
         SDPA prefill / CUDA paged decode
         o projection
       fused attention residual + post-attention RMSNorm (CUDA)
         -> normalized + summed
       MLP
         gate/up projection
         fused SiLU×Up (CUDA)
         down projection
       residual_add(summed, MLP output) (CUDA)
  -> final RMSNorm (CUDA)
  -> lm_head (浮点 PyTorch Linear)
~~~

构造时传递 elementwise_backend，每个 RMSNorm、MLP 和 DecoderLayer 保存相同选择。参数名字保持与 HuggingFace state_dict 一致，量化替换的七类 projection 也不改变。

## 7. Windows 使用

~~~powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python -m mini_engine --model "D:\models\Qwen3-0.6B" --prompt "介绍一下 KV Cache" --device cuda --dtype bf16 --prefill-backend auto --decode-backend cuda --elementwise-backend cuda --local-files-only
~~~

权重路径是示例。加载前预编译 extension，避免编译时间计入 TTFT。可附加 --quantization int4 或 fp8_e4m3，但该量化 Linear 仍为反量化 reference。

## 8. 验证结果

本机 Windows 11、RTX 5080、PyTorch 2.9.1+cu128、nvcc 13.3、MSVC：

~~~text
新增 elementwise/fallback 专项：52 passed
全量（含 CUDA paged attention 和量化）：126 passed, 2 warnings
~~~

两条 Python warning 为 Toolkit 13.3 / PyTorch CUDA 12.8 差异提示和 setuptools 使用私有 MSVC 环境接口。没有跳过 CUDA 测试。

覆盖：

- FP16/BF16；1、17、128、1024、3072、12345 等 hidden；
- 零值、小值、residual 抵消、非单位 norm 权重；
- scalar pointwise、空 batch、零长度 pointwise；
- 非连续输入、CPU 输入、dtype/device/shape 错误、非法 eps、autograd 拒绝；
- 非默认 stream，CUDA Graph replay；
- 两层随机 Qwen3 prefill 加六次 decode，对齐 Transformers eager 和 native torch；
- INT4/FP8 projection 与 CUDA elementwise/attention 的组合；
- 实际检查每次 forward 的 7 次普通 norm、2 次融合 norm、2 次 residual add、2 次 SwiGLU 调用。

测试命令：

~~~powershell
. .\scripts\enter_project.ps1
$env:MINI_ENGINE_TEST_CUDA = "1"
.\.venv\Scripts\python -m pytest -q -o cache_dir=build/pytest_cache
~~~

## 9. 单算子实测

~~~powershell
.\.venv\Scripts\python benchmarks\benchmark_elementwise.py --dtype bf16
.\.venv\Scripts\python benchmarks\benchmark_elementwise.py --dtype fp16 --output benchmarks\results\elementwise_fp16.json
~~~

5 次 warmup；每次 Graph 含 32 个调用、测量 20 次重放；取每调用 device timeline 中位数。另记录 5 组 eager wall 样本。Graph 时间包含节点执行/调度，不称为纯 kernel 指令时间；eager wall 包含 Python、分配和末尾同步。

BF16，单位 microseconds：

| rows×hidden | 算子 | torch Graph | CUDA Graph | torch/CUDA |
|---|---|---:|---:|---:|
| 1×1024 | RMSNorm | 9.669 | 1.797 | 5.38 |
| 1×1024 | Residual+RMSNorm | 10.558 | 2.224 | 4.75 |
| 1×1024 | Residual add | 0.843 | 0.944 | 0.89 |
| 1×1024 | SiLU×Up | 1.903 | 1.008 | 1.89 |
| 128×3072 | RMSNorm | 17.443 | 3.504 | 4.98 |
| 128×3072 | Residual+RMSNorm | 18.690 | 4.850 | 3.85 |
| 128×3072 | Residual add | 1.253 | 1.320 | 0.95 |
| 128×3072 | SiLU×Up | 2.671 | 1.541 | 1.73 |

FP16 两个对应形状的 RMSNorm 比值为 5.38 / 5.10，融合 norm 为 4.78 / 3.94，SiLU×Up 为 1.84 / 1.76。原始 JSON 保存所有样本，紧凑摘要在 stage07_elementwise_summary.json。

独立 residual add 在这些 shape 下略慢，保留这个结果，不保证所有自研 kernel 都有收益。当前收益主要来自 norm/激活融合、减少参考路径的 kernel 数；不能外推为整模型加速比或超过 vLLM。

## 10. 未完成项与下一阶段

- 完整权重质量与端到端 TPOT 尚未测量。
- Prefill 仍为 SDPA，本机 wheel 不含 Flash backend。
- Embedding、RoPE、KV 写入尚未自研 CUDA。
- 浮点 GEMM 仍由 PyTorch CUDA 库执行；量化 Tensor Core GEMM 未实现。
- sampling、continuous batching 执行、服务 API 尚未完成。
- Norm 的小维度 warp mapping 和跨层 residual 融合仍可优化。

阶段 08 按计划实现 Embedding、RoPE 和 KV scatter，复用 block table，重点消除每层 CPU/GPU metadata 同步。阶段完成后继续测试、报告和本地 Git commit。
