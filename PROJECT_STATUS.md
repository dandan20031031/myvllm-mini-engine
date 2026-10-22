# MiniQwenEngine 项目总览

本文把 `reports/STAGE_*.md` 中的阶段结果合并成一份当前状态说明，方便直接了解项目已经实现的能力、实际验证结果和下一步工作。阶段报告仍保留，用于查看具体代码调用链、数据布局和实验细节。

## 项目定位

MiniQwenEngine 是一个面向 Qwen3-0.6B 的 Windows 11 单 GPU 推理引擎实验项目。当前约束是：

- Windows 11 原生运行，PowerShell + MSVC + nvcc；
- 不依赖 Linux、WSL 或 Triton；
- Transformers 只负责模型权重、配置和 tokenizer 加载，以及正确性参考；
- native forward、KV Cache、调度和 CUDA backend 由项目自己实现；
- 先聚焦 Qwen3-0.6B 和单 GPU，再扩展模型、量化和服务能力。

当前验证机器：RTX 5080（SM 12.0）、PyTorch 2.9.1+cu128、Windows CUDA 工具链。完整 Qwen3-0.6B 权重放在本地模型目录，不进入公开仓库。

## 当前架构

```text
CLI / benchmark
        |
NativeQwen3Runner / NativeBatchRuntime
        |
Scheduler -> GenerationRequest -> BlockManager
        |
NativeQwen3 -> KVCachePool -> CUDA kernels / PyTorch reference
        |
PagedAttention / CUDA Graph / Projection Fusion
```

主要代码位置：

| 模块 | 代码位置 | 作用 |
|---|---|---|
| 请求和调度 | `mini_engine/engine/` | 请求状态、调度、batch decode、Graph bucket |
| 物理 KV Cache | `mini_engine/cache/` | block 分配、tensor pool、prefix cache、引用计数 |
| native Qwen3 | `mini_engine/model/` | Qwen3 layer、prefill、增量 decode、模型 runner |
| CUDA dispatch | `mini_engine/kernels/*.py` | torch/CUDA backend 选择和 Python 参数边界 |
| CUDA 源码 | `mini_engine/kernels/csrc/` | PagedAttention、elementwise、数据移动、prefill |
| 量化 reference | `mini_engine/quantization/` | INT8、INT4、FP8 打包和反量化参考路径 |
| benchmark | `benchmarks/` | 单算子、单请求、batch eager/Graph 计时 |

## 已实现功能

### 1. 推理引擎基础

- `waiting/running/finished/aborted` 请求生命周期；
- 固定大小的 `BlockManager`，逻辑 token 到物理 KV block 的映射；
- `KVCachePool`，布局为 `[layer, physical_block, token, kv_head, head_dim]`；
- 跨 block KV 写入、清理、释放和容量检查；
- scheduler 驱动的连续批处理框架；
- 单请求 `NativeQwen3Runner` 和 scheduler 驱动的 `NativeBatchRuntime`；
- EOS、最大输出长度、异常路径和 block 回收。

### 2. Qwen3-0.6B native forward

- Qwen3 embedding、Q/K RMSNorm、RoPE、GQA、SwiGLU；
- Q/K/V/O、Gate/Up/Down、LM head 的 native projection 路径；
- prefill + 单 token decode；
- 自有 Paged KV Cache，不使用 Transformers `past_key_values` 执行 native forward；
- 与小型随机 Qwen3 和完整 Qwen3-0.6B 的 reference 对齐；
- Windows 本地单轮推理和多轮 CLI Chat。

### 3. 已接入的 CUDA 算子

- CUDA PagedAttention：FP16/BF16、GQA、非连续物理 page、长上下文 partition；
- CUDA RMSNorm；
- CUDA Residual + RMSNorm；
- CUDA Residual Add；
- CUDA fused SiLU×Up；
- CUDA Embedding gather；
- CUDA RoPE；
- CUDA 单请求和 batch KV Cache 写入；
- CUDA causal prefill correctness-first kernel；
- 显式 CUDA Linear，底层调用 PyTorch cuBLAS/cuBLASLt。

PyTorch fallback 仍保留，便于对拍和在不具备 CUDA 工具链的环境中运行 reference 测试。

### 4. Batch、Prefix Cache 和 CUDA Graph

- batch decode 一次执行多个请求的 input、position、block table 和模型 forward；
- Prefix Cache 共享完整物理 block；
- block 引用计数和 LRU 淘汰保护；
- prefill 目前仍逐请求执行；
- CUDA Graph shape bucket，固定 batch size、block table 宽度和 GPU buffer 地址；
- inactive slot 使用独立 scratch KV pages；
- Graph capture/replay 已接入真实 `NativeBatchRuntime.decode_step`；
- Graph capture 路径跳过会触发 host 同步的 metadata reduction 和 kernel launch error query。

### 5. Projection Fusion

阶段 18 已加入可选 projection fusion：

- Q/K/V 权重拼接为一次 QKV 矩阵乘，再切分成 Q、K、V；
- Gate/Up 权重拼接为一次矩阵乘，再切分给 SiLU×Up；
- 默认路径不改变；
- 量化 reference 暂不与 projection fusion 混用；
- CLI 和 benchmark 使用 `--fuse-projections` 开启。

这属于 projection-level fusion，矩阵乘底层仍由 PyTorch/cuBLAS/cuBLASLt 执行，不等于已经实现专用 Tensor Core GEMM。

### 6. 多精度和量化 reference

- FP32、FP16、BF16 浮点路径；
- INT8 weight-only reference；
- packed INT4 reference；
- FP8 E4M3/E5M2 weight-only reference；
- NVIDIA 精度能力探测和能力矩阵；
- 明确区分权重存储格式、激活格式、累加格式、KV 格式和真实计算后端。

当前 INT4/FP8 reference 会反量化后执行普通 Linear，不能直接称为量化加速。NVFP4、MXFP8、FP8/INT8 KV Cache 尚未完成。

## 已完成的验证

当前阶段 19 的 CUDA capture-safe 修改后，全量回归为：

```text
156 passed, 2 warnings
```

warning 是当前 Windows 工具链与 PyTorch CUDA runtime 的版本提示，以及 setuptools 的 MSVC 环境提示。

完整 Qwen3-0.6B 已在 RTX 5080 BF16 全 CUDA backend 下完成文本生成验收。当前可本地运行 CLI Chat，也可以直接运行 batch benchmark。

## Benchmark 结果

### PagedAttention 单算子

配置：RTX 5080、FP16、Qwen3-0.6B attention shape、GQA、`block_size=256`。

| batch/context | 当前 CUDA kernel | 项目 PyTorch fallback | 相对 fallback |
|---|---:|---:|---:|
| 1 / 128 | 0.0474 ms | 0.3779 ms | 7.97x |
| 8 / 512 | 0.0951 ms | 2.6245 ms | 27.60x |
| 32 / 2048 | 0.6999 ms | 10.2962 ms | 14.71x |

同配置下，MiniVLLM Triton 分块 kernel 约为 `0.019 / 0.017 / 0.377 ms`，因此当前 PagedAttention 仍慢于 MiniVLLM，差距约为 `2.0x～5.6x`。当前主要短板是 vectorized KV load、batch=1 并行度和 flash-decoding。

### 单请求 Qwen3-0.6B

已有完整模型基线：

```text
GPU: RTX 5080
dtype: BF16
TTFT: 约 22～24 ms
TPOT: 约 20～21 ms/token
```

阶段 18 的单请求 projection fusion 方向性测试，在相同脚本下得到约 `3.7%～8.6%` 的 TPOT 改善；这个结果受 Windows/GPU 调度抖动影响，需要以 batch Graph benchmark 为主要判断依据。

### Batch eager / CUDA Graph

配置：RTX 5080、Qwen3-0.6B、BF16、3 条请求、`block_size=16`、32 token 输出预算、warmup=1、repetitions=3。计时包含逐请求 prefill 和 scheduler 生命周期。

| 模式 | median total | median aggregate TPS |
|---|---:|---:|
| eager | 438.38 ms | 189.33 tokens/s |
| CUDA Graph | 175.15 ms | 473.87 tokens/s |
| Projection Fusion + CUDA Graph | 159.88 ms | 519.15 tokens/s |

CUDA Graph 相对 eager 约 `2.50x`。这是当前项目 runtime 的真实 batch 基线，不是只统计 decode kernel，也不是和 MiniVLLM/vLLM 完全同口径的最终对比。

## 当前短板

1. **Prefill 仍逐请求执行**：还没有 chunked prefill 和真正的 batched prefill。
2. **Prefix Cache 还没有完全跳过 prompt 计算**：目前已经共享物理 block 和生命周期，但 next logits 复用路径仍需完善。
3. **PagedAttention 仍有优化空间**：需要 vectorized load、block size sweep、split-KV/flash-decoding。
4. **Linear 仍主要依赖 cuBLAS**：projection fusion 减少了矩阵乘次数，但还不是专用高性能 GEMM。
5. **量化 GEMM 尚未进入 native 主路径**：INT4/FP8 目前主要是 reference dequant。
6. **Sampling 仍是 PyTorch reference**：尚未实现 CUDA greedy、top-k、top-p 融合采样。
7. **服务层尚未完成**：没有 Windows HTTP/OpenAI/SSE 服务。
8. **Graph benchmark 还需扩展**：当前已有真实 batch 路径，但还需要 batch=1/2/4/8、FP16、P50/P95、不同 block size 和 prefill/decode 分离统计。
9. **行业级公平对比尚未完成**：还需要统一模型、精度、prompt、输出长度、batch、采样和计时范围，对比 MiniVLLM、vLLM、SGLang 或 TensorRT-LLM。

## 后续路线

### 第一优先级：dandan_ops 正式接入

`dandan_ops` 是独立开发的高性能算子库，当前已经包含：

- FP16 HGEMM；
- SM120 TMA 路径；
- `backend="cuda"/"tma"/"auto"` 选择；
- W4A16 `QuantizedLinear`；
- CUDA Graph benchmark 和 Compute Sanitizer 验证记录。

接入计划：

1. 移除 MiniQwenEngine 中写死的本机 `file:///D:/...` 依赖；
2. 通过可选依赖、环境变量或本地 editable install 发现 `dandan_ops`；
3. 建立 `DandanLinear` adapter，保持 `nn.Linear` 权重 `[N,K]` 布局；
4. 先接 Q/K/V/O、Gate/Up/Down 和 LM head 的 FP16 inference path；
5. 先支持 `auto`，再分别验收 V8 HGEMM 和 V10 TMA；
6. 加入 dandan_ops 与 torch/cuBLAS 的数值对拍、Graph replay 和端到端 benchmark；
7. 再接入 `QuantizedLinear`，将 W4A16 从 reference 路径替换成真正的 Tensor Core path；
8. 只有在独立仓库和公开安装方式稳定后，才把它作为公开项目推荐后端。

### 第二优先级：PagedAttention

- vectorized KV load；
- block size `16/32/64/128/256` sweep；
- batch=1 split-KV；
- flash-decoding partition/merge；
- 重新做 MiniVLLM 同口径算子对比。

### 第三优先级：系统能力

- chunked/batched prefill；
- Prefix Cache 真正跳过已缓存 prompt；
- CUDA sampling；
- 更完整的 request cancellation、异步资源释放和服务 API；
- batch=1/2/4/8 的 Graph benchmark 和 P50/P95。

### 第四优先级：精度和高级解码

- INT4/FP8 Tensor Core GEMM；
- FP8/INT8 KV Cache；
- NVFP4/MXFP8 能力验证和 Windows backend；
- 投机解码：greedy 多 token 验证、KV 回退、prompt lookup、小 draft model、随机采样校正。

## 当前结论

项目已经从基础骨架发展为可以在 Windows RTX 5080 上运行完整 Qwen3-0.6B 的 native 推理引擎：有自有 KV Cache、CUDA PagedAttention、CUDA elementwise、batch decode、Prefix Cache、真实 CUDA Graph 和 projection fusion。

当前最有竞争力的方向是把 `dandan_ops` 接入 Linear/GEMM/量化主路径，再结合 PagedAttention 和 prefill 优化。现阶段不应宣称已经超过 MiniVLLM 或 vLLM；但系统架构、CUDA 可替换性和后续自研算子空间已经具备继续对比和开源迭代的基础。
