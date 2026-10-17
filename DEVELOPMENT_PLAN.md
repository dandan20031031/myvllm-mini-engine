# MiniQwenEngine 开发方案

## 项目目标

MiniQwenEngine 是一个面向单 GPU 的小型语言模型推理引擎，第一阶段聚焦 Qwen3-0.6B。项目借鉴 MiniVLLM/vLLM 的请求调度和分页 KV Cache 思路，同时保留清晰的 PyTorch fallback，后续逐步替换为自研 CUDA 算子。

第一阶段的竞争力来自一个明确的优化目标：在 Qwen3-0.6B 的 decode 阶段，结合 Paged KV Cache、continuous batching 和多精度 KV Cache，降低显存占用与每 token 延迟。

## 范围与取舍

### 第一阶段支持

- Qwen3-0.6B
- 单 GPU
- FP16/BF16
- 单请求和 continuous batching
- Paged KV Cache 的分配、回收和 block table
- prefill/decode 状态分离
- 流式 token 生成接口的基础抽象
- PyTorch fallback 算子
- 可插拔量化后端接口

### 暂不支持

- 多 GPU、张量并行、流水线并行
- 多模型通用适配
- 一开始同时实现 GPTQ、AWQ、FP8、INT4 全部格式
- 自己实现 GEMM

矩阵乘优先使用 PyTorch/cuBLASLt；自研 CUDA 优先放在 decode paged attention、RMSNorm、RoPE、SwiGLU 和 sampling。

## 架构

```text
serving / CLI
       |
LLMEngine -- Scheduler -- Request
       |          |
       |       BlockManager -- KVCache
       |
ModelRunner -- Qwen3 model -- QuantBackend
       |
Kernels: paged attention / rmsnorm / rope / swiglu / sampling
```

目录职责：

```text
mini_engine/
  model/          Qwen3 配置和 Transformers runner
  engine/         引擎、请求、调度器
  cache/          block allocator 和 KV Cache 抽象
  kernels/        PyTorch fallback，后续接 CUDA extension
  quantization/   BF16、W8A16、W4A16 的后端接口
  serving/        服务层预留
tests/            不依赖模型下载的单元测试
```

## 迭代路线

1. **模型基线**：加载 Qwen3-0.6B，支持 tokenizer、chat template、`enable_thinking=False`，与 Transformers 对齐。
2. **增量解码**：完成 prefill/decode 分离和 KV Cache。
3. **调度与分页缓存**：加入 waiting/running/finished 状态、continuous batching、block table 和显存回收。
4. **CUDA 优化**：先 RMSNorm/RoPE，再 decode paged attention、fused SwiGLU、sampling，最后评估 CUDA Graph。
5. **多精度**：先 W8A16，再 W4A16，最后做 BF16/INT8 KV Cache 的按压力选择。

## 多精度设计

模型层只依赖 `QuantBackend` 接口，不直接依赖某种量化格式：

```python
class QuantBackend:
    def pack(self, weight): ...
    def linear(self, x, packed_weight): ...
```

实现顺序为 BF16/FP16、W8A16、W4A16。embedding、norm、residual 和 lm_head 默认保留高精度，量化主要应用于 Linear。

## 主要创新点

根据 batch size、上下文长度和显存压力选择 KV Cache 精度及 decode kernel：

- 显存充足：BF16 KV Cache；
- 显存压力较高：INT8 KV Cache；
- 短序列、小 batch：低延迟 kernel；
- 长序列、大 batch：高吞吐 kernel。

## 验收与 benchmark

至少比较 Transformers、MiniVLLM、PyTorch baseline 和自研 engine，记录 TTFT、TPOT、tokens/s、显存占用、最大并发数、P50/P95 延迟以及量化后的生成质量。每个 kernel 额外记录 CUDA event latency、有效带宽、occupancy 和 L2 命中率。

## 当前实现状态

当前提交已经完成单请求 native Qwen3 增量 forward，并接入第一版 Windows CUDA Paged Attention 和单算子 benchmark：prefill 使用 PyTorch SDPA，decode 可选项目 CUDA kernel 或 PyTorch fallback，编译、缓存和 benchmark 产物位于项目 D 盘目录。下一步是根据 profiling 结果优化 kernel，再做 HTTP 服务、continuous batching model execution 和 vLLM benchmark。

## 路线修订：全算子 CUDA、NVIDIA 精度和阶段提交

本节依据用户最新要求修订；与前面的初始路线冲突时，以本节为准。

### 运行和存储约束

- Windows 11 原生，PowerShell + MSVC + nvcc，不依赖 Linux、WSL 或 Triton。
- 源码、虚拟环境、模型、构建输出、临时文件和可配置缓存均保存在当前 D 盘项目中。
- 每类模型算子逐步接入明确的 CUDA 后端，保留 PyTorch reference 用于验证。Tokenizer、HTTP 和调度元数据仍由 CPU 处理。

### CUDA 覆盖清单

- Embedding gather。
- 所有 RMSNorm，包括 input、post-attention、final 和 Q/K head norm。
- Residual add，可与 RMSNorm 融合。
- RoPE、位置数据计算、KV Cache 写入，可组合融合。
- Prefill 使用 Windows 可用的 Flash 风格 CUDA；现有 SDPA 保留为参考路径。
- Decode 使用已实现的 CUDA Paged Attention，继续优化 GQA 和访存。
- SwiGLU 的 SiLU 和逐元素乘法。
- Q/K/V/O、Gate/Up/Down projection 和 LM head：明确使用 cuBLAS/cuBLASLt 等 CUDA GEMM/GEMV，按性能数据评估自研，不把每个 GEMM 从零实现作为前置条件。
- Greedy、temperature、top-k、top-p sampling。
- 量化 pack/dequant/scale 及格式对应的 CUDA GEMM。

融合后不要求每个逻辑算子单独 launch。必须区分自研 CUDA、调用 NVIDIA CUDA 库和 PyTorch reference 的覆盖情况。

### 多精度路线

- FP32：正确性参考；TF32：显式的 FP32 GEMM 计算策略，不作为权重存储量化格式。
- FP16/BF16：浮点默认计算和自研 kernel 首版目标。
- W8A16：保留现有 INT8 reference，后续接 fused dequant GEMM。
- W4A16：先实现 group-wise packed INT4 reference，再接 CUDA 性能后端。
- FP8 E4M3/E5M2：先实现 scaled reference，明确 scale、饱和和累加；再接硬件 FP8 GEMM。
- MXFP8：单独设计微缩放格式，按硬件和 Windows 后端能力推进。
- NVFP4：重点规划 NVIDIA FP4 E2M1 和块缩放及额外缩放策略；INT4 或存在 float4 dtype 不等于完整 NVFP4 支持。
- FP8/INT8 KV Cache：需要一起适配 scale、写入和 paged attention 读取。

分别记录权重存储、激活、累加、KV 格式及实际计算后端。每种格式分为接口预留、数值 reference、CUDA kernel、模型集成和性能验收五种状态。RTX 5080 的 SM 12.0 本身不保证所有库和 shape 可用，NVFP4/MXFP8 必须实际验证。当前新增格式尚未实现，不宣称可加速。

### 后续阶段

1. 阶段 06：精度接口和能力矩阵，packed INT4/FP8 reference，native Linear 可选集成；NVFP4/MXFP8 先明确接口及能力边界。
2. 阶段 07：所有 Norm/Residual/SwiGLU CUDA。
3. 阶段 08：Embedding/RoPE/KV 写入 CUDA，复用 GPU block table。
4. 阶段 09：Windows Prefill Flash 风格 CUDA。
5. 阶段 10：Linear/LM head CUDA 库后端和 sampling CUDA，逐精度验收。
6. 阶段 11：完整 Qwen3-0.6B 权重生成验收、profiling、TTFT/TPOT 和实际输出计数；性能检查也贯穿前面阶段。
7. 阶段 12：调度器接通 native model execution、continuous batching、HTTP/SSE 和取消/异常回收。
8. 阶段 13：FP8/INT4 CUDA 深化、NVFP4/MXFP8 和低精度 KV。
9. 阶段 14：Transformers、具体 MiniVLLM 和 vLLM 的正式对比。若 vLLM 使用其他 OS，明确记录环境差异；Windows 引擎不引入 vLLM 运行依赖。

每阶段报告解读文件职责、调用链、数据布局、测试与限制，维护 README 索引。阶段 05 的 Event 计时可能包含 CPU 调度间隙；历史 speedup 只是相对带 Python 循环和 gather 的参考路径，不等于纯 kernel 或相对 vLLM 的加速。

### Git 工作流

- 当前 main 仓库已初始化，尚未提交，未配置 remote。
- 阶段 00–05 保存为一个历史基线，不捏造早期的分阶段提交历史。
- 从阶段 06 起：实现、测试、阶段报告、计划/README、diff 检查、stage、每阶段本地 commit。
- 不提交虚拟环境、模型、缓存、编译产物或大体积 trace。
- 使用已有提交身份；默认只做本地 commit，push 和远端创建按用户另外授权执行。
- 测试失败不标记完成；权限阻塞时保留改动并记录未执行的动作。