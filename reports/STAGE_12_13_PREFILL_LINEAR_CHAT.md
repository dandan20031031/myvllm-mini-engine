# 阶段 12+13：Windows CUDA Prefill、显式 CUDA Linear 与本地 Chat

## 1. 完成内容

本阶段合并完成：

- Windows CUDA causal prefill attention correctness backend；
- 显式 `CudaLinear`，使用 PyTorch CUDA matmul 调度 cuBLAS/cuBLASLt；
- native Qwen3 各 projection/MLP/lm_head 的 `linear_backend` 选择；
- CLI `--chat` 多轮交互模式；
- CLI `--prefill-backend cuda`、`--linear-backend cuda`。

## 2. CUDA Prefill

文件：`mini_engine/kernels/csrc/prefill.cu`。

接口：

```python
prefill_attention(q, k, v, backend="cuda")
```

输入为：

```text
q [batch, query_heads, tokens, head_dim]
k/v [batch, kv_heads, tokens, head_dim]
```

kernel 使用 causal mask，每个 CTA 负责一个 `(batch, query_head, query_position)`，GQA 通过 query head 到 KV head 的映射处理。当前是 correctness-first 的在线 softmax 实现，限制 token 长度不超过 8192；长 prompt 使用 SDPA `auto/math`。

这不是当前 Windows PyTorch wheel 中缺失的 FlashAttention backend，也不宣称已经达到 FlashAttention/TensorRT-LLM 的性能；阶段目标是建立可替换的 CUDA prefill 接口和数值证据。

## 3. Explicit CUDA Linear

文件：`mini_engine/kernels/linear.py`。

`CudaLinear` 将输入展平成二维矩阵，执行：

```python
torch.mm(flat, weight.t())
```

在 CUDA 上由 PyTorch 选择底层 cuBLAS/cuBLASLt 路径；它不是本项目自研 GEMM。这样能把 Linear 后端显式纳入模型配置，后续可替换为 CUTLASS、DeepGEMM 或专用量化 GEMM。

native Qwen3 可配置：

```python
NativeQwen3(..., linear_backend="cuda")
```

覆盖 Q/K/V/O、Gate/Up/Down 和 lm_head；embedding、norm、residual 仍由各自 backend 控制。

## 4. Windows 本地交互聊天

当前已具备单进程、本地模型、命令行多轮 chat：

```powershell
. .\scripts\enter_project.ps1
$env:MINI_ENGINE_TEST_CUDA = "1"
.\.venv\Scripts\python -m mini_engine `
  --model "D:\models\Qwen3-0.6B" `
  --chat `
  --device cuda `
  --dtype bf16 `
  --prefill-backend auto `
  --decode-backend cuda `
  --elementwise-backend cuda `
  --data-backend cuda `
  --linear-backend cuda `
  --max-new-tokens 128 `
  --local-files-only
```

交互命令：

```text
/clear  清空对话历史
/exit   退出
```

对话历史通过 Qwen3 chat template 重新编码，每轮使用当前 native runner 生成。当前这是本地 CLI 交互，不是 HTTP/OpenAI 服务。

## 5. 当前能否本地对话

可以，但需要满足：

1. 本机有完整 Qwen3-0.6B 权重和 tokenizer 目录；
2. 已安装项目 `.venv` 的 Transformers 依赖；
3. 使用 CUDA 时有 Windows CUDA PyTorch、CUDA Toolkit、MSVC 和 Ninja；
4. 模型配置属于当前支持的 dense Qwen3：无 scaled RoPE、无 sliding-window 变体；
5. 先用 `--prefill-backend auto` 验证，再切换 `cuda` prefill。

当前代码仓库没有自动下载或内置完整模型权重，因此不能仅靠 clone 后立即生成文本。完整 0.6B 权重下的真实文本输出、质量和端到端 TPOT 仍需在本机模型目录上验收。

## 6. 当前短板

- 没有 HTTP/OpenAI/SSE 服务；
- continuous batching 已有 batch decode runtime，但 prefill 仍逐请求；
- Prefix Cache 共享物理 block 已完成，尚未跳过 prompt 计算；
- CUDA Graph bucket 基础已完成，尚未接入完整 runtime replay；
- CUDA prefill 是 correctness-first，不等于 FlashAttention 性能；
- CudaLinear 调用 cuBLAS/cuBLASLt，量化 INT4/FP8 仍是 dequant reference；
- FP8/INT8 KV Cache、NVFP4/MXFP8 尚未实现；
- 完整 Qwen3-0.6B benchmark 和行业项目对比尚未完成；
- 投机解码已进入开发计划阶段 17，尚未实现。

## 7. 测试结果

新增 prefill/linear/chat API 测试 4 项，全量结果：

```text
146 passed, 2 warnings
```

warning 仍是 CUDA Toolkit 13.3 与 PyTorch CUDA 12.8 差异及 setuptools MSVC 环境提示。

## 8. 下一步

先把本地 CLI 交互路径用完整 Qwen3-0.6B 权重实际验收，再实现 HTTP/SSE。之后接入 CUDA Graph bucket 到 batch runtime，最后进入 FP8/INT4 CUDA GEMM 和投机解码阶段。
