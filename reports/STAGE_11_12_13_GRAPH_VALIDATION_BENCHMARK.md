# 阶段 11/12/13 补充：Graph Runtime 接入、真实模型验收与端到端 Benchmark

## 1. 本阶段完成内容

- CUDA Graph bucket 接入 `NativeBatchRuntime.decode_step`；
- graph replay 使用固定 input/position/block table buffer；
- graph padding slot 必须显式提供 scratch KV pages，避免覆盖活跃请求；
- 新增完整 Qwen3 权重验收脚本；
- 新增 native 单请求端到端 benchmark；
- 增加脚本路径和 JSON 输出测试。

## 2. Graph 接入调用链

```text
NativeBatchRuntime.decode_step
  -> 组装 active token、positions、block_tables
  -> DecodeGraphManager.get(batch, max_blocks)
  -> bucket.capture()（首次）
  -> bucket.replay()
  -> batch logits
  -> sampler / Scheduler.on_token
```

Graph bucket 不负责 CPU 请求调度、Prefix Cache 或 block 所有权。它只负责固定形状和固定地址的模型 decode。padding slot 必须使用外部预留的 scratch block；没有 scratch pages 时 runtime 会明确报错，不会把活跃请求的 page 当 padding 使用。

## 3. 完整模型验收

文件：`benchmarks/validate_qwen3.py`

脚本会：

1. 检查本地模型目录；
2. 加载 Transformers reference；
3. 生成固定 prompts；
4. 加载 native runner；
5. 输出 reference/native 文本、token ids、TTFT、TPOT 和 finish reason；
6. 写入项目内 JSON。

当前机器没有本地 Qwen3-0.6B 权重，因此本次实际结果是：

```json
{
  "status": "blocked",
  "reason": "local model directory is missing; provide a downloaded Qwen3-0.6B directory"
}
```

输出文件：`benchmarks/results/qwen3_validation.json`。脚本不伪造真实模型输出或性能数字。

运行方式：

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\validate_qwen3.py `
  --model "D:\models\Qwen3-0.6B" `
  --device cuda --dtype bf16 `
  --prefill-backend auto `
  --decode-backend cuda `
  --elementwise-backend cuda `
  --data-backend cuda `
  --linear-backend cuda `
  --local-files-only
```

## 4. 端到端 Benchmark

文件：`benchmarks/benchmark_native_qwen3.py`

支持：

- warmup/repetitions；
- 多 prompt；
- TTFT、TPOT、总延迟；
- 实际输出 token 数；
- GPU、PyTorch、CUDA runtime、backend 配置记录；
- JSON 结果输出。

运行方式：

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\benchmark_native_qwen3.py `
  --model "D:\models\Qwen3-0.6B" `
  --device cuda --dtype bf16 `
  --prefill-backend auto --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda `
  --linear-backend cuda --local-files-only
```

当前脚本是 native 单请求基线，不是 vLLM 对比；完整权重缺失时不会运行。

## 5. 测试结果和真实状态

全量测试：

```text
147 passed, 2 warnings
```

新增测试覆盖：

- Graph runtime 修改后的固定 buffer/padding 行为；
- validation/benchmark 脚本存在且不硬编码 C 盘；
- 既有 CUDA、data、elementwise、batch runtime、prefix cache、量化回归。

当前结论：

- Graph 已接入代码路径，但完整 native batch runtime 的 Graph replay 需要预留 scratch KV pages；
- 验收和 benchmark 框架已经可运行；
- 完整 Qwen3-0.6B 文本质量、TTFT/TPOT 和显存数字尚未取得，因为本地没有权重；
- 不能据此宣称项目已经完成部署或超过 vLLM。

## 6. 当前剩余短板

- 需要实际下载/提供 Qwen3-0.6B 权重后完成真实验收；
- Graph bucket 还需做完整模型 capture/replay benchmark；
- prefill CUDA 是 correctness-first，不等于 FlashAttention 性能；
- 量化 Linear 仍是 reference dequant，未实现 INT4/FP8 GEMM；
- Prefix Cache 尚未跳过 prompt 计算；
- HTTP/OpenAI/SSE 尚未实现；
- 投机解码仍在阶段 17 计划中。
