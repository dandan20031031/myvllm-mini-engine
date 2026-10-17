# 阶段 00 报告：项目骨架与推理调度基础

## 1. 阶段目标

这是项目的第一阶段，目标不是立即实现完整的 vLLM，而是先建立一套可以逐步替换和扩展的推理引擎骨架：

- 固定 Qwen3-0.6B 作为第一阶段模型目标；
- 把模型、请求、调度、缓存、算子和量化后端分层；
- 实现请求生命周期和基础 continuous batching 调度；
- 实现 Paged KV Cache 的 block 元数据管理；
- 提供 PyTorch fallback 算子作为正确性参考；
- 预留 Qwen3 Transformers runner 和 CUDA kernel 接口；
- 为后续 W8A16、W4A16 和 FP8 扩展保留量化抽象。

本阶段的核心原则是：先让每个模块可以单独测试，再逐步把模块连接成真实的 Qwen3 增量推理链路。

## 2. 初始目录

```text
mini_engine/
  config.py
  sampling.py
  engine/
    request.py
    scheduler.py
    llm_engine.py
  cache/
    block_manager.py
  kernels/
    fallback.py
  model/
    qwen3.py
  quantization/
    backends.py
tests/
  test_core.py
```

同时创建了：

```text
pyproject.toml
README.md
DEVELOPMENT_PLAN.md
```

## 3. 配置层

文件：`mini_engine/config.py`

### 3.1 `QuantConfig`

`QuantConfig` 统一描述量化相关参数：

```python
QuantConfig(
    weight_dtype="bf16",
    activation_dtype="bf16",
    group_size=128,
    symmetric=True,
)
```

字段含义：

- `weight_dtype`：权重精度，当前支持浮点和参考 INT8 路径；
- `activation_dtype`：激活精度；
- `group_size`：为后续 group-wise INT4 预留；
- `symmetric`：为对称/非对称量化预留。

模型代码只应依赖这个配置，而不应该直接判断某个具体量化库。

### 3.2 `EngineConfig`

`EngineConfig` 保存引擎级参数：

- 模型路径；
- device 和 dtype；
- block size；
- GPU block 数量；
- 单轮最大 token budget；
- 默认最大生成长度；
- 量化配置；
- thinking 模式开关；
- 是否启用自定义 kernel。

它是后续从命令行、配置文件或服务 API 创建引擎的统一入口。

## 4. 请求生命周期

文件：`mini_engine/engine/request.py`

### 4.1 状态机

请求状态为：

```text
WAITING -> RUNNING -> FINISHED
             |
             -> ABORTED
```

- `WAITING`：已经进入队列但还没有分配 KV block；
- `RUNNING`：已经被调度并持有 block；
- `FINISHED`：达到 stop 条件或最大生成长度；
- `ABORTED`：被用户取消或主动中止。

### 4.2 `GenerationRequest`

请求对象保存：

- `request_id`；
- prompt token ids；
- 最大生成长度；
- temperature、top-k、top-p；
- 输出 token ids；
- 已生成 token 数；
- 当前 block ids；
- finish reason。

几个重要属性：

```python
prompt_len = len(prompt_token_ids)
total_len = prompt_len + generated_tokens
is_finished = state in (FINISHED, ABORTED)
```

`append_token` 负责更新输出和生成长度，`finish` 负责统一设置完成状态和原因。

## 5. BlockManager 调用逻辑

文件：`mini_engine/cache/block_manager.py`

### 5.1 设计目的

`BlockManager` 不保存 K/V tensor，只管理物理 block 的所有权。这样调度逻辑和 GPU 存储可以独立演进。

### 5.2 分配流程

```text
Scheduler.schedule
        |
BlockManager.can_allocate(prompt_len)
        |
BlockManager.allocate(request_id, prompt_len)
        |
request.block_ids = [physical block ids]
```

`blocks_required(token_count)` 使用向上取整：

```python
(token_count + block_size - 1) // block_size
```

例如 `block_size=16` 时：

```text
1 token  -> 1 block
16 token -> 1 block
17 token -> 2 blocks
```

### 5.3 Decode 阶段扩容

每生成 token 后，调度器调用：

```python
block_manager.grow(request_id, request.total_len)
```

当逻辑 token 数超过当前 block 容量时，allocator 从 free list 中追加物理 block。请求结束后调用 `release`，把所有 block 放回 free list。

### 5.4 `BlockTable`

`BlockTable` 保存：

```text
request_id
block_ids
```

后续 Paged Attention kernel 会使用 block ids，把逻辑 token 位置映射到非连续的物理 KV block。

## 6. Scheduler 调用逻辑

文件：`mini_engine/engine/scheduler.py`

### 6.1 队列

调度器维护：

```text
waiting: 等待 admission 的请求
running: 已经分配 KV block 的请求
```

请求完成后从 `running` 移除，并释放 block。

### 6.2 `schedule()`

调用链：

```text
LLMEngine.add_request
        -> Scheduler.add_request
        -> Scheduler.schedule
        -> 返回 SchedulerOutput(prefill, decode)
```

调度器会按 waiting 队列顺序尝试 admission，同时检查：

- prompt 是否超过当前 token budget；
- 是否有足够的物理 block；
- 请求是否已经 aborted。

新 admission 的请求进入 `prefill`，此前已经运行的请求进入 `decode`。这样上层 model runner 可以使用同一轮调度结果分别处理 prompt 计算和单 token 解码。

### 6.3 `on_token()`

模型 runner 生成一个 token 后调用：

```python
scheduler.on_token(request, token_id, finished=False)
```

这个调用会：

1. 把 token 写入请求输出；
2. 增加生成计数；
3. 根据新的总长度扩容 block table；
4. 达到 stop 或长度限制时完成请求；
5. 释放请求拥有的 block。

## 7. PyTorch fallback 算子

文件：`mini_engine/kernels/fallback.py`

### 7.1 RMSNorm

`rms_norm` 对最后一维求均方根倒数，再乘以可学习权重：

```text
variance = mean(x²)
output = x * rsqrt(variance + eps) * weight
```

统计部分使用 float 计算，输出转换回输入 dtype，以减少低精度累积误差。

### 7.2 RoPE

`apply_rope` 将 head dimension 切成两半，使用预先计算的 `cos` 和 `sin` 对 Q、K 做旋转位置编码。函数同时返回旋转后的 query 和 key，方便后续融合成一个 CUDA kernel。

### 7.3 SwiGLU 激活

`silu_and_mul` 实现：

```text
SiLU(gate) * up
```

后续可与 projection 或 residual 融合，减少 global memory round-trip。

### 7.4 Paged Attention reference

`paged_attention` 是正确性优先的参考实现：

- 输入 query 形状为 `[batch, query_heads, head_dim]`；
- K/V cache 按物理 block 保存；
- block table 描述每个请求的物理 block；
- context lens 描述每个请求有效 token 数；
- 支持 GQA/MQA 的 query head 到 KV head 映射。

它会先按 block table 读取历史 K/V，再执行 attention。这个实现用于验证输出，不代表最终性能；真正的 CUDA kernel 将在 kernel 内直接读取分页 cache。

## 8. 采样器

文件：`mini_engine/sampling.py`

`sample_next_token` 的调用顺序是：

```text
logits
  -> temperature
  -> top-k filtering
  -> top-p filtering
  -> multinomial
```

当 `temperature <= 0` 时走 greedy argmax。这个模块独立于模型和调度器，后续可以替换为 fused CUDA sampling，而不改变请求接口。

## 9. 量化后端

文件：`mini_engine/quantization/backends.py`

### 9.1 统一接口

```python
class QuantBackend:
    def pack(self, weight): ...
    def linear(self, x, packed_weight, bias=None): ...
```

### 9.2 FloatBackend

浮点后端把权重转换到目标 dtype，然后调用 PyTorch `linear`。它是默认基线，也可用于和量化结果做数值对比。

### 9.3 Int8WeightOnlyBackend

这是参考版 W8A16：

1. 对每个输出通道计算绝对值最大值；
2. 用该值除以 127 得到 scale；
3. 权重除以 scale 后四舍五入为 INT8；
4. 线性层执行时反量化到 float，再调用 PyTorch linear；
5. 输出转换回激活 dtype。

当前实现重点是接口和正确性，不是最终性能。后续需要用 fused dequant + GEMM CUDA kernel 替换。

## 10. Qwen3 Transformers runner

文件：`mini_engine/model/qwen3.py`

`Qwen3ModelRunner` 是可选的 HuggingFace 参考适配器，负责：

- 加载 tokenizer；
- 加载 `AutoModelForCausalLM`；
- 设置 dtype 和 device；
- 使用 Qwen3 chat template；
- 控制 `enable_thinking`；
- 提供最后 token logits；
- 提供完整 reference generate。

调用链：

```text
Qwen3ModelRunner.generate
        -> tokenizer.apply_chat_template
        -> model.generate
        -> tokenizer.decode
```

它的定位是正确性 baseline，不是最终的 native model runner。后续自研 forward 可以替换这个 runner，而 `LLMEngine`、请求和调度器接口保持不变。

## 11. LLMEngine 统一入口

文件：`mini_engine/engine/llm_engine.py`

`LLMEngine` 负责组装：

```text
EngineConfig
  -> BlockManager
  -> Scheduler
  -> QuantBackend
  -> optional ModelRunner
```

常用调用路径：

```python
engine = LLMEngine(config)
request = engine.add_request(prompt_token_ids)
scheduled = engine.schedule()
```

如果配置了 reference runner，还可以调用：

```python
engine.generate_reference(prompt)
```

当前 `step_reference` 只验证调度入口，不会假装完成模型 forward。这样可以避免在 native decode 尚未完成时产生误导性的端到端结果。

## 12. 测试与验证

本阶段初始测试覆盖：

- block 分配、扩容和回收；
- prefill/decode 调度；
- 请求完成后的资源释放；
- RMSNorm 数值输出；
- greedy sampling。

阶段完成时测试结果为：

```text
3 passed
```

之后阶段 01 增加了 KV Cache 和 benchmark 测试，目前全项目测试为 6 个。

## 13. 阶段限制

本阶段明确没有实现：

- 真实 GPU KV Cache tensor；
- Qwen3 原生 forward；
- 使用 KV Cache 的增量 decode；
- CUDA extension；
- W4A16/FP8；
- HTTP 服务；
- 和 vLLM 的端到端性能对比。

这些限制是有意保留的边界，避免在基础接口未稳定前过早进入性能优化。

## 14. 与下一阶段的衔接

阶段 01 将 `BlockManager` 后面接入了真实 `KVCachePool`。接下来要把 Qwen3 每层生成的 K/V 写入该 pool，并让单请求 decode 使用 block table 读取历史缓存，形成第一条真实的增量生成链路。
