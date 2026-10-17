# 阶段 02 报告：Qwen3 参考增量生成路径

## 1. 阶段目标

本阶段首次建立一条真实的“prefill + decode”生成链路，但仍使用 Transformers 作为模型执行后端。目标是先验证：

- prompt 能正确转换为 Qwen3 chat template；
- prefill 可以返回 `past_key_values`；
- 后续每一步只输入一个 token；
- 采样结果能继续作为下一步输入；
- EOS 和最大生成长度能正确结束请求；
- TTFT、TPOT 和总延迟可以被记录。

这条路径是 native Qwen3 forward 和 CUDA Paged Attention 的 correctness baseline。

## 2. 修改文件

```text
mini_engine/model/qwen3.py
mini_engine/model/__init__.py
mini_engine/engine/llm_engine.py
tests/test_incremental_api.py
README.md
reports/STAGE_02_INCREMENTAL_REFERENCE.md
```

## 3. 结果对象

新增 `IncrementalGenerationResult`：

```python
IncrementalGenerationResult(
    text="...",
    token_ids=[...],
    finish_reason="stop" | "length",
    ttft_ms=...,
    tpot_ms=...,
    total_latency_ms=...,
)
```

字段含义：

- `text`：生成 token 解码后的文本；
- `token_ids`：实际生成的 token id；
- `finish_reason`：EOS 结束或达到长度上限；
- `ttft_ms`：从 prefill 开始到第一个生成 token 的时间；
- `tpot_ms`：首 token 之后，每个 decode token 的平均时间；
- `total_latency_ms`：整个 prefill + decode 循环耗时。

## 4. Prompt 构造逻辑

`Qwen3ModelRunner._prompt_tensor` 统一负责 prompt 编码：

```text
prompt
  -> tokenizer.apply_chat_template
  -> add_generation_prompt=True
  -> enable_thinking=config
  -> input_ids [1, prompt_len]
  -> model.device
```

如果 tokenizer 不支持 Qwen3 的 chat template 参数，代码会 fallback 到普通 tokenizer 编码。这让 reference runner 更容易用于小型 mock 模型和其他兼容模型。

## 5. 增量生成调用链

入口：

```python
engine.generate_incremental(prompt, max_new_tokens=64)
```

完整调用链：

```text
LLMEngine.generate_incremental
        |
Qwen3ModelRunner.generate_incremental
        |
_prompt_tensor
        |
model(input_ids=full_prompt, use_cache=True)
        |
prefill logits + past_key_values
        |
sample_next_token
        |
model(input_ids=[one token], past_key_values=past)
        |
更新 past_key_values 和 next_logits
        |
直到 EOS 或 max_new_tokens
```

### 5.1 Prefill

prefill 一次输入完整 prompt：

```python
outputs = self.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    use_cache=True,
    return_dict=True,
)
```

使用最后一个 prompt token 的 logits 生成第一个输出 token，同时保存 `outputs.past_key_values`。

### 5.2 Decode

每一轮 decode 只输入上一步生成的 token：

```python
outputs = self.model(
    input_ids=input_ids[:, -1:],
    attention_mask=attention_mask,
    past_key_values=past,
    use_cache=True,
    return_dict=True,
)
```

在当前实现中，`input_ids` 是 `[1, 1]`，`attention_mask` 逐步追加一个有效位置，`past_key_values` 由 Transformers 负责管理。

这正是后续 native KV Cache 需要复现的语义：历史 K/V 只写入一次，decode 只追加当前 token 的 K/V。

## 6. 采样逻辑

每一步都会调用：

```python
token = sample_next_token(
    next_logits,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
)
```

流程为：

```text
logits
  -> temperature scaling
  -> top-k filtering
  -> top-p filtering
  -> greedy argmax 或 multinomial
```

当 `temperature <= 0` 时使用 greedy，这对于和 vLLM 做确定性 correctness 对比最重要。

## 7. 结束条件

结束条件有两个：

1. 生成 token 命中 tokenizer 的 EOS id；
2. 生成数量达到 `max_new_tokens`。

返回的 `finish_reason` 分别为：

```text
stop
length
```

如果 tokenizer 没有 EOS id，代码会把 EOS 集合视为空集合，此时由长度上限结束。

## 8. 延迟统计

时间点定义：

```text
started          = prefill 开始
first_token_time = 第一个 token 采样完成
finished         = 最后一轮 decode 完成
```

计算：

```text
TTFT = first_token_time - started
TPOT = (finished - first_token_time) / max(generated_tokens - 1, 1)
total = finished - started
```

CUDA 端到端 benchmark 仍需要在外层使用 `torch.cuda.synchronize()`，因为 CUDA kernel 默认异步执行。当前 `benchmark_callable` 已经具备这套同步逻辑。

## 9. LLMEngine 接口

新增：

```python
engine.generate_incremental(prompt, max_new_tokens=64)
```

该方法只要求传入一个实现了 `generate_incremental` 的 runner，因此之后可以替换为：

```text
TransformersQwen3Runner
NativeQwen3Runner
CudaQwen3Runner
```

上层调用和 benchmark adapter 不需要改变。

## 10. 测试

新增 `tests/test_incremental_api.py`，覆盖：

- `LLMEngine` 能把请求委托给 incremental-capable runner；
- 没有 runner 时返回明确错误；
- 返回对象包含文本、token ids 和延迟字段。

当前全量测试结果：

```text
8 passed
```

由于真实 Qwen3 权重下载和 GPU 推理不是单元测试的稳定依赖，本阶段没有在测试中下载模型；真实模型验证应使用 README 中的 reference runner 命令。

## 11. 当前限制

这条路径仍然不是 native engine，限制包括：

- K/V 存储由 Transformers 的 `past_key_values` 管理；
- 没有写入项目自己的 `KVCachePool`；
- 单请求 batch size 为 1；
- decode 仍由 Transformers model forward 执行；
- 没有 continuous batching 的真实模型执行；
- 没有 CUDA Paged Attention。

因此当前结果适合做正确性和 API 基线，不适合宣称性能超过 vLLM。

## 12. 下一阶段

下一阶段将把 Qwen3 的模型配置和 K/V 形状接到项目自己的 cache：

1. 读取 `num_hidden_layers`、`num_key_value_heads`、`head_dim`；
2. 初始化 `KVCachePool`；
3. 为 native layer forward 设计 `[batch, token, hidden]` 输入接口；
4. 在每层把新 K/V 写入物理 block；
5. 用 block table 读取历史 K/V；
6. 先实现 PyTorch native attention，再替换成 CUDA Paged Attention。
