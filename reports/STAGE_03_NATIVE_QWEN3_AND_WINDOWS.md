# 阶段 03 报告：原生 Qwen3 Forward、Paged Decode 与 Windows 适配

## 1. 阶段目标

本阶段把项目从 Transformers `past_key_values` 参考循环推进到项目自己的模型 forward：

- 用 HuggingFace 只负责读取配置和权重；
- 自己实现 Qwen3 的 RMSNorm、Q/K/V projection、Q/K head norm、RoPE、GQA、SwiGLU 和输出层；
- prefill 使用 PyTorch SDPA，支持 `auto`、`math`，并为 `flash` 保留严格入口；
- decode 使用项目自己的 `KVCachePool` 和 paged attention fallback；
- 建立单请求 native runner；
- 提供 Windows 11 可执行的命令行推理入口；
- 完成随机小模型与 Transformers 的数值对齐测试。

## 2. 新增和修改文件

```text
mini_engine/model/native_qwen3.py
mini_engine/model/native_runner.py
mini_engine/kernels/prefill.py
mini_engine/__main__.py
tests/test_native_qwen3.py
.gitignore
```

修改：

```text
mini_engine/kernels/fallback.py
mini_engine/cache/kv_cache.py
mini_engine/model/qwen3.py
mini_engine/model/__init__.py
mini_engine/engine/llm_engine.py
pyproject.toml
README.md
DEVELOPMENT_PLAN.md
```

## 3. Windows 11 约束

本阶段没有引入 Linux 专属依赖：

- 不依赖 Triton；
- 不依赖 `flash-attn` 第三方包；
- 不要求 WSL；
- CUDA extension 后续使用 Windows 的 MSVC + `nvcc`；
- prefill 使用 PyTorch 自带 SDPA；
- 命令行使用 PowerShell 可直接运行的 `python -m mini_engine`。

当前机器检测到 RTX 5080、CUDA 可用和 BF16 可用。当前 PyTorch Windows wheel 报告 `torch.backends.cuda.is_flash_attention_available() == False`，因此：

```text
prefill_backend=auto -> 让 PyTorch 选择可用 backend
prefill_backend=math -> 强制数学参考实现
prefill_backend=flash -> 无 Flash backend 时明确报错
```

代码不会把 math fallback 冒充成 Flash。

## 4. Native Qwen3 结构

文件：`mini_engine/model/native_qwen3.py`

### 4.1 配置转换

`Qwen3Config.from_hf` 从 HuggingFace config 读取：

- vocab size；
- hidden size；
- intermediate size；
- layer 数；
- attention heads；
- KV heads；
- head dimension；
- RoPE theta；
- RMSNorm epsilon；
- 权重是否 tied。

当前明确拒绝未实现的模型变体：

- 非 Qwen3 模型；
- 非 SiLU activation；
- scaled RoPE；
- sliding-window attention。

这样可以避免 silently 产生错误输出。

### 4.2 Attention

每层 attention 的调用顺序：

```text
hidden
  -> q_proj / k_proj / v_proj
  -> Q/K head RMSNorm
  -> RoPE
  -> 写入当前 token 的 K/V
  -> prefill SDPA 或 decode paged attention
  -> o_proj
```

Qwen3 的 GQA 通过把 KV head repeat 到 query head 数量来实现。后续 CUDA kernel 可以直接在 shared memory 或寄存器中完成 GQA 复用，避免显式 repeat。

### 4.3 MLP

MLP 使用：

```text
down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

当前 `silu_and_mul` 是 PyTorch fallback，后续可替换成 fused CUDA kernel。

### 4.4 Native forward

```python
logits = native(input_ids, cache, block_ids, position=0)
```

规则：

- `position=0`：输入完整 prompt，执行 prefill；
- `position>0`：只允许输入一个 token，执行 decode；
- caller 必须提前分配足够的物理 blocks；
- 返回所有输入位置的 logits；runner 截取最后一个位置，供 sampler 生成下一个 token。

## 5. Prefill 和 Decode 的分工

### 5.1 Prefill

prefill 使用：

```python
torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)
```

并通过 `mini_engine/kernels/prefill.py` 选择 backend。当前为正确性优先的 Qwen3 GQA 路径，会显式扩展 K/V head；最终高性能版本应融合 GQA 和 attention，减少中间 tensor。

### 5.2 Decode

decode 先写入当前 K/V：

```text
KVCachePool.write(layer, block_ids, position, k, v)
```

随后通过：

```text
block table + context length + physical cache
        -> paged_attention fallback
```

当前 decode 是 PyTorch reference kernel。下一步是把它替换成 Windows CUDA extension，接口保持不变。

## 6. Native runner 调用链

入口：

```python
NativeQwen3Runner.from_pretrained(...)
```

加载过程：

```text
AutoTokenizer.from_pretrained
AutoModelForCausalLM.from_pretrained
        -> NativeQwen3.from_hf_model
        -> Qwen3Config.from_hf
        -> NativeQwen3.create_cache
```

生成过程：

```text
prompt
  -> apply_chat_template
  -> BlockManager.allocate
  -> NativeQwen3.forward(prefill)
  -> sample_next_token
  -> BlockManager.grow
  -> NativeQwen3.forward(decode, one token)
  -> EOS / max_new_tokens
  -> release blocks
```

`try/finally` 保证异常时也会释放 block，并清理 `_busy` 状态，避免一次错误请求永久占用 KV Cache。

## 7. 命令行部署入口

现在可以用单进程、单请求方式运行：

```powershell
pip install -e ".[model]"
python -m mini_engine `
  --model "D:\models\Qwen3-0.6B" `
  --prompt "介绍一下 KV Cache" `
  --device cuda `
  --dtype bf16 `
  --prefill-backend auto `
  --max-new-tokens 128 `
  --local-files-only
```

输出为 JSON，包含：

```text
text
token_ids
finish_reason
ttft_ms
tpot_ms
total_latency_ms
```

这提供了本地开发推理入口。当前还不是 HTTP 服务，也不支持多请求并发。这里的权重路径是示例，项目没有自动下载完整 Qwen3-0.6B 权重。

## 8. 测试

新增测试包含：

- 非连续 block + GQA 的 paged attention 对齐；
- CPU 随机 Qwen3 native forward 与 Transformers eager forward 对齐；
- CUDA BF16/FP16 对齐（有 CUDA 时运行）；
- native runner forward 次数和生成长度；
- EOS 提前停止；
- 异常后 block 释放；
- KV 容量不足和非法 block id；
- Windows SDPA auto/math/flash 能力分支。

当前基础 Python 环境结果：

```text
15 passed, 3 skipped
```

独立 `.venv` 安装 Transformers 4.57.6 后的结果为：

```text
18 passed in 22.36s
```

其中确实运行了 CPU FP32、RTX 5080 BF16 和 FP16 的随机小模型对齐测试：完整 prompt 加六次 decode，与 Transformers 对完整前缀重新计算的 logits 比较，覆盖非连续 block 和跨 block 边界。

**本阶段尚未加载完整 Qwen3-0.6B 权重验证真实文本生成，也未实际启动服务。** 小模型测试验证实现逻辑，不能替代完整权重下的模型验收和性能测试。

## 8.1 修复的已有问题

- 修正 paged attention 的 K/V 维度排列：原实现转置后仍按 token/head 顺序做 einsum，导致维度含义不一致。
- RMSNorm 归一化乘法先使用 FP32，再转回输入 dtype，贴近 Qwen3 参考语义。
- 参考增量循环到达最后一个输出 token 后立即结束，避免一次无用 forward；单输出 token 的 TPOT 为零。
- 优先读取 generation config 的 EOS，并保留 tokenizer EOS fallback。
- 模板错误直接暴露，避免静默切换到普通文本编码破坏参考对比。
- 拒绝负数物理 block id，避免 Python 负索引读取错误的 cache 页面。
- `max_new_tokens=0` 不再被引擎的默认值覆盖。

## 8.2 本阶段运行边界

Native runner 自己持有 BlockManager 和 KVCachePool，尚未由 LLMEngine.scheduler 驱动；两条接口不能视为已经接通的 continuous batching。`LLMEngine.quant_backend` 也尚未替换 native Linear，当前 native 模型只有浮点权重路径。计时从预编码 prompt 的模型 prefill 开始，到最后一个 token 完成采样为止，不包含 tokenizer 和文本解码，不等同于 HTTP 服务 TTFT。

## 9. 当前部署能力判断

### 已达到

- 单请求本地 Qwen3 native 推理入口；
- 自有 KV Cache pool；
- prefill/decode 分工；
- Windows 11 + CUDA 路径；
- 异常资源回收；
- 可输出 TTFT/TPOT。

### 尚未达到

- HTTP/OpenAI 兼容 API；
- continuous batching 的真实 model execution；
- CUDA Paged Attention；
- CUDA fused RMSNorm/RoPE/SwiGLU；
- W8A16/W4A16 native GEMM；
- 与 vLLM 公平端到端 benchmark。

## 10. 下一阶段

下一阶段优先实现 Windows CUDA Paged Attention extension：

1. 用 `torch.utils.cpp_extension` 建立 MSVC + nvcc 编译入口；
2. 保持 `paged_attention(query, key_cache, value_cache, block_tables, context_lens, block_size)` API；
3. 先实现 decode 单 token、GQA、BF16/FP16；
4. 与 PyTorch fallback 做逐元素和 logits 对齐；
5. 再接 CUDA Graph 和 benchmark；
6. 最后添加 FastAPI/SSE 服务层。
