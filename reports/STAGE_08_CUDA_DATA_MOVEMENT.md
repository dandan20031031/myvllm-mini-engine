# 阶段 08：Windows CUDA Embedding、RoPE 与 Paged KV 写入

## 完成内容

本阶段把 native Qwen3 forward 中三类数据操作接入 Windows CUDA：

- token embedding gather；
- Q/K rotary position embedding；
- 根据 block table 跨物理页写入 KV Cache。

PyTorch backend 仍保留，CUDA 通过 `data_backend="cuda"` 开启。完整 Qwen3-0.6B 权重质量和端到端 benchmark 尚未完成。

## 文件职责

| 文件 | 职责 |
|---|---|
| `mini_engine/kernels/data.py` | data 算子的 torch/cuda dispatch |
| `mini_engine/kernels/csrc/data.cu` | CUDA gather、RoPE、paged scatter |
| `mini_engine/kernels/csrc/bindings.cpp` | shape、dtype、device、page 校验和导出 |
| `mini_engine/cache/kv_cache.py` | `write_cuda` 入口，复用物理 block table |
| `mini_engine/model/native_qwen3.py` | native forward 的 data backend 调用点 |
| `mini_engine/model/native_runner.py` | 加载模型前准备 data extension |
| `mini_engine/__main__.py` | `--data-backend torch/cuda` |
| `tests/test_data_kernels.py` | direct、stream、边界和 native 对齐 |

## 调用链

```text
input_ids
  -> embedding_lookup(data backend)
  -> Q/K/V projection
  -> Q/K head norm
  -> apply_rope(data backend)
  -> write_kv_cache(data backend)
  -> SDPA prefill / Paged decode
```

KV 的物理地址为：

```text
logical = token_offset + token
block_index = logical // block_size
in_block = logical % block_size
physical = block_ids[block_index]
```

kernel 直接写入 `[physical, in_block, kv_head, head_dim]`，不生成连续 gather tensor。

## 数值处理

RoPE CUDA 路径保留 FP16/BF16 每个乘法的中间舍入，避免 BF16 与 PyTorch reference 的误差放大。GQA 下 Q 对所有 query heads 旋转，K 只对唯一 KV heads 写入。

## Windows 使用

```powershell
. .\scripts\enter_project.ps1
$env:MINI_ENGINE_TEST_CUDA = "1"
.\.venv\Scripts\python -m mini_engine `
  --model "D:\models\Qwen3-0.6B" `
  --prompt "介绍一下 KV Cache" `
  --dtype bf16 --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda `
  --local-files-only
```

## 验证结果

阶段专项测试：

```text
10 passed
```

全量回归：

```text
136 passed, 2 warnings
```

覆盖 embedding 的 FP32/FP16/BF16 和 int32 ids、GQA RoPE、跨页 KV 写入、非默认 stream、非法输入，以及 native Qwen3 的 CUDA data backend 与 torch data backend 的 prefill/decode 对齐。

## 限制

- embedding 仍是 correctness-first gather，未做 persistent/vectorized vocabulary gather；
- RoPE 尚未和 Q/K projection 融合；
- KV 写入尚未和 RoPE 融合；
- KV Cache 仍为 FP16/BF16；
- native runner 仍是单请求，scheduler 尚未驱动真实批量执行；
- prefix/radix cache 和 CUDA Graph decode 尚未接入。

## 下一阶段

按调整后的系统路线，阶段 09 优先将 Scheduler 接到 native model execution，实现真实 continuous batching、批量 block table、取消、EOS 和异常回收；之后做 Prefix/Block Cache 和 CUDA Graph shape bucket。
