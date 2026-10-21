# 阶段 04 报告：Windows CUDA Paged Attention

## 1. 阶段目标

本阶段开始正式替换 CUDA kernel，目标是把 decode 的 PyTorch paged attention fallback 替换成项目自己的 Windows CUDA extension：

- 使用 MSVC + nvcc + PyTorch C++ extension；
- 不引入 Linux、WSL 或 Triton 依赖；
- 支持 FP16/BF16；
- 支持 GQA/MQA；
- 支持非连续物理 KV block；
- 支持跨 block 边界和长上下文分区；
- 保持 Python fallback 和 CUDA backend 可切换；
- 所有编译输出放在项目 D 盘目录。

## 2. 新增文件

```text
mini_engine/runtime.py
mini_engine/kernels/cuda_extension.py
mini_engine/kernels/paged.py
mini_engine/kernels/csrc/bindings.cpp
mini_engine/kernels/csrc/paged_attention.cu
tests/test_cuda_paged_attention.py
scripts/enter_project.ps1
scripts/migrate_project_cache.ps1
reports/cache_migration.json
```

修改：

```text
mini_engine/model/native_qwen3.py
mini_engine/model/native_runner.py
mini_engine/model/qwen3.py
mini_engine/__init__.py
pyproject.toml
.gitignore
```

## 3. Windows 编译链

当前机器的工具链：

```text
OS              Windows 11
GPU             NVIDIA GeForce RTX 5080
PyTorch         2.9.1+cu128
CUDA Toolkit    13.3
MSVC            14.51 / Visual Studio Build Tools 18
Ninja           项目 .venv 内安装
```

PyTorch CUDA runtime 12.8 与本机 Toolkit 13.3 不同。编译入口会保留显式 warning，不会修改 PyTorch 版本检查，也不会把两个版本伪装成一致。当前目标机器已实际编译并加载成功，但换机器部署前应重新运行 CUDA 测试。

### 3.1 D 盘路径策略

`mini_engine.runtime.configure_local_cache()` 会把以下位置配置到项目目录：

```text
build/tmp
build/torch_extensions
.cache/cuda
.cache/torch
.cache/huggingface
.cache/pip
.cache/inductor
```

`scripts/enter_project.ps1` 可在 PowerShell 中 dot-source：

```powershell
. .\scripts\enter_project.ps1
```

它设置当前 shell 的缓存路径，不会修改系统全局环境变量。

本阶段通过 `scripts/migrate_project_cache.ps1` 把本项目安装的 Transformers、HuggingFace Hub、Tokenizers、Ninja pip 下载缓存迁到项目目录 `.cache/pip`，每个文件迁移后进行 SHA256 校验。原始迁移记录包含本机路径，仅保留在开发环境，不纳入公开仓库。C 盘共享 NVIDIA ComputeCache 无法可靠归属本项目，且本项目运行期间没有发现近期文件，因此保留原位。

## 4. CUDA Python 接口

文件：`mini_engine/kernels/paged.py`

```python
paged_attention(
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lens,
    block_size,
    backend="torch" | "cuda",
    validate=True,
)
```

调用逻辑：

```text
backend=torch -> fallback.paged_attention
backend=cuda  -> load_cuda_extension().paged_attention
```

使用 `backend="cuda"` 时，Python 层先检查 device 和 dtype，C++ binding 再检查 contiguous、shape、dtype、GQA、block size、context length 和物理 block id。

`validate=False` 只适合 trusted native model path：跳过可能同步 GPU 的 metadata 检查，以减少 launch 前开销。CUDA kernel 仍然对非法物理 page 使用安全的 `valid` 判断，避免坏 page 读取任意地址。

## 5. CUDA kernel 调用逻辑

文件：`mini_engine/kernels/csrc/paged_attention.cu`

### 5.1 Grid 映射

```text
grid.x = batch * query_heads
grid.y = partitions
block  = 128 threads = 4 warps
```

一个 CTA 负责一个请求、一个 query head 和一个上下文分区。

### 5.2 物理页面读取

逻辑 token `t` 映射为：

```text
physical_page = block_tables[request, t / block_size]
offset         = t % block_size
KV address     = [physical_page, offset, kv_head, head_dim]
```

kernel 不创建连续 K/V gather tensor，也不显式 repeat GQA 的 K/V。

### 5.3 Softmax

每个分区内执行：

1. query-key dot product；
2. warp/block max reduction；
3. `exp(score - max)`；
4. warp/block sum reduction；
5. value 加权累加。

上下文长度超过 256 tokens 时，先写入每个 partition 的 `partial output + logsumexp`，再由第二个 kernel 做 log-sum-exp merge。这样可以支持长上下文，同时保持每个 attention CTA 的工作量有限。

### 5.4 支持的 head dimension

当前编译 dispatch 覆盖：

```text
16, 32, 64, 80, 96, 128, 192, 256
```

Qwen3-0.6B 的实际 head dimension 会从 HuggingFace config 读取，不硬编码。

## 6. Native Qwen3 接入

`NativeQwen3` 新增 `decode_backend`：

```python
NativeQwen3(config, prefill_backend="auto", decode_backend="torch")
```

每个 decode attention layer 调用：

```text
Q/K/V projection
  -> current K/V 写入 KVCachePool
  -> backend=torch/cuda 的 paged_attention
  -> output projection
```

`NativeQwen3Runner.from_pretrained(..., decode_backend="cuda")` 会在加载模型前编译/加载 extension，避免把第一次编译时间计入 TTFT。

CLI 示例：

```powershell
. .\scripts\enter_project.ps1
python -m mini_engine `
  --model "D:\models\Qwen3-0.6B" `
  --prompt "介绍一下 KV Cache" `
  --device cuda `
  --dtype bf16 `
  --prefill-backend auto `
  --decode-backend cuda `
  --max-new-tokens 128 `
  --local-files-only
```

## 7. 测试结果

CUDA 专项测试：

```text
24 passed
```

覆盖：

- FP16/BF16；
- head dimension 16/32/64/80/96/128/192/256；
- GQA/MQA；
- 非连续 block table；
- block size 4/16/32；
- 1、255、256、257、1025、4097 等边界；
- 多 partition 合并；
- 当前 CUDA stream；
- invalid metadata；
- trusted mode 的坏 page 安全性；
- native logits 的 torch decode/cuda decode 对齐。

全量基础测试在当前环境为：

```text
18 passed, 3 skipped
```

这里的 3 个 skip 是非 CUDA 环境分支；CUDA 专项运行使用：

```powershell
. .\scripts\enter_project.ps1
$env:MINI_ENGINE_TEST_CUDA = "1"
python -m pytest tests/test_cuda_paged_attention.py -q
```

## 8. 当前性能状态

本阶段已完成 kernel 编译和正确性验证，但还没有把 CUDA kernel 的单算子 latency 与 PyTorch fallback 做正式 benchmark，也没有宣称端到端加速。下一步必须测：

- batch 1/4/8/16；
- context 128/1K/4K/16K；
- block size 16/32；
- FP16/BF16；
- torch fallback vs CUDA kernel；
- kernel latency、有效带宽、显存和端到端 TPOT。

## 9. 下一阶段

阶段 05 先做 kernel benchmark 和 profiler 数据，再根据结果优化：

1. 低 batch decode 的 warp 映射；
2. GQA K/V 复用；
3. vectorized load；
4. register/shared-memory 使用；
5. CUDA Graph；
6. 最后再接 RMSNorm、RoPE + cache write 和 SwiGLU fused kernel。
