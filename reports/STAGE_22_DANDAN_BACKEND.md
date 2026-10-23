# 阶段22：独立算子库后端、W4A16 与 Flash Prefill

日期：2026-10-23。平台：Windows 11、RTX 5080 SM120、PyTorch 2.9.1+cu128。

## 1. 本阶段交付

本阶段将独立维护的 `dandan_ops` 接入引擎，不修改算子库源码，也不将它绑定为必装依赖。默认 BF16/CUDA/cuBLAS 路径保留。

- FP16 Linear：`dandan` 调用库 auto 路由；`dandan_cuda` 强制 HGEMM。QKV/GateUp 融合后的 Linear 同样支持这两个后端。
- W4A16：decoder projection 使用 packed INT4，FP16 激活，真实 CUDA GEMV/Tensor Core 计算；支持 group64/128 和先融合再量化。
- Flash prefill：FP16、head_dim=128、原生 GQA，不先 repeat_interleave K/V。decode 保持引擎分页 KV 接口。
- 提前检查依赖、CUDA/dtype、所请求能力和模型 head_dim；报告库源码、二进制、JSON路由表指纹。
- 修复 Graph capture 和多个 inactive slots 共用 KV 写入页的问题。
- 增加完整模型数值验收、固定实际 token 数的 batch benchmark、真实 Graph logits/KV 回归。

## 2. 文件与调用逻辑

### `mini_engine/kernels/dandan_backend.py`

`require_dandan_ops()` 延迟导入可选库，缓存模块，并验证 CUDA 扩展的 W4A16 contract version=2。普通 torch/CUDA 引擎不触发导入。`validate_dandan_runtime()` 拒绝 CPU、非 FP16、不可用 GPU 和 SM80 以下设备。

`require_dandan_capability()` 按实际使用检查 `flash_attention/flash_inference` 或 `pack_int4/QuantizedLinear`。`dependency_report()` 对包中 Python、JSON profile 和本机二进制计算 SHA256，供外部库升级后的回归和结果追踪使用。指纹是本地文件身份记录，不等于已锁定发布版本或阻止其他进程修改文件。

### `mini_engine/kernels/linear.py`

`DandanLinear` 继承 `nn.Linear`，保持原 checkpoint 命名和 `[N,K]` 权重布局。forward 将任意前导维展平为 `[M,K]`，调用 `dandan_ops.gemm(..., weight_transposed=True)`，再恢复输出形状。FP16/CUDA/推理要求在入口检查，库错误直接向上传递。

`FusedLinear` 使用同一个后端分派。模型 MLP 保存 `linear_backend`，修复融合 Gate/Up 时丢失原后端的问题。`dandan` 允许库 auto 根据实测选择 cuBLAS/HGEMM/TMA；这不保证调用一定使用自研 kernel。需要强制验证自研 GEMM 时选 `dandan_cuda`。

### `mini_engine/quantization/dandan.py`

`DandanQuantLinear` 使用现有引擎量化器，保留对称量化结果：

```text
FP16 weight [N,K]
  -> Int4WeightOnlyBackend(group64/128).pack
  -> signed INT4，两个 nibble/uint8
  -> nibble XOR 8，等价于 q_unsigned=q_signed+8
  -> 八个 nibble/int32，zero=8
  -> 源 group128 的 scale 重复两次，源 group64 保持
  -> dandan_ops.QuantizedLinear(storage group64, backend="cuda")
```

转换不重新量化，部分尾组和 padding 也受测试覆盖。底层计算权重先反量化并舍入到 FP16，再参与乘法；因此数值 oracle 必须使用同一 packed state 的 FP16-rounded 权重。旧 FP32 反量化 reference 与这个计算契约有舍入差异。

`quantize_dandan_linears()` 先准备全部替换，再修改模型，失败时不留下部分量化模型。它选择 decoder 内 `nn.Linear/FusedLinear`；embedding、Norm、residual、LM head 和 KV 保持浮点。bridge metadata 保存源 group size、形状和 bridge version；底层 QuantizedLinear 保存实际 packing/version，拒绝不兼容 checkpoint。

`storage_bytes` 是 projection packed 权重和 scale/zero 的字节数，不代表整模型峰值显存。`accelerated_quant_gemm=true` 表示调用 CUDA 量化计算 kernel，不表示端到端比 cuBLAS 更快。

### `mini_engine/kernels/prefill.py`

`backend="dandan"` 在 GQA 展开之前直接调用 `flash_attention(..., causal=True, query_start=0, backend="cuda")`。库负责布局适配；实际拷贝成本计入引擎运行。它使用 dense prompt K/V，不改变分页 decode 的存储方式。

### `mini_engine/model/native_qwen3.py`、`native_runner.py`、`__main__.py`

CLI 参数传给 Runner；Runner 先检查配置和依赖，需要 Flash 时先读取 AutoConfig 检查 head_dim，再加载 tokenizer/权重。Native 模型复用 HF 权重，按要求先融合，再执行 `quantize(config, backend="dandan")`。

`backend_report()` 记录线性、attention、融合、量化和依赖信息。单轮 CLI JSON 带此报告；默认单请求/chat 仍是 eager，Graph 验收通过 scheduler batch runtime 完成。

### `mini_engine/engine/cuda_graph.py`

原来 capture 全槽和 inactive slots 都在 position=0 写同一个 scratch page。现在每个 slot 的 block table 第0列指向不同预留物理页，其余列可共享，因为 padding 的 context_len 固定为1。预留至少 `max(max_blocks,batch_size)` 页；不随 width×batch 过度分配。活跃请求仍使用自己的 block table。

### `benchmarks/benchmark_native_batch.py`

继续测 scheduler 生命周期，而非单个 attention kernel。新增量化选择、可选库信息、固定 token 预算、prefix 开关、实际 token 摘要和重复样本 P95。默认关闭 Prefix Cache；`--ignore-eos` 用于确保各路径都实际输出相同 token 数。P95 是这一小批重复样本的延迟分位数，不是在线服务请求 P95。

### `benchmarks/validate_dandan_qwen3.py`

真实0.6B模型分别验证 FP16/cuBLAS 与 dandan，W4A16 与相同 packed 权重的 FP16-rounded/FP32 accumulation oracle。记录 teacher-forced logits 和独立 greedy 序列；原 FP16 到 INT4 的量化变化单独报告，不能用 kernel 误差掩盖模型质量损失。

## 3. 验证与结果边界

- 最终全量回归：`201 passed, 2 warnings`，其中新后端测试48项；日志保存在 `build/stage22_pytest.log`。两个warning为既有CUDA Toolkit13.3与PyTorch cu128差异、setuptools私有MSVC接口。
- `tests/test_dandan_backend.py` 覆盖可选依赖、下载前拒绝、GQA Flash、group64/128/K70尾组、checkpoint、scale保持、融合、tiny完整模型 prefill/decode/logits/KV，以及 B4 Graph capture 和 active2/1 replay。
- 真实 Qwen3-0.6B 使用2条 prompt，每条 prefill+4次decode，FP16及W4A16同权重oracle均验证独立greedy序列；阈值为 logits `atol=0.125, rtol=0.01`。CUDA prefill、dandan Flash prefill 两种配置分别验证。
- 这只是短序列数值验收，不是 INT4 困惑度/任务质量评估，也不等于完整0.6B所有并发Graph形状验收。
- 两prompt结果：FP16与cuBLAS最大logit误差0.043～0.059（CUDA prefill）/0.047（Flash prefill）；W4与相同权重oracle最大误差0.059～0.063/0.060～0.066，teacher-forced argmax及独立greedy均匹配。原始FP16到INT4的量化变化明显，最大logit差约9.3～9.8，部分prompt输出改变；这不属于kernel误差，也不能据此宣称质量无损。
- 原始模型验收与性能文件在 `benchmarks/results/stage22_*`；可提交摘要在 `reports/stage22_backend_validation_summary.json`。
- 本轮性能运行期间独立库文件发生改变，GPU有外部持续负载。当前性能样本不用于 backend 间加速结论，不写入 README 作为正式速度结果。后续固定库版本、空闲 GPU 后重新验收。

## 4. 本机复现

```powershell
. .\scripts\enter_project.ps1
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:MINI_ENGINE_TEST_CUDA = "1"
.\.venv\Scripts\python -m pytest -q -p no:cacheprovider --basetemp build\tmp\stage22_pytest

.\.venv\Scripts\python benchmarks\validate_dandan_qwen3.py `
  --model models\Qwen3-0.6B --linear-backend dandan_cuda `
  --prefill-backend dandan --fuse-projections --decode-steps 4

.\.venv\Scripts\python benchmarks\benchmark_native_batch.py `
  --model models\Qwen3-0.6B --dtype fp16 `
  --prefill-backend dandan --linear-backend cuda --fuse-projections `
  --quantization int4 --quant-backend dandan --quant-group-size 128 `
  --ignore-eos --prefix-cache disabled --max-new-tokens 32 `
  --warmup 2 --repetitions 5 --local-files-only
```

算子库需预先安装到相同虚拟环境。引擎不修改独立库，也不自动构建它。

## 5. 后续

先固定依赖构建与GPU负载补公平性能测试；量化需评估较长任务输出质量。随后按阶段20优化 PagedAttention，并逐项接入库的新形状路由；Prefix 命中跳过、chunked/batched prefill、低精度KV和投机解码继续按计划推进。
