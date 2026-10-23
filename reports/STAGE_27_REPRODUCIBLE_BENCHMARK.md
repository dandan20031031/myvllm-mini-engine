# 阶段 27：可复现的 OpenAI SSE 外部负载测试

本阶段提供独立 HTTP benchmark 客户端。MiniQwenEngine、vLLM、SGLang 只要提供兼容的 `/v1/chat/completions` SSE 接口，就可以使用同一份请求负载和同一套计时方法。客户端只依赖 Python 标准库，可在 Windows 原生运行，不加载 PyTorch 或 CUDA。

这交付了横向测试工具与统计口径，没有交付三种引擎的真实性能排名，也没有根据引擎标签假定测试条件相同。

## 实现文件与调用关系

### `benchmarks/benchmark_http.py`

调用链为：

```text
load_workload
  → 规范化 messages，计算 workload SHA256
  → 固定 seed 打乱测量请求顺序
  → run_phase(warmup)，等待全部结束
  → run_phase(measured)
      → ThreadPoolExecutor 限制同时发出的请求数
      → run_stream_request
          → HTTP POST /chat/completions
          → stream_events 解析 SSE
          → 记录 content 时间点、真实 usage、结束或错误
  → summarize 根据原始 request records 计算统计量
  → 保存完整 JSON
```

`load_workload()` 支持两种 JSON：

```json
{"prompts": ["Explain KV Cache in one sentence.", "用一句话解释连续批处理。"]}
```

```json
{"requests": [{"messages": [{"role": "system", "content": "回答简洁。"}, {"role": "user", "content": "什么是 KV Cache？"}]}]}
```

它们都会规范化为 `messages` 列表，再以确定性的 JSON 序列化计算 `workload_sha256`。支持 Windows PowerShell 常见的 UTF-8 BOM 文件。工作负载仅描述输入；`max_tokens`、并发、重复次数和 temperature 等条件单独保存在报告中。

`run_stream_request()` 发出 `stream=true`、`temperature=0`、`stream_options={"include_usage":true}` 的请求。SSE 解析支持注释、多行 `data:`、空 role chunk、内容 chunk、独立 usage chunk 与 `[DONE]`。流中的错误、缺少 `[DONE]`、非法 JSON、连接错误和 HTTP 429 等都会保留一条失败记录，包括已经收到的部分文本。

`run_phase()` 使用标准库线程执行请求。每个请求使用自己的连接；并发度是同时进行 HTTP 请求的上限，不是固定到达率。测量请求按照 workload × repetitions 展开，以 seed 决定提交顺序；seed 不会被包装成已经设置了服务器采样 RNG。

`compare_reports()` 是可选的严格比较入口。它不会通过标签决定哪个报告来自更快的引擎，而是先检查条件和实际输出，再决定是否计算观测吞吐比。

### 服务端兼容项

`mini_engine/server.py` 已支持 `stream_options={"include_usage":true}`。正常顺序为：内容 chunk → finish_reason chunk → `choices=[]` 的 usage chunk → `[DONE]`。usage 使用服务端实际生成的 token 数，包含不会显示为普通文本的生成 token。

客户端使用相同参数向其他兼容服务请求 usage。如果服务不返回，就明确记录缺失。

### `tests/test_http_benchmark.py`

CPU 测试通过本机动态端口的 `ThreadingHTTPServer` 模拟 SSE，不加载真实模型、不占用 GPU。

专项结果：**23 passed**。覆盖内容 chunk 与 token 计数分离、真实 usage 的 TPOT、缺失/非法 usage、单 token、无可见内容、HTTP 错误、流中错误、不完整终止、非法 JSON、连接失败、并发上限、aggregate TPS、request 级 P50/P95、多行 SSE、workload 哈希、UTF-8 BOM、warmup 隔离、API key 不写入报告，以及比较条件不一致的拒绝行为。

沙箱曾阻止 `127.0.0.1` 连接并返回 WinError 10013。允许本机回环连接后，上述测试通过；没有将该环境限制当作客户端通过测试的证据。

## 计时与 token 口径

| 指标 | 定义 |
|---|---|
| request latency | HTTP worker 准备发送请求开始，到收到 `[DONE]` 或失败为止，含建连、网络、服务端排队和流解析 |
| TTFT | 请求开始到首个非空 `delta.content` |
| TPOT | `(last_content_ms-first_content_ms)/(usage.completion_tokens-1)`；仅在完成 token 数大于 1 且存在可见内容时提供 |
| 总输出 TPS | 成功请求的实际 `usage.completion_tokens` 之和，除以整个测量阶段 wall time |
| P50/P95 | 从 request 原始样本直接计算的线性插值分位数，不平均各次分位数 |

注意：

- **一个 SSE content chunk 不等于一个 token。** 报告的 `content_events` 只是传输观察值，绝不用来替代 tokenizer token 数。
- 任一成功请求没有有效 completion usage 时，`total_completion_tokens/total_tokens_per_second` 为 `null`，相应 request 的 TPOT 也为 `null`；已知 token 小计仍单独保留。
- TTFT 测量的是首个可见字符串，不是服务端内部生成首枚 token 的时间。特殊 token、UTF-8 拼接或文本成组发送都会影响该差异。
- 总 TPS 的分母包含失败请求所耗的时间；分子只统计成功请求的真实 completion tokens，不纳入错误流的部分 token。
- 成功请求 latency 分位数与所有终止请求 latency 分位数分别保存，HTTP/传输/协议错误另行计数，避免失败请求被丢弃。
- 线程执行器中的客户端等待不进入单 request latency，但进入整个负载阶段的 wall time。服务端排队包含在 request latency 中。
- warmup 全部结束后才开始测量，warmup records 单独保存。它们不进入测量分位数或总 TPS；服务端缓存仍可能受到 warmup 的影响，必须声明缓存策略。
- `--timeout` 设置 HTTP/socket 操作超时；该值与服务端自己的请求超时应一并记录，不应把它解释为可强制中断 GPU kernel 的期限。

## 使用方法

先在项目目录准备环境：

```powershell
. .\scripts\enter_project.ps1
$env:PYTHONDONTWRITEBYTECODE = '1'
```

在已经启动的本机服务上使用默认的两条 prompt：

```powershell
.\.venv\Scripts\python benchmarks\benchmark_http.py `
  --base-url http://127.0.0.1:8000/v1 `
  --engine-label mini_engine --model qwen3-0.6b `
  --server-os "Windows 11" --gpu "RTX 5080" `
  --concurrency 4 --repetitions 10 --max-tokens 32 `
  --warmup-requests 2 --seed 41 `
  --output benchmarks\results\http_mini_engine.json
```

`--model` 是服务端公布的 model 名字，不是本地模型目录；MiniQwenEngine 默认的 served model 名为 `qwen3-0.6b`。不同引擎可以使用相同 served alias，但 alias 相同本身不能证明权重相同。

需要固定输入时，将上文 JSON 保存为项目内的 `benchmarks/workload.json`，增加：

```powershell
--workload benchmarks\workload.json
```

测试另一个服务时只替换 endpoint 和引擎标签，并保持可比条件。客户端不会为 vLLM 或 SGLang 安装依赖、启动服务或改变其运行环境。需要鉴权时，从 `--api-key-env` 指定的环境变量读取 Bearer key，默认变量名为 `OPENAI_API_KEY`；不将 Authorization header 保存到报告。

每个工作负载条目会测量 `repetitions` 次。默认两个 prompt、重复 10 次对应 20 个测量请求；warmup 请求数量额外计算。

## 后端 manifest 与严格比较

使用 `--backend-manifest path.json` 保存运行条件。建议将以下字段放在 `comparison` 对象中，并把具体引擎版本、启动命令、算子实现、构建指纹等放到单独的 `implementation` 对象：

| 必需 comparison 字段 | 内容 |
|---|---|
| checkpoint_id | 实际 checkpoint 的文件哈希或固定 revision |
| tokenizer_id | tokenizer 的文件哈希或固定 revision |
| chat_template_sha256 | 使用的聊天模板哈希 |
| dtype / quantization | 实际权重、激活精度和量化方案 |
| kv_cache_dtype | KV 存储精度 |
| max_model_len | 配置的模型上下文上限 |
| kv_cache_bytes | 声明的 KV 容量预算 |
| prefix_cache_policy | prefix cache 的启用状态与预热条件 |

以上字段由操作者提供，客户端不会伪装成已经验证了远端硬件或文件。驱动、CUDA、服务端启动参数和其他会影响比较的条件，也应加入 comparison；该对象整体不一致时，比较工具拒绝计算倍率。

```powershell
.\.venv\Scripts\python benchmarks\benchmark_http.py `
  --compare benchmarks\results\http_mini_engine.json benchmarks\results\http_other.json `
  --output benchmarks\results\http_comparison.json
```

工具检查：model、workload hash、负载参数、sampling、声明的 OS/GPU、comparison manifest、逐请求实际 prompt/completion token 数及输出文本哈希。缺少原始记录、usage、后端条件或存在失败请求时，也拒绝计算倍率。

OS 或精度有意不同的实验仍可以分别保留报告、解释差异；严格比较入口会保留拒绝原因，不把这种实验自动描述成同条件速度对比。即便条件完全一致，输出也带有 `remote_conditions_independently_verified=false`，仅表示声明条件匹配。

## 报告内容与当前边界

JSON 保存引擎标签、model、客户端实际 OS/Python、用户配置的服务端 OS/GPU、seed、后端 manifest 及哈希、规范化工作负载及哈希、独立 warmup、测量 raw records、错误、usage 覆盖率、P50/P95 和 aggregate TPS。原始请求记录足以复核统计口径。

本阶段尚未提供真实三引擎对照结果。客户端不控制远端 GPU 的独占使用、频率、驱动、tokenizer 文件或 KV 初始状态，也不支持固定请求到达率、延迟预算下的容量搜索或多轮长时间稳定性压测；后续可以在当前记录格式上继续扩展。

## 完整模型系统验收（2026-10-24）

`benchmarks/validate_system_extensions.py`加载本地Qwen3一次，每组实验创建独立KV池。`counted_prefill`用forward hook计数实际计算的prompt token，`execute`驱动真实scheduler；检查prefix共享页内容与allocator引用计数。Graph测试确实捕获batch2/4，并与eager逐请求输出比较。可选`validate_http`用本机临时端口并发发JSON/SSE请求，校验usage、finish、DONE以及服务关闭后的资源释放。

本机RTX5080以下三组全部通过chunk/full、cold/warm/partial prefix及batch2/4 Graph对拍：

| 模式 | Linear / Prefill | 结果 |
|---|---|---|
| BF16 | CUDA/cuBLAS + 自有CUDA | 全通过；另含真实并发HTTP/SSE |
| FP16 | dandan_cuda + dandan Flash | 全通过 |
| W4A16 | CUDA LM head + dandan INT4投影 + dandan Flash | 全通过 |

每组64-token前缀完整命中后的计算量64→0；5-token后缀仅计算5；回收后页引用归零。输入64-token、输出8-token的固定短序列仅检验系统语义；量化对拍比较相同量化模型的不同执行路径，不是声称W4与BF16质量相同。库指纹在各次验收期间一致。

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\validate_system_extensions.py --fuse-projections --http-smoke
.\.venv\Scripts\python benchmarks\validate_system_extensions.py --dtype fp16 --linear-backend dandan_cuda --prefill-backend dandan --fuse-projections
.\.venv\Scripts\python benchmarks\validate_system_extensions.py --dtype fp16 --prefill-backend dandan --quantization int4 --quant-backend dandan --fuse-projections
```

2026-10-24最终全量回归：**325 passed、2 warnings**；CUDA实际执行，HTTP loopback实际连接。日志在`build/internship_pytest_final.log`；紧凑数值摘要为[stage27_system_summary.json](stage27_system_summary.json)。`benchmark_native_batch.py`已改用runtime正式的`enable_prefix_cache`开关，不再替换cache对象，保持scheduler淘汰回调与对象一致。

阶段23–26的实现、部署和本机验证已完成；阶段27完成比较工具和系统验收。阶段15的外部引擎运行数据、全面任务质量、独占GPU公平性能实验仍未完成；本阶段不以接口完成替代这些结果。
