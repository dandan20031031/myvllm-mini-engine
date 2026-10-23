# 阶段25：Windows 原生 HTTP / SSE 推理服务

日期：2026-10-23。

本阶段把已有 `NativeBatchRuntime` 接入本地 HTTP 服务。不同 HTTP 请求进入同一个 GPU worker，由 continuous batching 合并执行；没有在每个 HTTP 线程中单独加载或运行模型。实现只依赖 Python 标准库与引擎已有依赖，支持 Windows 11，不引入 Linux、WSL、Triton 或 ASGI 服务依赖。

## 已实现的接口

| 接口 | 行为 |
|---|---|
| `GET /health` | 返回 worker 是否可用；关闭或异常退出后为503 |
| `GET /v1/models` | 返回当前唯一的 served model name |
| `POST /v1/chat/completions`，`stream=false` | 返回 OpenAI 风格的 message、finish_reason 和 usage |
| 同接口，`stream=true` | SSE role/content delta、结束原因、`data: [DONE]` |

支持请求字段为 `model`、`messages`、`max_tokens`、`stream`、`temperature`、`stream_options`。`model` 必须匹配服务名；messages 仅接受 system/user/assistant 角色和非空字符串 content。推理固定使用 greedy，temperature 必须为0（缺省也是0）。流式请求可传 `stream_options: {"include_usage": true}`，在finish chunk之后、`[DONE]`之前返回choices为空数组的usage chunk。未支持的字段直接400，避免接受了参数却忽略其含义。

JSON 请求必须声明 `application/json` 与唯一 `Content-Length`，拒绝重复 JSON 字段、NaN/Infinity、超大 body、chunked 请求 body 和不完整内容。输出 UTF-8 JSON；SSE 首条为 assistant role，最后一条 completion chunk 包含 finish_reason，随后发出 `[DONE]`。错误在stream开始前返回HTTP错误码；开始后返回SSE error对象并结束。

## 代码与调用逻辑

本阶段新增三个文件：

- `mini_engine/server.py`：服务、工作线程、输入检查、HTTP和SSE，以及独立启动CLI。
- `tests/test_server.py`：使用fake runner/runtime/tokenizer的真实loopback HTTP测试，不下载权重、不使用GPU。
- 本报告。

调用链：

```text
python -m mini_engine.server
  -> NativeQwen3Runner.from_pretrained：主线程只进行一次模型加载
  -> InferenceService.start
       -> 单一 mini-engine-worker 创建 NativeBatchRuntime
       -> 传入 prefill_chunk_size / max_num_batched_tokens / enable_prefix_cache
  -> ChatHTTPServer：有界 HTTP handler threads
       -> 校验 JSON / model / messages / decoding 参数
       -> InferenceService.submit：有界提交队列
       -> worker：tokenizer.apply_chat_template + context/KV 检查
       -> GenerationRequest + Scheduler.add_request
       -> NativeBatchRuntime.step：decode 优先，prefill 按 chunk 执行
       -> 更新 request.output_token_ids / state
       -> worker 解码文本，写有界结果队列
       -> HTTP thread 只序列化 JSON / SSE 并写 socket
```

所有运行期模型调用、runtime创建、tokenizer调用、scheduler状态修改、KV分配和释放都在同一个worker线程。HTTP线程通过线程安全队列和取消事件通信，不直接操作模型/cache。新请求可在已有请求decode期间入队，随后由同一runtime组成batch。

非流式响应只在请求结束时解码文本，避免每生成一个token都重复解码完整输出。流式响应只在新增token时解码，关闭tokenizer空格清理；尚未完成的UTF-8字节导致末尾替代字符时暂缓该后缀，等待后续字节。usage中的completion_tokens统计模型实际产生的token，包括特殊结束token；文本使用 `skip_special_tokens=True`。

## 有界资源与释放

- 默认最多32个正在服务/排队的推理请求，超限429；每请求结果队列默认32项。
- 默认最多64个HTTP连接处理线程；超过连接上限503，socket读写超时默认5秒。
- 请求body默认最多1MiB，messages最多128项。
- 请求总超时默认120秒，max_tokens默认上限512；prompt加输出预算受模型context和物理KV容量双重约束。
- 服务在admission时保守预留整个prompt+max_tokens所需页数，剩余请求等待；runtime仍按需分配物理页，并可淘汰prefix cache的外部引用以保证已接纳请求增长。
- Prefix Cache默认启用，可用 `--disable-prefix-cache` 显式关闭；服务默认prefill chunk为128 token。
- Graph开启后，同时执行的请求数不超过最大batch bucket；context超出固定Graph bucket覆盖范围时返回400。
- 客户端断线通过socket关闭检测/写入错误触发取消事件。worker在下一次调度前调用scheduler.abort，释放请求持有的KV引用。
- 慢SSE消费者不能阻塞worker：结果队列满时取消该请求，并发布slow_consumer错误。socket本身也有写超时。
- 编码/输入错误只结束对应请求。一个batch kernel异常无法可靠归因到某一行，因此结束当前active batch并释放其请求资源；worker继续接纳后续请求。不可恢复的GPU上下文错误仍可能需要重启进程。
- 服务关闭时取消active/waiting请求，清理Prefix Cache的pin和Graph scratch ownership。无法中断正在执行的CUDA kernel，退出等待发生在当前runtime.step返回之后。

## Windows 启动示例

在项目根目录启动：

```powershell
. .\scripts\enter_project.ps1

.\.venv\Scripts\python -m mini_engine.server `
  --model ".\models\Qwen3-0.6B" `
  --served-model-name "qwen3-0.6b" `
  --device cuda --dtype bf16 `
  --prefill-backend cuda --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda --linear-backend cuda `
  --fuse-projections --prefill-chunk-size 128 `
  --host 127.0.0.1 --port 8000 --local-files-only
```

需要Graph时附加 `--cuda-graph`，要求CUDA FP16/BF16，以及decode/data两个backend均为cuda。Linear/prefill可选dandan路径和INT4选项沿用runner；服务支持`--quantization int4 --quant-backend reference|dandan`，配置限制仍由runner提前检查。

另开PowerShell发送非流式请求：

```powershell
$body = @{
  model = "qwen3-0.6b"
  messages = @(@{ role = "user"; content = "你好，简单介绍一下自己" })
  temperature = 0
  max_tokens = 64
  stream = $false
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/chat/completions" `
  -Method Post -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

流式客户端将stream设为true，并逐行读取 `data: ...` 直至 `[DONE]`。Windows自带curl.exe可用于流式读取；不要将PowerShell的curl别名与curl.exe混淆。

默认只绑定 `127.0.0.1`。本阶段没有认证、TLS和多租户隔离；它是本地开发服务，不包含公网部署配置。

## 验证与边界

本阶段的服务测试使用真正 `127.0.0.1` TCP请求，没有用handler mock代替HTTP行为：

```powershell
.\.venv\Scripts\python -m pytest tests\test_server.py -q -o cache_dir=build/pytest_cache
```

本次结果：**29 passed（2.77秒）**。覆盖health/models、非流式/流式usage、SSE role/content/finish/DONE、UTF-8字节边界、连续batch与单worker所有权、输入/媒体/body限制、排队过载、请求错误后继续服务、断线取消、慢消费者、关闭释放和显式关闭Prefix Cache。

执行环境的文件系统沙箱最初拦截localhost client socket（WinError10013）；对上述纯CPU本地测试进行工具审批后，实际loopback测试通过。此事实不代表Windows服务本身需要管理员权限。

没有在本阶段运行GPU服务benchmark。fake-runtime HTTP测试验证的是协议/线程/生命周期；真实Qwen3的HTTP延迟、并发吞吐、长连接压力仍需在固定GPU环境独立测量，不能把此前CUDA Graph或单算子计时当作HTTP服务结果。

当前不是完整OpenAI API实现：未支持 tools/function calling、图像/音频输入、stop序列、n>1、随机采样、logprobs、response_format、embeddings、模型热切换、会话服务端记忆、draft-model或投机解码HTTP选项。每次请求应提供完整聊天历史，服务只处理请求内历史。
