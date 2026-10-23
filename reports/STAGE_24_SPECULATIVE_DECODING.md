# 阶段 24：可执行的 greedy Prompt Lookup 投机解码

本阶段接通单请求投机解码：从 prompt 和已接受输出的历史 ngram 中复制草稿，由目标模型一次验证多个 token。没有引入额外 draft 模型，也没有引入 Linux、WSL 或 Triton 依赖。

## 已实现的功能

- 查找当前历史后缀的最长 ngram 匹配；长度相同时优先最近的历史位置，复制已知后续 token。
- 将一枚待写入 KV 的已输出 token 与整段草稿一起送入目标模型，以正确的 suffix causal offset 一次计算多个 logits。
- 接受连续正确草稿；第一枚不匹配的位置输出目标模型的纠正 token；全部接受时额外输出一枚 bonus token。
- 拒绝后回退逻辑 KV 长度，释放不再需要的整页；保留页内的拒绝位置在下一轮覆写，不参与后续 attention。
- 无匹配草稿时退化为普通逐 token greedy；`max_draft_tokens=0` 也可显式关闭草稿。
- 正确处理首 token、接受草稿、纠正 token 和 bonus token 上的 EOS，以及输出长度限制、异常和资源清理。
- 可选依据最近 8 个有草稿轮次的 `accepted/proposed` 历史比例调整 draft budget。这里没有使用模型置信度。
- 返回提案命中率、token 接受率、验证轮次、target forward 次数、纠正与 bonus 数量、逻辑回退 token 数和真实请求耗时。

## 代码与调用逻辑

### `mini_engine/speculative.py`

`prompt_lookup()` 从已存在的历史中复制候选，不预测未见 token。候选可能错误，必须通过目标验证。搜索窗口目前为完整历史，复杂度随历史长度增长；后续可使用 ngram 索引降低 Python 查找成本。

`AcceptanceBudget` 维护最近的接受/提案计数。固定模式始终使用配置上限；自适应模式将接受比例映射到 `[1,max_draft_tokens]`，没有提案时保持原先历史。最后一轮可能因剩余输出预算而将实际提案上限降到 0。

`generate_speculative(runner, ...)` 重用 `NativeQwen3Runner` 的模型、tokenizer、BlockManager 和 KV cache，返回 `SpeculativeGenerationResult`。结果保留普通生成的 `text/token_ids/finish_reason/ttft_ms/tpot_ms/total_latency_ms`，增加可直接序列化为 JSON 的 `stats`。

执行顺序：

1. 校验参数、设备上下文容量、KV 容量与单请求 busy 状态。非 greedy 的 temperature/top-k/top-p 配置直接拒绝。
2. 对 prompt 正常 prefill，输出第一枚目标 greedy token。此时 prompt 已缓存，刚输出 token 尚未写入 KV。
3. 查找历史草稿，将 `[pending_token, draft...]` 一次送入 `model(ids, cache, blocks, position=...)`。
4. 按顺序比较草稿与目标 greedy logits。遇到拒绝立即停止接受后续草稿；全部接受则取最后一行 logits 的 bonus token。
5. 只保留 pending 输入和已接受草稿对应的 KV 前缀；纠正或 bonus token 成为下一轮 pending 输入。
6. 遇 EOS 或长度上限结束；`finally` 释放本请求拥有的全部 KV 页并清除 busy 状态。分配失败时不会释放其他请求已拥有的 `single` 页。

### 目标验证与 KV 接口

相关支持位于 `mini_engine/model/native_qwen3.py`、`mini_engine/kernels/prefill.py` 和 `mini_engine/cache/block_manager.py`：

- `model(ids, cache, blocks, position=n)` 支持非空多 token suffix，返回 `[len(ids),vocab]`。
- suffix 的 query 位置从 `n` 开始，causal mask 只允许读取各 query 之前及本位置的 KV。
- 单 token decode 保持分页读取；多 token suffix 验证当前会 gather 所需 KV，再调用支持 offset 的 causal attention。
- `BlockManager.truncate(request_id,keep_tokens)` 释放逻辑尾部整页，不清零保留页，不删除其他 owner/prefix pin 的引用。

这意味着阶段 24 已能运行真实验证，但多 token 验证尚未采用直接分页的高性能 verification kernel。它的成本可能抵消 forward 次数减少带来的收益。

### 为什么拒绝后不会读到错误草稿

设已缓存长度为 `p`，本轮输入为 `[pending,d0,d1,d2]`。若只接受 `d0`，则目标已经物理写入 4 个位置，但逻辑上只保留 `pending,d0`，下一轮从 `p+2` 写入纠正 token。attention 的有效长度以新的逻辑位置为准，尾部旧内容不会被读取。

草稿上限还受 `remaining_output_tokens-1` 限制，为纠正或 bonus 预留一枚输出预算。因此最远写入位置始终不超过普通生成检查的 `prompt_tokens+max_new_tokens-1`。

### `tests/test_speculative.py`

CPU 测试使用实际物理页编号访问的 token cache，而不是忽略 KV 的空壳模型。目标 logits 可依赖完整因果前缀，用来验证拒绝后的旧 KV 不会污染下一轮。

本阶段专项测试记录：**27 passed**。

覆盖：最长/最近 ngram 匹配、完整接受与 bonus、首位置拒绝、接受部分前缀再纠正、页回退、无草稿回退、四种 EOS 位置、输出预算边界、异常清理、页所有权冲突、busy/容量/参数校验、接受率预算、小型原生 Qwen3 对拍，以及 8 组因果前缀/EOS/自适应组合验证。

### `benchmarks/benchmark_speculative.py`

对同一个 runner 的全新请求，配对测量普通 eager greedy 和投机路径。重复实验交替两种路径的执行顺序，报告实际生成 token、逐次输出一致性、耗时、接受率、提案命中率与 target 调用次数。两种路径都不使用 decode CUDA Graph 或连续批处理。

计时从 prefill 前开始，包含草稿查找、目标执行、接受和 KV 回退，排除权重加载、tokenization 与最终文本 decode。投机路径会成组产出 token；报告的 TPOT 是总体平均值，不代表逐 token 流式交付间隔。

## 可复现实验

在项目目录的 PowerShell 中先执行：

```powershell
. .\scripts\enter_project.ps1
$env:PYTHONDONTWRITEBYTECODE = '1'
```

不需要 GPU 或模型下载的确定性实验：

```powershell
.\.venv\Scripts\python benchmarks\benchmark_speculative.py `
  --toy --max-new-tokens 32 --warmup 1 --repetitions 3 `
  --output benchmarks\results\stage24_speculative_toy.json
```

本次 CPU toy 记录：两种路径都输出 32 枚相同 token；普通路径调用目标 32 次，投机路径 8 次；24 枚提案全部接受，提案命中率为 `6/7`。CPU toy 的中位耗时约为 `0.623/0.308 ms`。这些数值仅说明算法与统计入口能执行，不能作为 Qwen3、CUDA 或服务性能的加速证明。

真实模型实验入口：

```powershell
.\.venv\Scripts\python benchmarks\benchmark_speculative.py `
  --model .\models\Qwen3-0.6B --device cuda --dtype bf16 `
  --prefill-backend cuda --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda --linear-backend cuda `
  --fuse-projections --max-new-tokens 32 --max-draft-tokens 4 `
  --warmup 1 --repetitions 3 `
  --output benchmarks\results\stage24_speculative_qwen3.json
```

可加 `--adaptive-draft` 测试接受率预算；可加 `--ignore-eos` 固定输出 token 预算。启用后一并作用于普通和投机路径。只有在输出一致且 GPU 无外部负载时，真实延迟比较才有意义。近似并行算法可能因数值舍入使接近并列的 logits 改变 argmax，实验必须保留逐次输出对照。

## 完整Qwen3-0.6B实测补充

RTX5080、BF16、项目CUDA、projection fusion、32实际输出token（ignore EOS）、warmup1、重复3次，两个prompt的逐token输出全部与普通greedy一致；算子库/后端记录未变。

| 场景 | 普通中位总耗时 | 投机中位总耗时 | target调用 | 接受率 |
|---|---:|---:|---:|---:|
| 重复指定英文句子 | 537.39 ms | 300.87 ms | 32→16 | 80% |
| 用一句话解释KV Cache | 422.81 ms | 415.17 ms | 32→32 | 0% |

观察比值分别1.79x/1.02x，仅适用于这次单请求eager实验；未证明独占GPU，不代表Graph或HTTP服务收益，也不代表所有prompt无损/加速。原始输出与所有样本在`benchmarks/results/stage24_speculative_qwen3.json`。无重复内容时应保留普通decode选择；并行suffix和单token矩阵乘的浮点舍入仍可能影响接近并列的logits。

## 当前边界与后续工作

- 已集成的是 greedy prompt-lookup，未集成随机采样分布校正。
- 当前不需要额外模型；尚未实现小 draft model 的加载和调度。
- EAGLE-3、MTP 等方法需要相应模型结构、draft head 或训练权重，本阶段没有将普通 prompt lookup 命名为这些方法。
- 未接入批量请求投机调度、流式服务、投机 CUDA Graph 和直接分页多 query 验证 kernel。
- 真实性能取决于重复内容命中率和验证 kernel 成本；无重复的 prompt 可能没有收益。
- 已补真实Qwen3短序列实测；尚无完整任务集质量、长上下文或服务性能验收。
