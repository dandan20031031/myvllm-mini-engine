# Benchmark 目录

当前入口：

| 脚本 | 范围 |
|---|---|
| `benchmark_paged_sweep.py` | 原CUDA与D128向量化kernel配对Graph计时，5种block size；非服务延迟 |
| `compare_paged_backends.py` | CUDA/Mini独立环境、实际张量哈希匹配、FP32 oracle及Graph/eager组件计时；不测引擎TPS |
| `benchmark_native_batch.py` | 真实scheduler eager/Graph对照，实际token统计和后端指纹 |
| `benchmark_ragged_prefill.py` | 同模型B1完整logits、B1末行logits、packed末行logits三路径forward消融；不测生成或HTTP |
| `benchmark_speculative.py` | 普通greedy/投机输出一致性、接受率和单请求耗时 |
| `benchmark_http.py` | 标准库HTTP/SSE负载，可对兼容服务采集usage、TTFT/P95与吞吐；`--compare`拒绝不匹配条件 |
| `validate_system_extensions.py` | 完整模型chunk/prefix/Graph与可选真实HTTP验证，仅正确性不测速 |
| `profile_native_batch.py` | 已预热真实runtime的PyTorch Profiler/NVTX，CPU范围、GPU kernel、Chrome trace；采集时间不是benchmark |

客户端usage缺失时，不以SSE chunk数代替token数。性能主对照为Nano-vLLM和MiniVLLM，vLLM/SGLang只作架构参考。本仓库尚无三个小型引擎的同条件实测排名。

benchmark工具统一放在本目录，推理包只包含运行代码。完整跨引擎对照还需共同离线adapter，不能直接把配置中的`output_tokens`当实测输出。

## Ragged Prefill内部消融

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\benchmark_ragged_prefill.py `
  --model .\models\Qwen3-0.6B --dtype fp16 `
  --prefill-backend cuda --data-backend cuda --elementwise-backend cuda `
  --linear-backend cuda --batch-sizes 1 2 4 --prompt-lengths 32 64 `
  --warmup 2 --repetitions 5 --output benchmarks\results\ragged_prefill.json
```

三种模式使用同一已加载模型、后端、输入token和预分配KV池：`sequential_full_logits`逐个B1计算全部query logits，`sequential_selected_logits`逐个B1仅计算末行，`ragged_selected_logits`合成一个packed batch并选择各请求末行。最大prompt长度按四分之一阶梯产生不同请求长度，先对拍末行logits、greedy token与KV，再轮转执行顺序测量。

计时包括公开forward校验、选择行索引校验/设备索引tensor创建、输出copy及完成同步；排除模型加载、tokenizer、PrefillBatch构造、host行选择规划、对拍、warmup、指纹采集与清理。CUDA Events记录dispatch时间线，包含launch间隙，不是纯kernel指令延迟。这里报告prefill输入token/s、forward wall time与显存峰值，**不测TTFT、decode、scheduler、sampling、生成TPS或HTTP**。实际服务收益另用`benchmark_native_batch.py`/HTTP工具验收。

CUDA模式直接读paged KV；显式`auto/math/dandan`模式仍逐请求dense attention，仅投影/MLP等合批，不报告为varlen Flash。显存分配峰值包含模型、KV池与输出buffer，额外allocated峰值相对调用前基线；reserved受allocator缓存及此前模式影响，不是隔离进程占用。结果范围和实际观察见[验证报告](../reports/VALIDATION_AND_BENCHMARKS.md)及[ragged摘要](../reports/data/ragged_prefill_summary.json)，不从功能覆盖预设加速比。

本机FP16 B2长度32/24在严格KV逐元素阈值下失败，脚本会保留诊断并在计时前停止。B1的严格验收可用`--batch-sizes 1`；显式`--skip-kv-check`仅记录探索性原始时间，标记`validation_incomplete`且禁止倍率。用`diagnose_ragged_prefill.py`复现逐层/head误差、8-token独立greedy与同舍入权重FP32对照。短输出一致不证明batch invariance，不应只为跑完矩阵而放宽阈值。

## Nano-vLLM / MiniVLLM 对照任务

架构、采样/Graph边界与固定版本见[架构报告](../reports/ARCHITECTURE.md)，历史数据、差异和完整协议见[性能评测报告](../reports/VALIDATION_AND_BENCHMARKS.md)。2026-10-26新增CUDA/Mini同输入PagedAttention组件对照，命令与限制见该报告；完整引擎共同adapter尚未实现，不能用组件结果宣布三方TPS排名。

`compare_paged_backends.py`的调用链为：`make_case`生成CPU输入与实际字节哈希 → 延迟加载provider并转GPU → FP32分页GQA oracle → 计时前对拍 → 重复单算子的Graph Event/eager wall计时 → 计时后对拍 → 保存输入、源码、binary、GPU快照和样本。`comparison_guard`拒绝未匹配输入、错误或源码变动的数据，并始终禁止完整引擎TPS结论；Nano不在该组件脚本中。

1. 固定模型哈希与离线token IDs；统一FP16、TP1、B=1/4/8/16、prompt128/512、输出128，分别eager/Graph。
2. Nano/Mini原版不接受greedy；共同temperature=0.6、top-k禁用/top-p=1，固定输出长度与ignore-EOS适配须披露。随机采样文本不相同不能直接判错，另做teacher-forced数值验证。
3. 使用`load/reset/enqueue/step/close`接口，统一timer只包执行循环；逐次记录actual tokens/elapsed、每请求首token/结束时间和平均TPOT，输出聚合TPS与P50/P95。
4. 首轮无共享前缀；冷/热Prefix、W4A16、HTTP服务各自单列，不把吞吐口径混为单请求延迟。
5. 记录commit、dirty diff、adapter、实际参数/KV dtype、模块`__file__`、软件栈、KV预算与GPU负载；关闭资源在计时之外。

平台情况：Nano当前本地GPU路径依赖NCCL/flash-attn，Windows未验证；Mini已有Windows修复，但editable安装仍指迁移前目录，须显式选定worktree。对照代码放独立环境，不引入本项目运行依赖；缺条件时结果标为未测。

本项目现成命令仍可独立用于内部消融：

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\benchmark_native_batch.py --help
.\.venv\Scripts\python benchmarks\benchmark_chat_graph.py --help
```

端到端 benchmark 应固定模型版本、GPU、dtype、采样参数、prompt 数据和 `max_new_tokens`，并分别报告 TTFT、TPOT、吞吐量、显存峰值以及 P50/P95 延迟。


## 阶段 07 elementwise 对比

~~~powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\benchmark_elementwise.py --dtype bf16
.\.venv\Scripts\python benchmarks\benchmark_elementwise.py --dtype fp16 --output benchmarks\results\elementwise_fp16.json
~~~

默认以 rows=[1,128]、hidden=[1024,3072] 做笛卡尔积；4 类算子各先校验输出再测量。
CUDA Graph Event timeline（32 调用/graph）降低 host 调度间隙；另保存 eager wall time，两种口径分开。
Graph 时间仍包含节点调度，不能称为纯 kernel 指令延迟。脚本不测 tokens/s 或模型吞吐，也不比较 vLLM。
