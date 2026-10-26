# 技术报告目录

更新：2026-10-26。当前实现、约束和实验按主题维护；历史测量保留原有范围，不把不同轮次测试数相加。

功能、启动和当前限制见[README](../README.md)，后续优先级与交付约定见[开发路线](../DEVELOPMENT_PLAN.md)。

| 报告 | 保留内容 |
|---|---|
| [推理引擎架构与实现](ARCHITECTURE.md) | CLI/HTTP调用链与资源所有权；PrefillBatch、直接paged CUDA prefill、选择性LM head；Nano/Mini主要源码对照与工业参考 |
| [系统验证与性能评测](VALIDATION_AND_BENCHMARKS.md) | 已执行验收与历史数据，Nano-vLLM/MiniVLLM计量协议、平台阻碍及当前不可比较条件；没有三方速度排名 |
| [算子后端性能复核](BACKEND_PERFORMANCE_RECHECK.md) | 2026-10-24 FP16/W4A16原始观察、基线漂移、显存与输出差异 |
| [推理引擎性能分析指南](PROFILING.md) | 真实eager/Graph热点、PyTorch/NVTX/Nsight命令与采集限制 |

当前包含ragged query合批、CUDA直接paged attention和选择性词表投影；`auto/math/flash/dandan`仍逐请求dense attention，Graph仍仅decode。FP16合批的严格KV逐元素阈值未通过，多请求时间不发布正式倍率；选择性LM head主要减少额外显存，速度未一致改善。最新测试状态、完整模型验收与三路径消融统一见[验证报告](VALIDATION_AND_BENCHMARKS.md)和[ragged摘要](data/ragged_prefill_summary.json)。

阶段Markdown、重复项目总览和一次性迁移/能力探测快照已退出活动文档集。整理前工作区字节备份在本机忽略的`build/archive/structure_cleanup_20261026`，Git保留提交历史；下表JSON统一放在`data/`，移动前后内容哈希相同，内部历史路径保留原样，不把旧数据当当前性能。

## 保留的数据

JSON保留原文件名与内容，便于追溯原实验。它们是对应日期的记录，不能代表每次结构调整后的最新性能。

| 文件 | 用途 |
|---|---|
| [ragged_prefill_summary.json](data/ragged_prefill_summary.json) | PrefillBatch/分页suffix/共享页与行选择的验收、原始记录入口和三路径forward消融；独立于decode或HTTP性能 |
| [nano_mini_comparison_summary.json](data/nano_mini_comparison_summary.json) | 2026-10-25两个主对照的固定版本、历史性能来源/条件、源码与报告哈希、Windows可行性；不是新增跑分 |
| [structure_validation_summary.json](data/structure_validation_summary.json) | 2026-10-25结构维护：403项回归、真实BF16 chunk/prefix/Graph/HTTP-SSE、资源归还和wheel检查 |
| [stage07_elementwise_summary.json](data/stage07_elementwise_summary.json) | FP16/BF16 elementwise算子、Graph与eager测量 |
| [stage22_backend_validation_summary.json](data/stage22_backend_validation_summary.json) | FP16/W4A16与同权重oracle的短序列数值对拍 |
| [stage26_paged_summary.json](data/stage26_paged_summary.json) | D128 PagedAttention配对测量、误差、环境与指纹 |
| [stage27_system_summary.json](data/stage27_system_summary.json) | 完整模型chunk/prefix/Graph/HTTP验收及投机实验 |
| [stage29_chat_graph_summary.json](data/stage29_chat_graph_summary.json) | Chat完整decode Graph、356 tests、真实FP16交替配对测量 |
| [stage28_profile_summary.json](data/stage28_profile_summary.json) | 真实引擎CPU/GPU热点与采集状态 |

完整逐次样本位于本机`benchmarks/results/`，大体积trace位于`build/profiles/`，均不随仓库提交。JSON是对应测量时刻的证据；文件名中的阶段编号仅用于追溯，不是新的开发顺序。
