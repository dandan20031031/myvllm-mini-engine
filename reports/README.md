# 技术报告目录

整理日期：2026-10-24。原阶段00–29报告按主题合并，保留当前实现、关键设计约束、验证结果和复现入口；删除重复的逐阶段文件清单、旧测试计数和已完成的“下一步”。阶段编号仅用于定位历史实验文件。

项目能力与待办见 [项目总览](../PROJECT_STATUS.md)，安装、启动及常用命令见 [项目 README](../README.md)，开发路线见 [开发方案](../DEVELOPMENT_PLAN.md)。

| 报告 | 保留内容 |
|---|---|
| [推理引擎架构与实现](ARCHITECTURE.md) | 模型调用链、KV所有权、Prefix/chunk调度、Graph、CUDA/量化后端、投机与HTTP服务 |
| [系统验证与性能评测](VALIDATION_AND_BENCHMARKS.md) | 已执行验收、关键实验条件与结果、计时定义、复现入口和未完成项 |
| [算子后端性能复核](BACKEND_PERFORMANCE_RECHECK.md) | 2026-10-24 FP16/W4A16原始观察、基线漂移、显存与输出差异 |
| [推理引擎性能分析指南](PROFILING.md) | 真实eager/Graph热点、PyTorch/NVTX/Nsight命令与采集限制 |

## 保留的数据

JSON保留原文件名与内容，便于追溯原实验。它们是对应日期的记录，能力探测快照不代表后续实现状态。

| 文件 | 用途 |
|---|---|
| [stage07_elementwise_summary.json](stage07_elementwise_summary.json) | FP16/BF16 elementwise算子、Graph与eager测量 |
| [stage22_backend_validation_summary.json](stage22_backend_validation_summary.json) | FP16/W4A16与同权重oracle的短序列数值对拍 |
| [stage26_paged_summary.json](stage26_paged_summary.json) | D128 PagedAttention配对测量、误差、环境与指纹 |
| [stage27_system_summary.json](stage27_system_summary.json) | 完整模型chunk/prefix/Graph/HTTP验收及投机实验 |
| [stage29_chat_graph_summary.json](stage29_chat_graph_summary.json) | Chat完整decode Graph、356 tests、真实FP16交替配对测量 |
| [stage28_profile_summary.json](stage28_profile_summary.json) | 真实引擎CPU/GPU热点与采集状态 |
| [precision_capabilities.json](precision_capabilities.json) | 历史精度能力探测快照 |
| [cache_migration.json](cache_migration.json) | 项目缓存迁移审计记录 |

完整逐次样本位于本机 `benchmarks/results/`，大体积trace位于 `build/profiles/`，均不随仓库提交。阶段29新增Chat Graph验收与配对测量，其余历史数值保留原实验范围；不把不同轮次测试数相加。
