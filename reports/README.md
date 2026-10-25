# 技术报告目录

整理日期：2026-10-24；结构维护、ragged/paged prefill更新：2026-10-25。阶段00–29及后续的实现、设计约束、验证结果与复现入口按主题维护，历史阶段资料和原始实验记录保留其原有范围。阶段编号用于定位历史交付，不把不同轮次测试数相加。

项目能力与待办见 [项目总览](../PROJECT_STATUS.md)，安装、启动及常用命令见 [项目 README](../README.md)，开发路线见 [开发方案](../DEVELOPMENT_PLAN.md)。

| 报告 | 保留内容 |
|---|---|
| [推理引擎架构与实现](ARCHITECTURE.md) | CLI/HTTP调用链与资源所有权；PrefillBatch、直接paged CUDA prefill、选择性LM head；Nano/Mini主要源码对照与工业参考 |
| [系统验证与性能评测](VALIDATION_AND_BENCHMARKS.md) | 已执行验收与历史数据，Nano-vLLM/MiniVLLM计量协议、平台阻碍及当前不可比较条件；没有三方速度排名 |
| [算子后端性能复核](BACKEND_PERFORMANCE_RECHECK.md) | 2026-10-24 FP16/W4A16原始观察、基线漂移、显存与输出差异 |
| [推理引擎性能分析指南](PROFILING.md) | 真实eager/Graph热点、PyTorch/NVTX/Nsight命令与采集限制 |

2026-10-25结构维护的计划放在[开发方案](../DEVELOPMENT_PLAN.md)，实现说明并入架构报告，验收见[系统验证与性能评测](VALIDATION_AND_BENCHMARKS.md)；不另增重复阶段报告。全量403项测试与真实BF16模型系统验收通过，没有新增性能结论；Qwen3.5等新架构与独立权重加载器暂缓。

随后实现ragged query合批、CUDA直接paged attention和选择性词表投影；`auto/math/flash/dandan`仍逐请求dense attention，decode Graph范围不变。当前验收与三路径forward消融统一放入验证报告和[ragged摘要](ragged_prefill_summary.json)，不将prefill输入token吞吐称为生成TPS，也不据此更新历史跨引擎排名。

本轮最终467 tests通过，BF16 CUDA、FP16 dandan及W4A16完整模型系统验收通过。选择性LM head减少额外显存，速度未一致改善；FP16合批的严格KV逐元素阈值未通过，报告保留数值诊断，多请求时间不发布正式倍率。

## 保留的数据

JSON保留原文件名与内容，便于追溯原实验。它们是对应日期的记录，能力探测快照不代表后续实现状态。

| 文件 | 用途 |
|---|---|
| [ragged_prefill_summary.json](ragged_prefill_summary.json) | PrefillBatch/分页suffix/共享页与行选择的验收、原始记录入口和三路径forward消融；独立于decode或HTTP性能 |
| [nano_mini_comparison_summary.json](nano_mini_comparison_summary.json) | 2026-10-25两个主对照的固定版本、历史性能来源/条件、源码与报告哈希、Windows可行性；不是新增跑分 |
| [structure_validation_summary.json](structure_validation_summary.json) | 2026-10-25结构维护：403项回归、真实BF16 chunk/prefix/Graph/HTTP-SSE、资源归还和wheel检查 |
| [stage07_elementwise_summary.json](stage07_elementwise_summary.json) | FP16/BF16 elementwise算子、Graph与eager测量 |
| [stage22_backend_validation_summary.json](stage22_backend_validation_summary.json) | FP16/W4A16与同权重oracle的短序列数值对拍 |
| [stage26_paged_summary.json](stage26_paged_summary.json) | D128 PagedAttention配对测量、误差、环境与指纹 |
| [stage27_system_summary.json](stage27_system_summary.json) | 完整模型chunk/prefix/Graph/HTTP验收及投机实验 |
| [stage29_chat_graph_summary.json](stage29_chat_graph_summary.json) | Chat完整decode Graph、356 tests、真实FP16交替配对测量 |
| [stage28_profile_summary.json](stage28_profile_summary.json) | 真实引擎CPU/GPU热点与采集状态 |
| [precision_capabilities.json](precision_capabilities.json) | 历史精度能力探测快照 |
| [cache_migration.json](cache_migration.json) | 项目缓存迁移审计记录 |

完整逐次样本位于本机 `benchmarks/results/`，大体积trace位于 `build/profiles/`，均不随仓库提交。阶段29新增Chat Graph验收与配对测量，其余历史数值保留原实验范围；不把不同轮次测试数相加。
