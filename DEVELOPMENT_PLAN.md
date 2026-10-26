# 开发路线与工程约定

当前范围：Windows 11、单GPU、Qwen3-0.6B。功能和启动见[README](README.md)，实现见[架构报告](reports/ARCHITECTURE.md)，数值与性能证据见[验证报告](reports/VALIDATION_AND_BENCHMARKS.md)。本文件只维护当前优先级和交付约定，不再累积阶段日志。

## 当前基线

- 原生Qwen3 forward、Paged KV、连续批处理、完整block Prefix复用与边界logits。
- decode优先的chunk调度；PrefillBatch打包query，投影/Norm/MLP合批。
- CUDA prefill直接读取paged历史KV；auto/math/flash/dandan仍逐请求dense attention。
- 选择性LM head保留末行和必要Prefix边界；默认完整logits接口继续供reference/投机验证。
- 单请求与batch decode Graph、固定metadata、独立scratch、投影融合。
- 可选dandan FP16/W4A16后端、greedy prompt lookup、CLI与HTTP/SSE聊天子集。
- runner、runtime和service的关闭与资源借用关系已经明确；prefill不在Graph内。

FP16合批的严格逐元素KV阈值仍未通过。短生成一致和FP32对照仅支持舍入差异解释，不能宣称batch invariance完成；多请求探索时间不发布正式加速倍率。选择性LM head减少额外显存，但尚未观察到稳定延迟收益。原始观察、阈值与限制保留在[ragged摘要](reports/data/ragged_prefill_summary.json)。

## 下一步优先级

| 优先级 | 交付 | 验收条件 |
|---|---|---|
| P0 数值边界 | FP16不同batch/投影形状的逐层误差与KV预算；扩展teacher-forced、较长生成和任务质量 | 预先定义误差协议、记录首个分歧位置；失败保留诊断，不为生成性能倍率临时放宽阈值 |
| P1 Prefill效率 | 在当前packed/paged路径上分析attention、GEMM、metadata与LM head成本，再决定分块、融合或访存优化 | B1完整行/B1选择行/packed选择行独立消融；同时验收共享页只读、不同历史长度、跨页与Prefix边界 |
| P1 主对照 | Nano-vLLM和MiniVLLM共同离线adapter与固定工作量协议 | 同模型哈希、实际dtype、token IDs、输出预算、采样和KV预算；记录commit/dirty补丁、计时范围与逐次样本；缺平台条件标未测 |
| P2 执行接口 | 减少CLI/Chat与batch路径在输入准备、采样、异常回收上的行为漂移 | 保留已支持的入口与结果语义；不为单卡项目引入多进程Executor或第二套调度框架 |
| P2 后端契约 | 根据实际热点扩展dandan varlen/paged能力或形状路由 | 明确dense/paged、dtype/head_dim、suffix位置、workspace和Graph条件；库升级重新验收，不能将dense Flash称为paged/varlen集成 |
| P2 环境复现 | 固定验证过的Windows依赖组合，补CPU CI与可选GPU验收入口 | clean环境可安装、导入和执行smoke；记录PyTorch CUDA/Toolkit/驱动与算子库版本，避免只在当前虚拟环境可用 |
| 后置研究 | FP8 GEMM、FP8/INT8 KV、小draft/随机投机、必要的异步或preemption | 每项独立给出Windows可行性、数值/质量、资源生命周期和性能证据；不由接口占位宣布完成 |

两套主对照均看架构与性能；vLLM/SGLang、DeepInfer只作架构或机制参考，不要求速度排名。Nano原版的NCCL/flash-attn依赖仍阻碍当前Windows GPU运行；Mini使用的本地Windows补丁必须与upstream区别记录。对照环境不成为本引擎运行依赖。

## 持续约束

1. **平台与存储**：不引入Linux、WSL或Triton运行依赖。源码、虚拟环境、模型、可配置缓存、编译与临时产物留在项目所在D盘；首次安装也先配置项目缓存。
2. **模型范围**：保留Transformers加载和正确性参考；Qwen3.5、混合架构、多模态、多GPU、PD分离及独立权重加载器暂缓。
3. **CUDA路径**：每类热点算子有明确后端和reference。Linear可使用cuBLAS/cuBLASLt；有CUDA实现不等于比库实现更快。融合不要求每个逻辑层独立launch。
4. **精度口径**：区分权重、激活、累加和KV格式。INT8/INT4/FP8 reference反量化不是低比特Tensor Core加速；NVFP4/MXFP8仍reserved。
5. **算子库独立**：dandan_ops显式可选，检查ABI、设备、dtype和contract；记录源码/二进制/route指纹。显式后端不支持时提前报错，不静默替换后端。
6. **资源所有权**：共享Prefix页只读；调度准入与物理页引用分离。异常、取消、EOS和关闭归还本会话引用，借用页不释放；GPU最后使用结束后才能回收。
7. **Graph边界**：仅decode；metadata地址固定、capture scratch与live页分离，保存输出视图需clone。权重或后端变化必须重新capture。
8. **投机边界**：现有greedy prompt lookup无独立draft模型。随机投机需要接受概率和拒绝分布校正；接受率预算不称为模型置信度。

## 每次交付的验收

- 对改动运行相关CPU/reference测试；改动CUDA、Graph或后端时实际运行对应GPU路径，明确skip和未安装能力。
- 模型层改动验证logits、KV、生成、Prefix/分块和资源归还；服务改动验证HTTP/SSE、取消、慢消费者和关闭。
- 需要性能结论时先完成数值验收，再固定GPU负载、模型、dtype、形状、warmup、缓存冷暖和计时边界。
- 区分kernel、forward、decode step、完整生成和HTTP。CUDA Event时间线包含dispatch空隙，不能称为纯指令时间；SSE chunk数不当token数。
- 保留eager/reference或原实现作为可复现对照。报告P50/P95与实际token数，记录源码、后端和模型指纹；未受控或未通过阈值的数据只作探索。
- Windows构建核对PyTorch CUDA与Toolkit组合；版本提示不能代替目标机测试。新kernel的数值测试不等于compute-sanitizer已通过。

## 文档与Git

- README只维护功能、快速开始、当前限制和少量有条件的性能结果。
- 本文件维护路线；架构、验证、性能分析按[主题报告](reports/README.md)维护，同一事实不在多个阶段报告重复扩写。
- JSON测量证据保留原内容与实验范围，不覆盖历史样本来伪造最新结果；大体积raw/trace留在忽略目录。
- 阶段Markdown和一次性迁移记录退出活动目录；Git保留历史，整理前的工作区原字节另存本机忽略的build/archive，不清理用户未提交脚本或模型缓存。
- 每个可验收工作单元在codex/分支完成必要检查、对应主题文档和Git提交，再按既有约定合入main并同步远端；不提交模型、缓存、构建产物或凭据。
- 保留用户已有独立改动；失败与未运行项明确报告，不把有阻塞的任务标记完成。
