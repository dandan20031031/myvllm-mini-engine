# 阶段 11：CUDA Graph Decode Buckets

## 1. 阶段目标

本阶段建立 shape-bucketed CUDA Graph 基础设施，减少 decode 阶段 Python/dispatcher/kernel launch 开销。投机解码已加入开发计划，但不在本阶段伪装成已实现。

## 2. Bucket 是什么

bucket 是固定 CUDA Graph 执行形状，不是 KV Cache block：

```text
KV block -> 管理物理 KV 页
Graph bucket -> 固定 batch size、block table 宽度和 tensor 地址
```

例如 batch=3、最大 block 数=40 会选择：

```text
batch bucket = 4
block bucket = 64
```

第 4 个 slot 使用 padding block 和 inactive 输入，不能复用活跃请求的 KV 页。

## 3. 文件和调用逻辑

文件：`mini_engine/engine/cuda_graph.py`

```text
DecodeGraphManager.shape_for
  -> 选择最小可容纳 batch/block bucket

DecodeGraphManager.capture
  -> DecodeGraphBucket 创建固定 GPU buffers
  -> warmup
  -> torch.cuda.graph capture

DecodeGraphBucket.replay
  -> copy input_ids / positions / block_tables 内容
  -> 填充 padding slot
  -> graph.replay()
  -> 返回 active logits
```

固定 buffer 包括：

```text
input_ids     [bucket_batch]
positions     [bucket_batch]
block_tables  [bucket_batch, max_blocks]
logits        [bucket_batch, vocab]
```

Graph 外的 CPU scheduler 仍负责请求进入/退出、BlockManager、Prefix Cache 和 slot 选择。Graph 内只执行固定形状的 decode forward。

## 4. 为什么能加速

普通路径每个 token 都需要 Python 调度和多个 kernel launch；Graph replay 只更新固定 GPU buffer 内容，再一次性重放已捕获的 kernel 序列。它主要降低小 batch decode 的 CPU launch/dispatcher 空洞，不改变单 kernel 的算法复杂度。

## 5. 当前边界

- 本阶段完成 bucket、固定 buffer、padding slot 和 replay 基础；
- 尚未将真实 `NativeBatchRuntime.decode_step` 切换到 Graph manager；
- 尚未处理完整模型中所有动态分配/metadata 的 graph-capture 约束；
- inactive slot 的 scratch KV page 需要由上层显式提供；
- 尚未报告端到端 TPOT 收益。

## 6. 投机解码计划

投机解码正式进入阶段 17，顺序为：

1. greedy 多 token causal 验证；
2. 接受连续前缀、拒绝位置 KV 回退；
3. prompt lookup/小 draft model；
4. confidence-scheduled draft length；
5. 随机采样接受概率和校正。

## 7. 测试

CPU 测试覆盖 bucket 选择和非法配置；CUDA 测试覆盖固定 buffer replay 与 active_count padding。完整模型 graph 接入和 TPOT benchmark 留在下一步。
