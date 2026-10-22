# 阶段 18：Projection Fusion

## 目标

根据 MiniVLLM RTX 5080 profiler，decode 的主要 GPU 时间集中在小 batch GEMM。Qwen3 每层原来分别发射 Q、K、V 三个 projection，以及 Gate、Up 两个 MLP projection。本阶段先减少 GEMM 和 dispatcher 次数，保留原始路径用于对照。

## 实现

新增 `FusedLinear`：

```text
input [rows, hidden]
    -> one mm(input, concat(weight_q, weight_k, weight_v).T)
    -> split [Q, K, V]
```

MLP 使用同样的方式合并 Gate/Up：

```text
input
    -> one mm(input, concat(weight_gate, weight_up).T)
    -> split gate/up
    -> SiLU(gate) * up
    -> down projection
```

`NativeQwen3.enable_projection_fusion()` 会为每一层启用两种融合。CLI 和 native benchmark 使用 `--fuse-projections` 开启。默认路径不变，量化 reference 路径暂不融合，避免破坏其 packed weight 和误差报告。

## 调用逻辑

```text
NativeQwen3Runner.from_pretrained(..., fuse_projections=True)
  -> NativeQwen3.enable_projection_fusion()
  -> Attention.fuse_qkv()
  -> MLP.fuse_gate_up()
  -> forward / forward_decode_batch
  -> FusedLinear.forward
  -> torch.mm / CUDA cuBLAS dispatch
```

这是 projection-level fusion，不是从零编写的 Tensor Core GEMM；底层 CUDA 矩阵乘仍由 PyTorch/cuBLAS/cuBLASLt 选择。

## 正确性

测试覆盖：

- FusedLinear 与多个普通 Linear 的拼接输出对齐；
- native Qwen3 prefill 对齐；
- native Qwen3 多步 decode 对齐；
- 重复启用融合不改变模块状态；
- 已量化模型明确拒绝融合；
- CUDA/torch backend 的既有线性和模型回归。

专项回归：

```text
17 passed, 2 warnings
```

RTX 5080、Qwen3-0.6B、BF16、全 CUDA backend、单请求、warmup=2、repetitions=7、max_new_tokens=32 的初步结果如下。两组使用相同 prompt 和计时脚本：

| prompt | 未融合 TPOT | 融合 TPOT | 变化 |
|---|---:|---:|---:|
| KV Cache | 15.24 ms/token | 14.68 ms/token | 约 3.7% |
| continuous batching | 15.91 ms/token | 14.53 ms/token | 约 8.6% |

这是一组单请求结果，仍有 GPU/Windows 调度抖动，不能代替 batch Graph benchmark。它说明融合路径已经能在完整模型上运行并有小幅收益；主要收益预期仍在 batch decode 和 CUDA Graph 场景。

## 限制

- 当前仍调用 PyTorch `torch.mm`，尚未实现专用量化 GEMM；
- QKV 合并后 Q/K RMSNorm、RoPE 和 KV 写入仍是独立算子；
- Gate/Up 合并后 SiLU×Up 仍单独执行；
- 量化 projection 仍保持 reference dequant 路径；
- 真实收益需要在 batch decode 和 CUDA Graph 路径中测量。

## 下一步

阶段 19 将真实 `NativeBatchRuntime` 接入 benchmark，比较 eager 与 CUDA Graph，并记录 projection fusion 的增益。

正式 benchmark JSON：

- `benchmarks/results/native_qwen3_unfused_stage18.json`
- `benchmarks/results/native_qwen3_fused_stage18.json`
