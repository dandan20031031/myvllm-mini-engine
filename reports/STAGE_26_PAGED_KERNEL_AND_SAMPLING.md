# 阶段26：PagedAttention向量化、CUDA greedy与可靠构建

## 实现与调用链

`NativeQwen3.forward_decode_batch → kernels/paged.py → bindings.cpp → paged_attention.cu` 仍直接读取 `[physical_block, token, kv_head, dim]`。D=128使用新kernel，其余head_dim保留原路径。

| 文件 | 代码与作用 |
|---|---|
| `kernels/csrc/paged_attention.cu` | `Pack4`将4个FP16/BF16元素作为对齐64-bit访问；Q/K点积由warp归约；4个warp分担不同token的V累加，再在shared memory合并；保留原scalar kernel和partition log-sum-exp归并 |
| `kernels/paged.py` | `cuda`使用D128优化，`cuda_baseline`为实验保留原实现；不增加dense gather，不改变GQA物理布局 |
| `kernels/csrc/sampling.cu` | 两级argmax：每4096词一块生成局部最大值/index，然后合并；ties、全负无穷、NaN选最小index，与torch.argmax相同 |
| `sampling.py` | 可选`backend="cuda"`，只支持temperature=0；默认torch仍支持原有采样，不把CUDA贪心实现称为top-p CUDA采样 |
| `kernels/csrc/prefill.cu` | 自有CUDA支持rectangular Q/K与query_start；绝对位置causal mask用于chunk和投机验证，K/V保持原始GQA头数 |
| `kernels/csrc/bindings.cpp` | 检查形状/设备/dtype，暴露baseline、greedy和offset prefill；所有4个旧CUDA文件恢复`C10_CUDA_KERNEL_LAUNCH_CHECK()` |
| `kernels/cuda_extension.py` | 源文件、PyTorch、Toolkit、目标设备等生成构建hash，DLL/module与目录随hash区分；避免Windows旧chat进程锁住`.pyd`时覆盖失败、继而加载旧接口 |
| `tests/test_cuda_sampling.py` | dtype、vocab尾部、tie/NaN/Inf、空batch、非法参数、Graph replay |
| `tests/test_cuda_paged_attention.py` | 旧16种dim/dtype对拍继续保留；新增5种block size × 2dtype的baseline和Graph对拍 |
| `benchmarks/benchmark_paged_sweep.py` | 同输入baseline/新kernel配对、交替执行顺序、Graph重复20次后用CUDA Event计时，保留全部原始样本 |

恢复launch检查只查询CUDA错误状态，不对GPU做同步；已经在Graph capture/replay测试中通过。不能用空检查掩盖kernel失败。

## 测试与实测

RTX5080，PyTorch2.9.1+cu128，Toolkit13.3。专项CUDA测试55 passed；阶段23–26整合后全量302 passed、2 warnings（Toolkit版本差异和MSVC私有接口）。真实模型验收另见系统验证摘要。

FP16、Hq=16/Hkv=8/D128，block16的配对测量：

| Batch / context | 原kernel Graph均摊中位数 | 新kernel Graph均摊中位数 | 配对比值 |
|---|---:|---:|---:|
| 1 / 128 | 33.99 us | 19.34 us | 1.76x |
| 8 / 512 | 72.38 us | 41.50 us | 1.74x |
| 32 / 2048 | 713.76 us | 400.30 us | 1.78x |

block16/32/64/128/256共15组均通过reference对拍，本次比值范围1.74–1.80x；FP16最大绝对误差不超过0.0002442。这是本机配对实验，未控制所有外部负载，原始文件保存GPU状态与扩展binary hash。**不是端到端服务加速、不是相对MiniVLLM/vLLM的结果，也不应与早期eager Event口径混比。** 当前不据此改变默认block size。

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\benchmark_paged_sweep.py `
  --block-sizes 16 32 64 128 256 --batches 1 8 32 --contexts 128 512 2048 `
  --dtype fp16 --output benchmarks\results\stage26_paged_sweep.json
```

## 边界

- 这是现有split-KV策略的向量化实现，不是新提出Flash-Decoding，也不使用Tensor Core attention。
- greedy kernel已验证正确性，尚无稳定的端到端优势证据；保持显式可选。
- suffix prefill仍会gather分页KV，correctness kernel最长8192；以后需要直接paged多query验证或更高效Flash后端。
- 暂未完成compute-sanitizer memcheck/racecheck；数值和Graph测试不能代替这些工具。
- 内容hash隔离构建不负责删除旧目录，避免删除用户仍在使用的DLL。
