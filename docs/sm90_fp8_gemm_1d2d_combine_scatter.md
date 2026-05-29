# SM90 FP8 grouped GEMM 的 combine-scatter epilogue 设计与评估

本文档说明在
`sm90_fp8_gemm_1d2d_impl` 的 m-grouped contiguous 路径上加入
**combine-scatter epilogue** 的背景、语义契约、实现细节和性能评估。

相关前置文档：

- [`sm90_fp8_gemm_1d2d_gather_index.md`](./sm90_fp8_gemm_1d2d_gather_index.md)
- [`sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md`](./sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md)

本文讨论当前已经实现的 **GEMM 输出直接 scatter 到 source-rank combine buffer，
再在 source rank 上 local reduction 时乘 top-k score** 这一段。

---

## 1. 背景

### 1.1 当前 all-gather + grouped GEMM 流程

在 EP MoE 场景里，每个 rank 持有一部分 token。为了让本 rank 上的 local
experts 计算来自所有 rank 的 token，需要先把 activation all-gather 成
`A_pool`：

```text
rank-local A  ──[all-gather]──►  A_pool[num_ranks * tokens_per_rank, hidden]
                                      │
                                      ▼
                         grouped FP8 GEMM / expert compute
                                      │
                                      ▼
                                local output D
```

已有的 `gather_index` 路径把 token routing 映射直接下推到 GEMM kernel 里：

```text
logical row m  ──gather_index[m]──►  A_pool[source_token]
logical row m  ──grouped_layout[m]──► local expert id
```

因此每个 output row 对应：

```text
D[m, :] = A_pool[gather_index[m], :] @ W[grouped_layout[m], :].T
```

### 1.2 fc2 + combine 的目标形态

MoE 的 fc2 之后，expert 输出需要回到 token 的 source rank，并写入该 token
对应的 top-k slot。抽象地说，目标 buffer 是：

```text
combine_buffer[src_rank][local_token, topk_slot, n]
```

当前假设每个 rank 在 all-gather 阶段已经拿到了完整的 top-k score。expert 输出先以
raw BF16 GEMM 结果写回 source rank，后续 source rank 在 slot 维度 reduction 时乘上
对应 score：

```text
out[token, :] = sum_j combine_buffer[token, j, :] * topk_scores[token, j]
```

当前 patch 完成的是：

```text
GEMM output row m
  └──> source token = combine_src_index[m]
  └──> source rank  = source token / tokens_per_rank
  └──> local token  = source token % tokens_per_rank
  └──> topk slot    = row_to_topk[m]
  └──> P2P store output to combine_buffer_ptrs[source_rank][local_token, topk_slot, :]
```

local reduction 由后续 `combine_reduce_slots` kernel 完成，并在该 kernel 中读取
source rank 本地 token 对应的 top-k score。

---

## 2. 动机

### 2.1 two-stage baseline 的问题

最直接的实现是两段式：

```text
1. grouped GEMM:
     D[m, n] 通过原有 TMA store 写回本 rank HBM

2. scatter-copy kernel:
     读取 D[m, :]
     根据 combine_src_index[m]、row_to_topk[m]
     写到 source rank 的 combine_buffer[token, slot, :]

3. local reduction kernel:
     读取本 rank 的 combine_buffer[token, slot, :]
     乘 topk_scores[token, slot]
     对 slot 维度求和
```

这条路径的问题有三个：

1. **多一次 D 的全量 HBM 写读**：GEMM 先把 D 写到 local HBM，scatter kernel
   再把 D 读出来。
2. **多一个 kernel launch**：scatter 是独立 kernel，无法和 GEMM epilogue 融合。
3. **P2P store 单独发生**：peer write 的延迟和带宽压力完全暴露在 GEMM 之后。

combine-scatter epilogue 的目标是把第 2 步融合进 GEMM epilogue：

```text
grouped GEMM accumulator
  └──> epilogue 直接 P2P store 到 source rank combine_buffer
```

这样可以避免物化本地 D，并让部分 peer store 成本与 GEMM 的尾部执行重叠。

### 2.2 为什么不是直接复用 TMA store

原有 epilogue 写回本地 D 时走 TMA store，目标地址是规则的 row-major
`D[m, n]`。combine-scatter 的目标地址由每一行的 source token 和 top-k slot
决定：

```text
dst_base(row) = combine_buffer_ptrs[src_rank]
              + ((local_token * top_k + topk_slot) * N)
```

不同 row 的 `dst_base` 不再是普通 TMA descriptor 能描述的规则矩阵切片，
因此当前实现不能直接用 TMA store 写最终 combine slot，只能改成 thread-level
global store。

---

## 3. 语义与接口契约

### 3.1 GEMM API 扩展

`m_grouped_fp8_fp4_gemm_nt_contiguous` 新增 combine-scatter 相关 optional 参数：

```c++
combine_src_index:        Optional[Tensor[int32]]
combine_row_topk:         Optional[Tensor[int32]]
combine_buffer_ptrs:      Optional[Tensor[int64]]
combine_scale_ptrs:       Optional[Tensor[int64]]
combine_tokens_per_rank:  Optional[int]
combine_top_k:            Optional[int]
combine_scatter_direct_accum_stg: bool = false
combine_scatter_fp8:      bool = false
```

当 `combine_row_topk` 非空时，GEMM 进入 combine-scatter epilogue；否则保持原有
TMA store D 的行为。`combine_src_index` 可不传；这种情况下实现会复用
`gather_index` 作为 source-token map，以兼容原来的 GEMM1-style 调用。

当 `combine_scatter_fp8=true` 时，GEMM epilogue 不再把 raw BF16 结果写入
combine buffer，而是把结果量化为 E4M3 FP8，并额外写出每行每 32 个 N 元素一个
FP32 scale。source rank 上的 local reduction 会读取 FP8 data 和 scale，反量化后再
乘 top-k score。

### 3.2 输入含义

| 参数 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `gather_index` | optional `(M,) int32 CUDA contiguous` | output row 到 A/SFA source row 的映射；为空时 A 已按 grouped row 排好 |
| `combine_src_index` | optional `(M,) int32 CUDA contiguous` | output row 到 global source token 的映射；为空时回退使用 `gather_index` |
| `combine_row_topk` | `(M,) int32 CUDA contiguous` | output row 最终写入的 top-k slot |
| `combine_buffer_ptrs` | `(num_ranks,) int64 CUDA contiguous` | 每个 source rank 的 combine buffer device pointer |
| `combine_scale_ptrs` | optional `(num_ranks,) int64 CUDA contiguous` | FP8 combine-scatter scale buffer device pointer |
| `combine_tokens_per_rank` | scalar int | 每个 rank 的 token 数 |
| `combine_top_k` | scalar int | combine buffer 的 top-k 槽数 |

`combine_buffer_ptrs[r]` 指向 rank `r` 暴露给本 rank 写入的 buffer，逻辑形状为：

```text
[tokens_per_rank, combine_top_k, N]
```

`combine_scatter_fp8=true` 时，`combine_buffer_ptrs[r]` 指向 FP8 buffer，逻辑形状仍为
`[tokens_per_rank, combine_top_k, N]`；`combine_scale_ptrs[r]` 指向 FP32 scale buffer，
逻辑形状为 `[tokens_per_rank, combine_top_k, N / 32]`。

device 侧写入位置：

```c++
src_token   = combine_src_index[row]
src_rank    = src_token / combine_tokens_per_rank
local_token = src_token - src_rank * combine_tokens_per_rank
topk_slot   = combine_row_topk[row]

dst = combine_buffer_ptrs[src_rank]
    + (local_token * combine_top_k + topk_slot) * N

dst[:] = bf16(gemm_output[row, :])
```

score 不在 GEMM epilogue 中处理。source rank 上的 local reduction 使用本地 token
对应的 `topk_scores[tokens_per_rank, combine_top_k]`，执行：

```text
out[token, col] = sum_slot bf16(combine_buffer[token, slot, col]) * topk_scores[token, slot]
```

### 3.3 约束

当前实现有如下约束：

- 只覆盖 SM90 FP8 1D2D m-grouped contiguous 路径。
- 默认 combine buffer dtype 为 BF16；`combine_scatter_fp8=true` 时为 E4M3 FP8。
- `combine-scatter` 必须提供 `combine_src_index` 或 `gather_index` 中的一个。
- `combine_buffer_ptrs.numel() <= 8`。
- BF16 combine-scatter 要求 `N % 8 == 0`，因为 epilogue 使用 16B store（8 个 BF16）。
- FP8 direct combine-scatter 要求 `N % 64 == 0`，因为它使用 N64 B 重排和 16B store
  （16 个 FP8）。
- multi-rank benchmark 中，`combine-scatter` 只允许 `all-ranks-local` routing。

最后一条是语义约束，不是实现细节。`random` routing 下多个 rank 可能写同一个
`[token, topk_slot]`，会产生数据竞争；`all-ranks-local` 通过
`topk_slot_offset = rank * local_top_k` 保证不同 rank 写 disjoint top-k slots。

---

## 4. 实现细节

### 4.1 JIT 入口

host runtime 里用两个指针状态作为编译期模板参数：

```c++
args.gather_index != nullptr      // -> kGatherA
args.combine_row_topk != nullptr  // -> kCombineScatter
```

这意味着：

- `kGatherA` 控制 producer WG 是否用 `gather_index` 间接读取 A/SFA。
- 不带 combine-scatter 的 kernel 不会携带额外 epilogue 分支。
- 带 combine-scatter 的 kernel 在 epilogue 中不再走 TMA store D，而是走
  scatter store。

这个拆分用于 GEMM2-style 路径：GEMM1 已经把 token 按 grouped row 连续排布，
后续 GEMM2 的 A/SFA 可以直接按 logical row 读取，但 combine-scatter 仍然需要知道
每一行最终属于哪个 source token。因此此时传入 `gather_index = None`、
`combine_src_index = row -> source token`。

### 4.2 epilogue 目标地址计算

kernel 端在每个 tile 开始时先为每个 logical row 计算目标 row base，并写入
shared memory。后面的 scatter loop 只读取预计算好的 base address，避免在 epilogue
store 热路径里反复做 source-token / top-k / peer-buffer 查表。

```c++
for (uint32_t row = threadIdx.x; row < BLOCK_M; row += kNumMathThreads) {
    const uint32_t logical_m = base_m_idx + row;
    uint64_t dst_base_u64 = 0;
    if (logical_m >= shape_m) {
        s_combine_scatter_base[row] = 0;
        continue;
    }

    const int src_token_i = __ldg(combine_src_index + logical_m);
    const int topk_i = __ldg(combine_row_topk + logical_m);
    if (src_token_i >= 0 and topk_i >= 0 and
        combine_tokens_per_rank != 0 and
        static_cast<uint32_t>(topk_i) < combine_top_k) {
        const uint32_t src_token = static_cast<uint32_t>(src_token_i);
        const uint32_t src_rank = src_token / combine_tokens_per_rank;
        if (src_rank < num_ranks) {
            const uint32_t local_token = src_token - src_rank * combine_tokens_per_rank;
            const uint64_t peer_base_u64 = __ldg(combine_buffer_ptrs + src_rank);
            const uint64_t dst_row_offset =
                (static_cast<uint64_t>(local_token) * combine_top_k +
                 static_cast<uint32_t>(topk_i)) * shape_n;
            dst_base_u64 = peer_base_u64 + dst_row_offset * sizeof(nv_bfloat16);
        }
    }
    s_combine_scatter_base[row] = dst_base_u64;
}
```

pad row 或非法 top-k slot 会写入 0，后续 scatter store 把它解释为 `nullptr`
并直接跳过。

`combine_buffer_ptrs` 通过 CUDA IPC 暴露给 peer rank。这里需要保留 PyTorch caching
allocator 的 base allocation offset：IPC handle 对应的是 allocation base，而 tensor
的 `data_ptr()` 可能是该 allocation 内的子偏移。如果 open 侧不把 offset 加回去，
writer rank 通过自己的 IPC mapping 读写会自洽，但 owner rank 用本地 tensor 指针做
local reduction 时会看不到 remote slot。

### 4.3 三种 scatter epilogue

默认 scatter epilogue 仍保留 staged 路径：

1. 用 STSM 把 WGMMA accumulator fragment 落到 row-major shared-memory tile。
2. 所有 store threads 从 shared memory 读出连续 16B chunk，用 `uint4` 写到目标
   combine buffer。

对应代码结构：

```c++
// accumulator -> row-major smem_d
ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(..., smem_ptr);
cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

// smem_d -> peer combine buffer, 8 BF16 / 16B per store
constexpr uint32_t kScatterVecElems = 8;
constexpr uint32_t kVecsPerRow = BLOCK_N / kScatterVecElems;

for (uint32_t linear = threadIdx.x; linear < BLOCK_M * kVecsPerRow;
     linear += kNumWGMMAStoreThreads) {
    row = linear / kVecsPerRow;
    vec = linear - row * kVecsPerRow;
    dst_base = reinterpret_cast<nv_bfloat16*>(s_combine_scatter_base[row]);
    scatter_vec(dst_base, row, vec);
}
```

`scatter_vec` 内部按 8 个 BF16 一组做 raw 16B 写：

```c++
uint4 packed;
auto* packed_bf16 = reinterpret_cast<nv_bfloat16*>(&packed);

#pragma unroll
for (uint32_t elem = 0; elem < 8; ++elem)
    packed_bf16[elem] = src[elem];

*reinterpret_cast<uint4*>(dst_base + dst_col) = packed;
```

当 `shape_n` 尾部不足 8 个 BF16 时，会退化成 element-wise tail store。不过 host
侧当前要求 `N % 8 == 0`，正常 benchmark 不会走 tail。

BF16 direct epilogue 由 `combine_scatter_direct_accum_stg=true` 打开。它要求调用侧
在 FP8 cast 之前按每 32 个 N 做一次 B 重排，使 WGMMA accumulator owner lanes 在
同一条 store 指令里写相邻 16B segment：

```text
lane0 -> n +  0 .. n +  7
lane1 -> n +  8 .. n + 15
lane2 -> n + 16 .. n + 23
lane3 -> n + 24 .. n + 31
```

benchmark 中该重排由 `_make_wgmma_n32_physical_to_logical_index` 完成。N32 direct
path 直接把 accumulator pack 成 8 个 BF16，并用 `st.global.v4.u32` 写到 peer
combine buffer，跳过 STSM 和 shared-memory reload。

FP8 direct epilogue 由 `combine_scatter_direct_accum_stg=true` 和
`combine_scatter_fp8=true` 同时打开。它使用 N64 B 重排，使每个 lane 能在同一行拿到
16 个连续 FP8 输出元素，并恢复 16B peer store：

```text
lane0 -> n +  0 .. n + 15
lane1 -> n + 16 .. n + 31
lane2 -> n + 32 .. n + 47
lane3 -> n + 48 .. n + 63
```

benchmark 中该重排由 `_make_wgmma_n64_physical_to_logical_index` 完成。FP8 path 对
每行每 32 个 N 元素生成一个 FP32 scale，因此每个 lane 写 16 个 FP8 元素时，会和
相邻 lane 做一次 2-lane amax reduction，共同覆盖 32 个元素的 scale group：

```text
lanes 0..1 -> scale group 0, cols n +  0 .. n + 31
lanes 2..3 -> scale group 1, cols n + 32 .. n + 63
```

这里的 `__shfl_xor_sync` 使用精确的 2-lane mask，而不是 full-warp mask。pad row 或
非法 row 会在 store lambda 里提前跳过，如果 full-warp mask 包含未参与的 lane，会造成
未定义同步行为并可能让多 rank benchmark 卡住。

### 4.4 与原 TMA store 路径的关系

combine-scatter 分支结束后直接 `continue`：

```c++
if constexpr (kCombineScatter) {
    ...
    continue;
}

// 原有 TMA store D 路径
```

因此带 combine-scatter 时不会再写本地 `D[m, n]`。`D` 在 API 中仍然保留，是为了
复用原有 grouped GEMM 入口和 descriptor 构造；语义上它不再是有效输出。

### 4.5 row_to_topk 来自哪里

`row_to_topk` 由 `build_gather_layout_for_rank_overlap` 生成。它和 source-token
map 一一对应；当前 benchmark 中这个 map 叫 `combine_src_index`：

```text
logical row m:
    combine_src_index[m] -> source token
    grouped_layout[m] -> local expert
    row_to_topk[m] -> final top-k slot
```

在 `all-ranks-local` routing 里，每个 rank 只负责本 rank 的 local experts，
但 final combine buffer 的 top-k slot 是 global top-k slot。因此构造 gather
layout 时会传入：

```python
topk_slot_offset = rank * local_top_k
```

这样 rank `r` 写 `[r * local_top_k, (r + 1) * local_top_k)` 这一段 top-k slots，
多个 rank 不会互相覆盖。

---

## 5. 辅助 kernel 与 benchmark

### 5.1 correctness checker

`check_combine_scatter_output` 用于验证 fused epilogue 或 standalone scatter-copy
是否把 raw `D[row, :]` 写到了正确位置。

输入：

```text
d_ref                 // 非 scatter GEMM 写出的本地参考 D
combine_src_index
row_to_topk
combine_buffer_ptrs
tokens_per_rank
top_k
```

checker 按每个 `(row, col)` 重新计算目标地址，并比较：

```text
bf16(d_ref[row, col])
  == combine_buffer[src_rank][local_token, topk_slot, col]
```

跨 rank 用 `all_reduce(max)` 汇总 `max_abs_diff`，用 `all_reduce(sum)` 汇总
`mismatch_count`。

benchmark 在 reduce-scatter correctness 路径里还会额外检查 owner rank 本地
`combine_buffer` 是否全为 finite。这个检查覆盖了 writer-side IPC mapping checker
无法单独发现的 IPC offset / owner-read 问题。

此外，`--check` 默认会为每个 rank 的前 32 个 local tokens 构造 CPU arithmetic
reference。每个 rank 在 CPU 上用普通 GEMM 输出 `D`、`combine_src_index`、
`row_to_topk` 和 `topk_scores` 计算：

```text
partial[dst_rank, local_token, col] += bf16(D[row, col]) * topk_scores[src_token, topk_slot]
```

随后把 partial 搬回 GPU 做一次跨 rank `all_reduce(SUM)`，得到每个 source rank 的
CPU reference output，并分别校验：

- fused scatter + local reduction
- GEMM + pack/local-reduce + NCCL reduce-scatter baseline

`--cpu-ref-tokens` 控制 CPU reference 覆盖的 local token 数：默认 32，`-1` 表示全量，
`0` 表示关闭。

### 5.2 standalone scatter-copy baseline

`combine_scatter_copy_rows` 是 two-stage baseline 的第 2 段：

```text
D[m, n] -> peer combine_buffer[token, topk_slot, n]
```

它不参与正式 fused 路径，只用于衡量：

- 如果 GEMM 仍然 TMA store 到本地 D；
- 再单独启动一个 scatter-copy kernel；
- 那么相对 fused epilogue 会慢多少。

该 kernel 每个 CTA 处理 `rows_per_block` 行，每行按 8 个 BF16 一组用 16B `uint4`
store。

### 5.3 reduce-scatter baseline 与 local reduction

为了和更接近完整 combine 的路径对比，benchmark 还加入了两条辅助路径：

```text
baseline:
  GEMM(no scatter)
    -> pack/local-reduce 到 rs_input[num_ranks, tokens_per_rank, N]
    -> NCCL reduce_scatter(SUM)

fused:
  fused raw combine-scatter GEMM
    -> source rank scored local reduction over top-k slots
```

`combine_pack_for_reduce_scatter` 从普通 GEMM 的本地 `D[M, N]` 读取每个 grouped row，
按 `combine_src_index[row]` 找到 source rank/token，按 `row_to_topk[row]` 取 score，
把 `bf16(D[row, col]) * score` atomic add 到 float32 `rs_input[src_rank, token, col]`。
随后 `nccl_reduce_scatter_sum` 对 float32 `rs_input` 做 SUM，每个 source rank 收到
自己的 `[tokens_per_rank, N]`。

`combine_reduce_slots` 则直接把 fused scatter 已经写到本 rank 的：

```text
combine_buffer[tokens_per_rank, combine_top_k, N]
```

以及本 rank local token 对应的 `topk_scores[tokens_per_rank, combine_top_k]`，做：

```text
out[token, col] = sum_slot combine_buffer[token, slot, col] * topk_scores[token, slot]
```

当前实现把 grid 拆成 `(token, N tile)`，并把 `top_k` 和每线程处理列数作为 JIT
模板参数。这样可以避免原先 per-element 线性索引里的除法/取模，并且每个 token
的每个 column tile 只把 top-k score 读到 shared memory 一次，再供该 tile 内线程复用。

FP8 combine-scatter 对应 `combine_reduce_slots_fp8`。它读取：

```text
combine_buffer_fp8[tokens_per_rank, combine_top_k, N]
combine_scales_fp32[tokens_per_rank, combine_top_k, N / 32]
```

然后在 local reduction 中做：

```text
value = fp8_to_float(combine_buffer_fp8[token, slot, col])
      * combine_scales_fp32[token, slot, col / 32]
out[token, col] += value * topk_scores[token, slot]
```

两条路径在数学上等价，但 reduction 顺序不同，所以 correctness 用 tolerance，而不是
要求 bit-exact。

当前 pack baseline 是为了建立对比闭环，尚未优化：它使用 per-element float
`atomicAdd`，性能不是最终形态。

### 5.4 benchmark 入口

主 benchmark 脚本：

```bash
tests/bench_combine_scatter.py
```

关键参数：

```bash
--bench-standalone-scatter    # 同时跑 standalone scatter-copy 和 two-stage baseline
--bench-reduce-scatter        # 同时跑 GEMM + pack/local-reduce + NCCL reduce-scatter baseline
--no-gather-a                 # GEMM2-style：A/SFA 已按 grouped row 连续排布
--combine-scatter-direct-accum-stg
                              # BF16 使用 N32，FP8 使用 N64 direct accumulator epilogue
--combine-scatter-fp8         # 使用 FP8 combine buffer + FP32 scales
--combine-scatter-local-buffer
                              # 诊断用：把 peer destination 重定向到本地 HBM
--fp8-debug-precision         # 配合 --check --combine-scatter-fp8 打印 FP8 误差来源诊断
--scatter-rows-per-block      # standalone scatter-copy 的 rows/CTA，默认 4
--compact-layout-order        # compact GEMM2 row order：token 或 ring，默认 token
```

该脚本只测 combine 相关路径，不把 all-gather 放进 timed path。每个 rank 用相同
seed 直接构造完整 `A_pool`，从而避免 all-gather overlap 逻辑干扰 combine 方案对比。
multi-rank 下脚本固定使用 `all-ranks-local` routing，保证不同 rank 写 disjoint top-k
slots，避免 random routing 下的写冲突。

---

## 6. 性能评估

### 6.1 测试环境与命令

测试环境：

- 8 卡 H200
- `NCCL_MIN_CTAS=64 NCCL_MAX_CTAS=64`
- routing 使用 `all-ranks-local`

当前目标 shape：

```text
num_ranks = 8
global source tokens = 55808
source tokens per rank = 55808 / 8 = 6976
experts per token on each rank = 2
global top-k per token = 8 * 2 = 16
local routed rows per rank before expert padding = 55808 * 2 = 111616
global routed rows before expert padding = 55808 * 16 = 892928
intermediate hidden = 1280
hidden = 2048
```

这里的 `55808` 是 8 个 rank 合计的原始 source token 数，不是每 rank token 数，也不是
route 后的 GEMM M。因为 `all-ranks-local` routing 会让每个 source token 在每个 rank
上命中 2 个 local experts，所以每个 rank 实际执行的 fc2 grouped GEMM 行数是
`55808 * 2 = 111616`。

GEMM2 benchmark 默认使用 compact GEMM2 layout：按 local expert 聚合 row，不再按
source rank 切 chunk，也不再使用 allgather-overlap 所需的 per-rank tile padding。
当前 DeepGEMM contiguous psum layout 仍要求 expert 起点按 128 对齐，所以
benchmark 输出里的 `m_logical` 会从真实 row 数 `111616` 增加到 `115456`。旧的
rank-padded overlap layout 会得到 `m_logical=131328`，可用
`--rank-padded-gemm2-layout` 复现。

compact GEMM2 layout 的 row order 可以通过 `--compact-layout-order` 控制：

- `token`：默认模式。每个 expert 内按 global source token 顺序排列 row。
- `ring`：每个 expert 内按当前 rank 起点的 source-rank ring order 排列 row，即
  `rank, rank + 1, ...`。该模式用于检查目的 rank 写入热点是否会明显影响
  combine-scatter epilogue。

对应到 benchmark 参数：

```text
tokens_per_rank = 55808 / 8 = 6976
top_k           = 16     # 8 ranks * 2 local experts per rank
hidden           = 1280   # GEMM K, fc2 input/intermediate dimension
n                = 2048   # GEMM N, fc2 output/hidden dimension
```

默认 staged epilogue 命令：

```bash
NCCL_MIN_CTAS=64 NCCL_MAX_CTAS=64 \
python3 tests/bench_combine_scatter.py \
  --num-local-ranks 8 \
  --tokens-per-rank 6976 \
  --hidden 1280 \
  --n 2048 \
  --top-k 16 \
  --global-num-experts 512 \
  --experts-per-rank-token 2 \
  --no-gather-a \
  --warmups 2 \
  --iters 5
```

BF16 N32 direct-accumulator epilogue 命令：

```bash
NCCL_MIN_CTAS=64 NCCL_MAX_CTAS=64 \
python3 tests/bench_combine_scatter.py \
  --num-local-ranks 8 \
  --tokens-per-rank 6976 \
  --hidden 1280 \
  --n 2048 \
  --top-k 16 \
  --global-num-experts 512 \
  --experts-per-rank-token 2 \
  --no-gather-a \
  --combine-scatter-direct-accum-stg \
  --warmups 2 \
  --iters 5
```

FP8 N64 direct-accumulator epilogue 命令：

```bash
NCCL_MIN_CTAS=64 NCCL_MAX_CTAS=64 \
python3 tests/bench_combine_scatter.py \
  --num-local-ranks 8 \
  --tokens-per-rank 6976 \
  --hidden 1280 \
  --n 2048 \
  --top-k 16 \
  --global-num-experts 512 \
  --experts-per-rank-token 2 \
  --no-gather-a \
  --combine-scatter-direct-accum-stg \
  --combine-scatter-fp8 \
  --warmups 5 \
  --iters 20
```

### 6.2 结果

形状：

```text
ranks = 8
tokens/rank = 6976
hidden = 1280
global_top_k = 16
local_top_k = 2
m_logical = 115456
n = 2048
gather_a = false
layout = compact-gemm2/token
```

结果按方案拆分如下。表里 `total` 是直接测整条路径的 median，不是把各组件 median
简单相加。除非特别说明，括号内为 CUDA event time。

| 方案 | GEMM no scatter | fused GEMM + scatter | local reduction | total fused + local reduction |
| --- | ---: | ---: | ---: | ---: |
| BF16 staged：STSM + smem reload + 16B peer STG | 783.89 us (732.29 us) | 1643.71 us (1587.74 us) | 206.62 us (164.38 us) | 1841.87 us |
| BF16 N32 direct accumulator STG | 785.14 us (801.94 us) | 1568.89 us (1492.59 us) | 203.51 us (162.82 us) | 1753.26 us |
| FP8 old direct，8B peer store（中间实验） | 839.56 us (765.97 us) | 2111.98 us (2031.46 us) | 273.01 us (203.57 us) | 2328.06 us |
| FP8 N64 direct，16B peer store | 784.41 us (758.13 us) | 1245.91 us (1188.58 us) | 227.33 us (185.49 us) | 1463.07 us |

correctness：

```text
Fused combine-scatter epilogue value check:
max_abs_diff = 0
mismatches   = 0

CPU reference check for fused scatter + local reduction:
max_abs_diff = 9.15527e-05
tokens       = 4
mismatches   = 0
```

上面的 correctness 是 BF16 path。FP8 path 会引入量化误差，不能和 BF16 CPU
reference 做 bit-exact 或严格 tolerance 对比。当前 benchmark 增加了
`--fp8-debug-precision`，用于把 FP8 误差拆成 slot 写入、量化本身和 local reduction
三个层面。该诊断会在 tolerance check 之前打印；如果希望 FP8 `--check` 整体通过，需要
显式设置符合预期误差范围的 `--reduce-check-rtol/--reduce-check-atol`。

目标 shape 下，`--check --combine-scatter-fp8 --fp8-debug-precision` 的典型输出：

| 对比项 | max abs | mean abs | RMSE | 结论 |
| --- | ---: | ---: | ---: | --- |
| local-reduce kernel vs torch reduce(actual slots) | 7.63e-05 | 5.66e-06 | 8.37e-06 | local reduction 实现正确 |
| CPU quantized slots vs CPU BF16 slots | 5.14 | 0.584 | 0.857 | FP8 per-32 量化本身的误差 |
| actual dequant slots vs CPU BF16 slots | 5.20 | 0.589 | 0.859 | 和 CPU 量化误差基本一致 |
| actual reduction vs CPU BF16 reference | 10.09 | 1.603 | 2.038 | reduction 后的端到端 FP8 误差 |

因此当前证据表明，FP8 path 的主要数值差异来自量化本身，而不是 slot layout、peer
store 或 local reduction 的实现错误。`actual dequant slots vs CPU quantized slots`
仍有残差，原因是 kernel 从 FP32 accumulator 直接量化，而 CPU diagnostic 用普通
GEMM 输出 `D` 作为参考，中间包含 BF16 写回/重排和 PyTorch CPU FP8 conversion 的差异。

### 6.3 解读：BF16、FP8 与 store 粒度

N32 direct epilogue 相比 staged epilogue：

```text
fused GEMM + raw scatter:
  staged  = 1643.71 us wall / 1587.74 us event
  N32 dir = 1545.07 us wall / 1492.58 us event
```

N32 direct epilogue 省掉约 `99 us` wall / `95 us` event。它证明绕开
`accumulator -> STSM -> shared memory reload` 这条路径有收益，但收益不是数量级变化。

与不做 scatter 的 GEMM 相比，N32 direct 仍然有明显额外成本：

```text
GEMM no scatter       ≈ 781 us wall / 733 us event
N32 fused raw scatter ≈ 1545 us wall / 1493 us event
extra                 ≈ 764 us wall / 760 us event
```

score multiply 已经移到 local reduction；local reduction 在这组测试里约
`164 us` event，并没有随 epilogue 形态明显变化。因此当前主要剩余瓶颈仍在
raw scatter epilogue 的 peer write 路径，而不是 score/reduction。

local reduction 的第一轮优化来自
按 token 和 N tile 分块、模板化 `top_k`、把 score 缓存在 shared memory，并移除
per-element 的除法/取模。

FP8 的第一次实现直接沿用 N32 思路，每个 lane 只写 8 个 FP8 元素，因此 peer store
退化为 8B `st.global.v2.u32`。该版本虽然把远端数据量减半，但 fused GEMM+scatter
event 反而从 BF16 的约 `1493 us` 退化到约 `2031 us`。

N64 重排修正了这个问题：每个 lane 写 16 个连续 FP8 元素，用 16B
`st.global.v4.u32`。同一 H200 目标 shape 下：

```text
BF16 N32 direct fused scatter event = 1492.59 us
FP8 N64 direct fused scatter event  = 1188.58 us
```

也就是说，FP8 N64 相比 BF16 N32 的 fused scatter 部分快约 `20.4%`，整条
`fused scatter + local reduction` 路径从 `1753.26 us` 降到 `1463.07 us`，
快约 `16.6%`。

local-buffer 诊断把 destination pointer table 临时替换成本地 HBM buffer，只用于分离
本地 store/quantization 和远端 peer store 成本。这个模式是 benchmark 诊断入口，
不用于 correctness，因为 local reduction 仍然读取正式的 source-rank combine
buffer。

| 方案 | fused GEMM + scatter event | 说明 |
| --- | ---: | --- |
| BF16 N32 remote | 1492.59 us | 正常 peer IPC 写 |
| FP8 N64 remote | 1188.58 us | 正常 peer IPC 写 |
| BF16 N32 local-buffer | 841.22 us / 830.40 us | 两次长迭代复测，写本地 HBM |
| FP8 N64 local-buffer | 913.74 us | 同一 address footprint，但写本地 HBM |

对应的 no-scatter event 分别为：

| 方案 | GEMM no scatter event | local-buffer fused event | delta |
| --- | ---: | ---: | ---: |
| BF16 N32 local-buffer run 1 | 818.94 us | 841.22 us | +22.28 us |
| BF16 N32 local-buffer run 2 | 780.08 us | 830.40 us | +50.32 us |
| FP8 N64 local-buffer | 811.09 us | 913.74 us | +102.65 us |

因此 BF16 local-buffer scatter 的 kernel event 开销只有几十微秒量级；如果短迭代里
wall time 看起来差到 100 us 以上，主要是跨 rank barrier、rank skew 和 benchmark
block 间抖动。FP8 N64 local-buffer 相比 BF16 多出的部分主要来自 per-32 amax/scale
计算、FP8 conversion、FP8 data store 和 FP32 scale store。远端 peer write 仍是 remote
path 的主瓶颈：把 FP8 N64 从 local-buffer 改回 remote 后，fused event 从
`913.74 us` 增加到 `1188.58 us`。

### 6.4 K sweep 与计算/通信 overlap 模型

为了估计 GEMM 主体计算隐藏通信的比例，固定 M/N/output volume，只 sweep GEMM K：

| K | pure GEMM event | GEMM + scatter event | 暴露通信 `F(K)-C(K)` |
| ---: | ---: | ---: | ---: |
| 128 | 184.22 us | 1397.55 us | 1213.33 us |
| 640 | 415.23 us | 1421.57 us | 1006.34 us |
| 1280 | 733.68 us | 1500.70 us | 767.02 us |
| 1920 | 1038.98 us | 1568.02 us | 529.04 us |
| 2560 | 1371.38 us | 1644.80 us | 273.42 us |

一个更贴近当前 kernel 的一阶模型是：

```text
pure_gemm(K) = fixed_overhead_x + compute(K)
```

其中 `fixed_overhead_x` 包括 kernel/scheduler/固定 epilogue overhead；主体计算
`compute(K)` 随 K 变化，并可以和通信 overlap。假设 K=128 时主体计算足够小，基本被
通信覆盖，则：

```text
comm = F(128) - x
exposed_comm(K) = F(K) - C(K)
hidden_comm(K) = comm - exposed_comm(K)
compute(K) = C(K) - x

compute_overlap_ratio(K) = hidden_comm(K) / compute(K)
comm_hidden_ratio(K) = hidden_comm(K) / comm
```

用 pure GEMM K sweep 线性拟合得到：

```text
pure_gemm(K) ~= 111.16 us + 0.488 us * K
```

即 `x ~= 111.16 us`。代入后：

| K | hidden comm | compute overlap ratio | comm hidden ratio |
| ---: | ---: | ---: | ---: |
| 128 | 73.06 us | 100.0% | 5.7% |
| 640 | 280.05 us | 92.1% | 21.8% |
| 1280 | 519.37 us | 83.4% | 40.4% |
| 1920 | 757.35 us | 81.6% | 58.9% |
| 2560 | 1012.97 us | 80.4% | 78.7% |

这里需要区分两个比例：

- `compute_overlap_ratio`：主体计算中有多少比例与通信同时发生。
- `comm_hidden_ratio`：通信本身有多少比例被主体计算隐藏。

以 target K=1280 为例，模型给出的结论是主体计算约 `83%` 与通信 overlap，但通信本身
只被隐藏约 `40%`。这和 kernel 观测一致：K 越大，暴露通信下降；但 target K=1280 时
peer store 仍然是主瓶颈。

### 6.5 通信量与带宽估算

在目标 shape 下，每个 compute rank 处理所有 source token 在本 rank 2 个 local experts
上的 fc2 输出。远端写只统计写到其他 7 个 source ranks 的部分：

```text
remote rows per compute rank = tokens_per_rank * experts_per_rank_token * (num_ranks - 1)
                             = 6976 * 2 * 7
```

BF16 remote payload：

```text
6976 * 2 * 2048 * 2 bytes * 7 = 400.03 MB
```

FP8 N64 remote payload 包含 FP8 data 和 FP32 scale：

```text
data  = 6976 * 2 * 2048 * 1 byte  * 7 = 200.02 MB
scale = 6976 * 2 * (2048 / 32) * 4 bytes * 7 = 25.00 MB
total = 225.02 MB
```

用 fused event 与 no-scatter event 的差值粗略估算暴露 remote payload bandwidth：

| 方案 | exposed scatter event | remote payload | effective exposed BW |
| --- | ---: | ---: | ---: |
| BF16 N32 | 1492.59 - 801.94 = 690.65 us | 400.03 MB | 579.2 GB/s |
| FP8 N64 | 1188.58 - 758.13 = 430.45 us | 225.02 MB | 522.8 GB/s |

这个 bandwidth 不是纯 NVLink bandwidth；它混合了 pointer/base 读取、FP8 量化、
store issue、scoreboard stall、以及部分计算 overlap。更可靠的结论是：FP8 N64
把远端 payload 降到约 `56.25%`，并把暴露 scatter event 从约 `691 us` 降到约
`430 us`。

### 6.6 NCU 指标

下面是 BF16 N32 direct remote scatter 的 K sweep NCU 指标。`NVLink TX` 对应每个
compute rank 的远端写 payload；K=2560 的 NCU counter 返回 `nan`，表里填理论值。

| K | NCU duration | inst executed | eligible warps/cycle | long scoreboard | NVLink TX |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1450 us | 55.6M | 0.06 | 43.84 | 400.03 MB |
| 640 | 1470 us | 153.5M | 0.16 | 13.36 | 400.03 MB |
| 1280 | 1510 us | 305.0M | 0.36 | 5.82 | 400.03 MB |
| 1920 | 1580 us | 442.5M | 0.51 | 2.75 | 400.03 MB |
| 2560 | 1730 us | 571.2M | 0.59 | 1.84 | 400.03 MB |

K sweep 的 NCU 指标和 event time 趋势一致：

- K 小时，kernel 接近纯 remote store，`long scoreboard` 很高，eligible warp 很低。
- K 增大后，WGMMA 计算量增加，更多通信等待被计算覆盖，`long scoreboard` 下降。
- NVLink TX 与 K 无关，因为输出量固定。

K=1280 的 local-buffer 诊断：

| path | NCU duration | long scoreboard | NVLink TX |
| --- | ---: | ---: | ---: |
| BF16 N32 remote | 1510 us | 5.82 | 400.03 MB |
| BF16 N32 local-buffer | 766 us | 1.66 | 0 MB |

这说明 BF16 N32 path 的主要额外 stall 来自远端 peer store，而不是 accumulator
pack 或本地 store 指令本身。

### 6.7 compact layout row-order 实验

为了确认 fused scatter 的额外开销是否来自 source-rank row order 导致的 peer-store
热点，benchmark 增加了 `--compact-layout-order {token,ring}`。

其中 `ring` 模式只改变每个 expert 内真实 row 的排列顺序，不改变数学语义：

- `combine_src_index` 仍然指向同一个 source token 集合。
- `row_to_topk` 仍然指向对应 rank 的 top-k slot。
- `psum_layout` 仍使用真实 row 结束边界，避免重新引入 expert padding work。
- `m_logical` 仍为 `115456`，padding rows 仍为 `3840`。

correctness smoke：

```text
layout = compact-gemm2/ring
Fused combine-scatter epilogue value check:
max_abs_diff = 0
mismatches   = 0

CPU reference check for fused scatter + local reduction:
max_abs_diff = 0.00012207
tokens       = 32
mismatches   = 0
```

core-only 对比命令只保留 fused path 和 local reduction，使用 `warmups=10`、
`iters=50`：

```bash
python3 tests/bench_combine_scatter.py \
  --num-local-ranks 8 \
  --tokens-per-rank 6976 \
  --hidden 1280 \
  --n 2048 \
  --top-k 16 \
  --global-num-experts 512 \
  --experts-per-rank-token 2 \
  --no-gather-a \
  --compact-layout-order token \
  --warmups 10 \
  --iters 50
```

把 `--compact-layout-order token` 改成 `ring` 即可复现 ring-order 对比。

结果：

| compact layout order | GEMM no scatter | fused GEMM + scatter | local reduction | total fused + local reduction |
| --- | ---: | ---: | ---: | ---: |
| `token` | 825.26 us (event 810.75 us) | 1636.27 us (event 1572.69 us) | 202.16 us (event 162.21 us) | 1840.91 us |
| `ring` | 777.25 us (event 789.52 us) | 1636.59 us (event 1568.43 us) | 201.10 us (event 161.81 us) | 1842.06 us |

带 standalone scatter-copy 的辅助对比，使用 `warmups=10`、`iters=20`：

| compact layout order | raw scatter-copy | total GEMM + scatter-copy |
| --- | ---: | ---: |
| `token` | 1259.73 us (event 1188.11 us, 384.79 GB/s) | 1994.24 us (event 1917.02 us) |
| `ring` | 1222.41 us (event 1179.94 us, 387.46 GB/s) | 1944.21 us (event 1898.59 us) |

结论：

- ring-order 对 fused GEMM + scatter 的 event time 只改善约 `4.26 us`，约 `0.27%`。
- standalone scatter-copy 有约 `0.7%` 的 event time 改善，但幅度仍然很小。
- 这说明当前主要瓶颈不是简单的 source-rank row order 热点。后续优化应继续聚焦
  combine-scatter epilogue 的 remote store 路径和 accumulator 写回结构。

### 6.8 direct-store 重排实验结论

direct accumulator store 的关键不是“每个 lane 自己连续”，而是同一条 warp store
指令里相邻 lane 写相邻的 16B segment。

撤回的 N128 重排让每个 lane 自己写连续 8 个 BF16，但同一轮 store 中 lanes 0..3
写的是：

```text
lane0 -> 0..7
lane1 -> 32..39
lane2 -> 64..71
lane3 -> 96..103
```

这对 peer store 极差。当前保留的 N32 重排改成：

```text
lane0 -> 0..7
lane1 -> 8..15
lane2 -> 16..23
lane3 -> 24..31
```

B 的 N32 physical-to-logical 映射为：

```text
physical = pair * 8 + lane_group * 2 + elem
logical  = lane_group * 8 + pair * 2 + elem
```

代码侧对应：

```c++
logical_col = n_block_idx * 128 + vec * 32 + (lane_idx & 3) * 8;
accum_pair_base = vec * 4;
```

同一正常 H200 节点上的目标 shape 结果：

| 路径 | fused GEMM + raw scatter | local reduction | total |
| --- | ---: | ---: | ---: |
| staged peer scatter | 1643.71 us (event 1587.74 us) | 206.62 us (event 164.38 us) | 1841.87 us |
| N128 direct peer scatter（已撤回） | 4626.54 us (event 4591.52 us) | 206.60 us (event 163.52 us) | 4837.86 us |
| N32 direct peer scatter（当前保留） | 1545.07 us (event 1492.58 us) | 205.77 us (event 163.07 us) | 1781.66 us |

local-buffer 诊断把 peer pointer table 临时替换为本地 buffer。长迭代复测中 BF16 N32
local-buffer fused event 为 `830.40 us` 到 `841.22 us`，对应 no-scatter event 为
`780.08 us` 到 `818.94 us`。该诊断说明 N32 direct epilogue 的本地写回开销已经接近
no-scatter GEMM，当前大头仍是 peer store。这个诊断入口现在保留为
`--combine-scatter-local-buffer`，用于后续分离本地 store/quantization 和远端 peer
store 成本。

FP8 path 的对应结论是：不能简单把 BF16 N32 映射套到 FP8 上。N32 时每个 lane 只有
8 个连续 FP8 元素，只能生成 8B peer store；N64 后每个 lane 有 16 个连续 FP8 元素，
可以恢复 16B peer store，因此从 `2031.46 us` event 改善到 `1188.58 us` event。

### 6.9 已撤回的失败尝试

以下实验代码已经移除，不再作为 benchmark 或 API surface 保留：

1. **store warpgroup prototype**

   额外启动一个 warpgroup 专门做 `STSM -> LDS -> peer STG`。该原型增加了线程数、
   barrier 协作和寄存器分配复杂度，未形成可验证的稳定收益，因此撤回。

2. **N128 direct accumulator store**

   该版本试图通过每 128 列 B 重排让每个 lane 的 accumulator 在 N 方向连续。
   结果每个 lane 内部连续，但跨 lane store 地址稀疏，peer store 退化严重。
   BF16 path 已被 N32 direct store 替代；FP8 path 使用 N64 direct store。

3. **row-major peer TMA / fixed-peer STG**

   这类实验需要把最终 combine 语义改成 row-major intermediate，无法直接表达
   `combine_buffer[token, topk_slot, n]` 的 per-row 动态目标地址。它们只能作为
   upper-bound 诊断，不适合作为当前实现方向，因此相关代码不再保留。

---

## 7. 当前局限与后续方向

### 7.1 local reduction 与 FP8 reduction

本实现默认把每个 expert output 以 raw BF16 写入：

```text
combine_buffer[token, topk_slot, :]
```

随后由 `combine_reduce_slots` 做：

```text
out[token, :] += combine_buffer[token, topk_slot, :] * topk_scores[token, topk_slot]
```

FP8 path 则写入 FP8 combine buffer 和 FP32 scale buffer，local reduction 中反量化后
再乘 score。当前 target shape 下 local reduction 约为：

| path | local reduction event |
| --- | ---: |
| BF16 combine buffer | 162.82 us |
| FP8 combine buffer + FP32 scale | 185.49 us |

因此当前已经覆盖 fc2 输出的 remote scatter 和 source-rank local combine，但 scatter
和 reduction 仍然是两个 kernel。FP8 local reduction 多读 scale，多一次反量化，因此比
BF16 reduction 慢约 `23 us` event。

FP8 数值验证已经有初步拆解：`--fp8-debug-precision` 会比较 actual FP8 slot 反量化、
CPU per-32 FP8 quantized slot、CPU BF16 slot，以及 actual reduction。现有结果说明
slot 写入和 local reduction 没有明显实现错误，主要误差来自 FP8 量化本身。正式接入前
仍需要根据模型容忍度确定 FP8 combine 的 reference、scale 粒度和 acceptance threshold。

### 7.2 可能的优化方向

后续优化应优先围绕 epilogue 结构，而不是继续调 grid/block：

1. **以 FP8 N64 direct peer store 作为当前主线**

   N64 恢复 16B peer store 后已经明显优于 BF16 N32。下一步应围绕 FP8 path 优化
   scale 生成、scale store 和 local reduction，而不是退回 8B store 形态。

2. **继续优化 remote store 路径，而不是优先调整 compact row order**

   `--compact-layout-order ring` 只带来噪声级改善，说明 source-rank row order 不是当前
   fused scatter 的主瓶颈。后续应继续关注 peer-store 指令形态、store 粒度、写入合并
   以及 epilogue 内等待和同步成本。

3. **确定 FP8 correctness gate**

   性能路径和诊断路径已经能把误差拆到 slot、quantization 和 reduction 层面，但 FP8
   量化引入了真实数值误差。需要明确 per-32 scale 的误差目标，或评估 per-token/per-128
   scale 等其他粒度，再确定正式 correctness gate。

4. **暂不优先推进 row-major TMA intermediate**

   已经做过 fixed-peer 和 tile-rank row-major TMA 实验。它们证明动态 source-rank
   descriptor 选择本身不是主要瓶颈，但 row-major TMA upper-bound 仍然没有显著改善
   端到端性能。除非后续整体 combine contract 发生变化，否则不应继续在这条线上投入
   大量工程复杂度。

5. **继续评估 local reduction 的融合机会**

   `combine_reduce_slots` 已经从朴素 per-element kernel 优化为 token/tile 分块。
   后续主要看它是否能和下游算子融合，或者是否需要改变 combine buffer layout 来减少
   中间 BF16 写读。

6. **优化 reduce-scatter baseline 的 pack/local-reduce**

   当前 `combine_pack_for_reduce_scatter` 使用 per-element float `atomicAdd`，只是为了
   快速建立等价 baseline。后续可以利用 gather layout 中 `all-ranks-local` 的结构，
   按 token 或 tile 聚合，避免大规模 atomic，才能更公平地评估 reduce-scatter 方案。

7. **重新评估 H100 NVLink 目标环境**

   当前数据来自 H200。最终 target 是 NVLink H100，因此 peer store 部分仍需要在
   H100 HBM3 / H100 NVL 上复测。

---

## 8. 文件索引

核心实现：

- `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`
  - combine-scatter epilogue device 逻辑。
- `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp`
  - host/JIT 参数检查、kernel launch 参数透传。
- `csrc/apis/gemm.hpp`
  - Python-visible grouped GEMM API 参数扩展。

辅助验证与 baseline：

- `deep_gemm/include/deep_gemm/impls/combine_scatter_check.cuh`
- `csrc/jit_kernels/impls/combine_scatter_check.hpp`
- `deep_gemm/include/deep_gemm/impls/combine_scatter_copy.cuh`
- `csrc/jit_kernels/impls/combine_scatter_copy.hpp`
- `deep_gemm/include/deep_gemm/impls/combine_reduce.cuh`
- `csrc/jit_kernels/impls/combine_reduce.hpp`
- `csrc/apis/comm.hpp`
- `tests/bench_combine_scatter.py`
