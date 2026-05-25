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
combine_tokens_per_rank:  Optional[int]
combine_top_k:            Optional[int]
```

当 `combine_row_topk` 非空时，GEMM 进入 combine-scatter epilogue；否则保持原有
TMA store D 的行为。`combine_src_index` 可不传；这种情况下实现会复用
`gather_index` 作为 source-token map，以兼容原来的 GEMM1-style 调用。

### 3.2 输入含义

| 参数 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `gather_index` | optional `(M,) int32 CUDA contiguous` | output row 到 A/SFA source row 的映射；为空时 A 已按 grouped row 排好 |
| `combine_src_index` | optional `(M,) int32 CUDA contiguous` | output row 到 global source token 的映射；为空时回退使用 `gather_index` |
| `combine_row_topk` | `(M,) int32 CUDA contiguous` | output row 最终写入的 top-k slot |
| `combine_buffer_ptrs` | `(num_ranks,) int64 CUDA contiguous` | 每个 source rank 的 combine buffer device pointer |
| `combine_tokens_per_rank` | scalar int | 每个 rank 的 token 数 |
| `combine_top_k` | scalar int | combine buffer 的 top-k 槽数 |

`combine_buffer_ptrs[r]` 指向 rank `r` 暴露给本 rank 写入的 buffer，逻辑形状为：

```text
[tokens_per_rank, combine_top_k, N]
```

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
- 输出 dtype 当前按 BF16 处理。
- `combine-scatter` 必须提供 `combine_src_index` 或 `gather_index` 中的一个。
- `combine_buffer_ptrs.numel() <= 8`。
- `N % 8 == 0`，因为 epilogue 使用 16B `uint4` store（8 个 BF16）。
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

### 4.3 为什么先写 shared memory

最早的直接实现是从 accumulator lane 里取 BF16x2，然后每次发一个小的 global
store。这个版本可以工作，但 store 粒度太小，peer store 数量太多，性能很差。

当前实现改成两步：

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
--scatter-rows-per-block      # standalone scatter-copy 的 rows/CTA，默认 4
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
global tokens = 55808
local routed rows before expert padding = 55808 * 2 = 111616
intermediate hidden = 1280
hidden = 2048
```

对应到 benchmark 参数：

```text
tokens_per_rank = 55808 / 8 = 6976
hidden           = 1280   # GEMM K, fc2 input/intermediate dimension
n                = 2048   # GEMM N, fc2 output/hidden dimension
```

主要命令：

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
  --bench-standalone-scatter \
  --bench-reduce-scatter \
  --warmups 10 \
  --iters 20
```

log：

```text
workspace/logs/bench_combine_scatter_gemm2_target_h200.log
```

### 6.2 结果

形状：

```text
ranks = 8
tokens/rank = 6976
hidden = 1280
global_top_k = 16
local_top_k = 2
m_logical = 131328
n = 2048
gather_a = false
```

结果按方案拆分如下。表里 `total` 是直接测整条路径的 median，不是把各组件 median
简单相加。

| 方案 | 组件 | 时间 |
| --- | --- | ---: |
| common | GEMM no scatter | 867.71 us (event 824.72 us) |
| fused scatter | fused GEMM + raw scatter | 1634.20 us (event 1566.21 us) |
| two-stage scatter-copy | GEMM no scatter | 867.71 us (event 824.72 us) |
| two-stage scatter-copy | raw scatter-copy | 1230.10 us (event 1182.38 us, 386.66 GB/s) |
| two-stage scatter-copy | total GEMM + scatter-copy | 2025.61 us (event 1980.83 us) |
| reduce-scatter baseline | GEMM no scatter | 867.71 us (event 824.72 us) |
| reduce-scatter baseline | pack/local-reduce | 1332.01 us (event 1289.18 us) |
| reduce-scatter baseline | NCCL reduce-scatter SUM | 1201.28 us (event 1157.58 us, alg-bw 345.57 GB/s) |
| reduce-scatter baseline | total GEMM + pack + RS | 3281.74 us |
| fused scatter + local reduction | fused GEMM + raw scatter | 1634.20 us (event 1566.21 us) |
| fused scatter + local reduction | scored local reduction | 422.82 us (event 381.58 us) |
| fused scatter + local reduction | total fused + local reduction | 2068.58 us |

correctness：

```text
Fused combine-scatter epilogue value check:
max_abs_diff = 0
mismatches   = 0

CPU reference check for fused scatter + local reduction:
max_abs_diff = 9.15527e-05
tokens       = 16
mismatches   = 0
```

standalone scatter-copy 写入数据量：

```text
tokens_per_rank * top_k * n * sizeof(bf16)
= 6976 * 16 * 2048 * 2
= 457,179,136 bytes
```

对应带宽：

```text
457.2 MB / 1.18238 ms = 386.66 GB/s
```

### 6.3 解读

以 two-stage 为 baseline：

```text
two-stage event       ≈ 1981 us
fused combine-scatter ≈ 1566 us
```

fused epilogue 省掉约 `415 us`，约 `1.26x`。这说明 fused 路径确实把独立
scatter-copy 的大部分成本藏进了 GEMM epilogue，而不是简单地把通信原样追加到
GEMM 后面。

如果看完整 combine 对比：

```text
GEMM + pack/local-reduce + reduce-scatter ≈ 3282 us
fused combine-scatter + local reduction   ≈ 2069 us
```

当前 fused 路径快约 `1.59x`。不过这个结论要谨慎解读：reduce-scatter baseline
里的 pack/local-reduce kernel 目前是朴素 atomic 实现，单独就要约 `1289 us` event；
NCCL reduce-scatter 本身约 `1158 us` event。因此这条 baseline 现在更多用于建立数学等价
验证和性能参照，不能代表充分优化后的 reduce-scatter 方案上限。

但 fused 仍然显著慢于“不做 scatter 的 GEMM”。同形状不带 combine-scatter 的历史
baseline 为：

```text
GEMM without scatter ≈ 1470.81 us
```

GEMM2-style no-gather A、target shape 的当前 baseline 为：

```text
GEMM without scatter ≈ 867.71 us
```

因此当前 fused epilogue 仍然额外引入约 `766 us` wall time，或约 `741 us` event。
score multiply 已经移到 local reduction，fused epilogue 的额外成本主要来自
`accumulator -> shared memory -> 16B peer store` 这条写回路径以及 remote store 本身。

### 6.4 诊断实验结论

为了定位融合 scatter 的额外开销，曾经在 score multiply 合入之前做过几组一次性
诊断实验。这些诊断开关已经从正式 benchmark 中移除，但结果对后续优化仍有参考价值。

| 诊断路径 | 结果 | 结论 |
| --- | ---: | --- |
| `null dst`：走 STSM + scatter loop，但不发 global store | 1783.55 us | 仅 epilogue 改写本身已经比 no-scatter GEMM 慢约 319 us |
| `local dst`：scatter 到本地 HBM | 1850.44 us | local HBM store 只比 null 多几十 us |
| `peer dst`：scatter 到 peer combine buffer | 约 2.1 ms | peer store 额外增加约 200-300 us |
| `row-major peer dst`：目标按 logical row 连续排布 | 2069.01 us | final slot 的跨行不连续不是主瓶颈 |

结论：

- 当前主要瓶颈不是 final slot 地址不连续。
- 也不是 standalone peer bandwidth 本身完全暴露，因为 fused 已经比 two-stage 快很多。
- 更大的成本来自当前 epilogue 结构：`accumulator -> shared memory -> 16B STG`
  这条路径引入了额外 STSM、barrier、shared-memory read 和 scatter loop。

---

## 7. 当前局限与后续方向

### 7.1 local reduction 已独立实现，尚未融合

本实现把每个 expert output 以 raw BF16 写入：

```text
combine_buffer[token, topk_slot, :]
```

随后由 `combine_reduce_slots` 做：

```text
out[token, :] += combine_buffer[token, topk_slot, :] * topk_scores[token, topk_slot]
```

因此当前已经覆盖 fc2 输出的 remote scatter 和 source-rank local combine，但 scatter
和 reduction 仍然是两个 kernel，中间通过 BF16 combine buffer 连接。

### 7.2 可能的优化方向

后续优化应优先围绕 epilogue 结构，而不是继续调 grid/block：

1. **减少 accumulator -> smem -> gmem 的额外路径成本**

   当前 vectorized store 需要先把 accumulator fragment 落到 row-major shared
   memory。诊断显示这部分本身已经很贵。需要探索是否能在不退回 BF16x2 小 store
   的情况下减少 staging/barrier 成本。

2. **写 row-major intermediate，再做 local combine**

   如果最终 combine 允许先写一个 row-major intermediate buffer，可以重新使用更规则
   的写回模式，再由 source rank 做 local reduction。这会改变 buffer contract，
   但可能比直接写 final `[token, topk, n]` slot 更适合高性能 epilogue。

3. **优化 local reduction**

   score multiply 已经从 GEMM epilogue 移到 local reduction。后续需要优化
   `combine_reduce_slots` 的访存和向量化，评估它是否能和后续算子融合，或者是否有
   更适合 top-k 维度 reduction 的 layout。

4. **优化 reduce-scatter baseline 的 pack/local-reduce**

   当前 `combine_pack_for_reduce_scatter` 使用 per-element float `atomicAdd`，只是为了
   快速建立等价 baseline。后续可以利用 gather layout 中 `all-ranks-local` 的结构，
   按 token 或 tile 聚合，避免大规模 atomic，才能更公平地评估 reduce-scatter 方案。

5. **重新评估 H100 NVLink 目标环境**

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
