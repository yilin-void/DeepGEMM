# SM90 FP8 1D2D GEMM 的 gather_index × per-rank flag overlap 设计

本文档分析在已有 `gather_index` 特性（见
`[sm90_fp8_gemm_1d2d_gather_index.md](./sm90_fp8_gemm_1d2d_gather_index.md)`）
之上，加入 **per-rank ready-flag 检查 + tile 级 zero-padding**，以实现
all-gather → grouped GEMM 的 fine-grained overlap。本文只讨论需求/设计/接口，
不包含具体实现 patch。

---

## 1. 背景与目标

### 1.1 上游场景

在 SP/EP MoE 训练或推理里，一次典型的 forward step 结构是：

```
local activations  ──[all-gather]──►  A_pool (来自全部 rank 的 token)
                                           │
                                           ▼
                                grouped FP8 GEMM (×W_expert)
                                           │
                                           ▼
                                    后续 expert 计算
```

- 单机内最多 8 个 rank（NVL8）。
- 每个 rank 通过 NVLink P2P 把自己的一份 token 送到其他 rank。
- 每个 rank 发的 token 数量 **相等且已知**（实际工程里通常是 `M_per_rank`，
在 batch 准备阶段确定）。
- 全部 rank 的 token 在本 rank 上 concat 到 `A_pool`，形状
`(num_ranks * M_per_rank, K)`。
- 之后的 grouped GEMM 在这份 `A_pool` 上做 token-to-expert 的 routing。

### 1.2 优化机会

朴素实现里，grouped GEMM 必须等 all-gather **整体完成**才能开始算。但
all-gather 是逐 rank 完成的：rank 0 送来的那份数据可能比 rank 7 早数十 µs
就到位。如果 GEMM 能识别“这一块输出只依赖 rank R 的数据”，就可以 **rank R
一到就开算**，与剩余 rank 的传输并行。

### 1.3 本特性的定位

我们要在已有的 `gather_index` 这套间接寻址路径上，再叠加：

1. **逐 rank 就绪检查**：每个 tile 计算前，等到“它依赖的那个 rank 的 flag = 1”。
2. **tile→rank 单一性**：通过 host 端 padding，保证每个 tile 只依赖单一
  rank，从而每 tile 至多等一个 flag。
3. **pad 行处理**：tile 内不够 `BLOCK_M` 的部分用 0 填充，**不再去 A 里取**，
  也就不需要等下一 rank 的 flag。

---

## 2. 需求形式化

### 2.1 输入与符号


| 符号             | 含义                  | 说明                                        |
| -------------- | ------------------- | ----------------------------------------- |
| `R`            | rank 数              | `R ∈ [1, 8]`                              |
| `T`            | 每个 rank 发送的 token 数 | 各 rank 一致；可不被 `BLOCK_M` 整除                |
| `A_pool`       | 拼接后的 token pool     | 形状 `(R*T, K)`                             |
| `M`            | grouped GEMM 输出行数   | 经 expert routing + per-rank padding 后的总行数 |
| `gather_index` | 输出行 → A_pool 行      | 长度 `M`，`int32`，**新约定**：负值表示 pad           |
| `flags`        | per-rank 就绪 flag    | 长度 `R`，`int64`，由远端写、本 kernel 读            |
| `tile_rank`    | tile → rank 映射      | 长度 `ceil(M/BLOCK_M)`，`int32`（或 `uint8`）   |


### 2.2 数据契约

A_pool 的内部布局：

```
 row [0,            T)        :  rank 0 的 T 个 token
 row [T,           2T)        :  rank 1 的 T 个 token
 ...
 row [(R-1)T,      RT)        :  rank R-1 的 T 个 token
```

因此对任何源行 `g ∈ [0, RT)`：

```
rank_of_source(g) = g / T          // 整数除
```

输出侧 `gather_index` 的约定（**新增**：相对原有 gather_index）：

- `gather_index[i] >= 0` ：合法源行号，`< RT`。
- `gather_index[i] <  0` ：**pad 行**，kernel 不读 `A_pool`，
对应输出行的 A 切片置 0。

`tile_rank` 的约定（**新增**）：

- `tile_rank[m_block_idx] >= 0`：这 tile 所依赖的 **唯一** rank。
- 对应 host 端构造保证：tile 内所有非 pad 行，
`gather_index[i] / T == tile_rank[m_block_idx]`。
- `tile_rank[m_block_idx] < 0`：mixed/unknown rank tile。这个情况会在
`expert_srank_padding=False` 时出现；kernel 保守等待所有 rank flags。
- 这是 host 端的 invariant；kernel 只验证非负值是否越界。

### 2.3 同步契约

flag 由通信侧（all-gather 的写端，或下游 NCCL/NVSHMEM 内核）按如下顺序写：

```
... 把 rank r 的所有 token 写入 A_pool 对应区段 ...
__threadfence_system();         // 或 fence.release.sys
flags[r] = 1;                   // cuStreamWriteValue64_v2 / st.release.sys 均可
```

GEMM kernel 读 flag 时必须用 acquire 语义：

```
while (deep_gemm::ptx::ld_acq_sys(&flags[r]) == 0) { /* spin */ }
```

`ld.acquire.sys` 既保证看到 flag=1 时同时看到 A_pool 的真实数据，又能
覆盖跨 rank（系统作用域）。本仓库里的
`deep_gemm/include/deep_gemm/ptx/ld_st.cuh` 已经提供
`ld_acq_sys(const uint64_t*)`，
可以直接复用。

flag 的状态机假设是 **单调** 的：

- 初始为 0；
- 写端把它写成 1 之后不会再清零；
- kernel 周期内不会发生 1→0。

如果有多轮迭代（例如多个 forward step 复用同一块 flag），
那是上层 buffer rotation 的事，kernel 只负责 “等到 flag = 1 再做”。

### 2.4 flag publish 实现与死锁规避

当前 H800 环境中 `cuStreamWriteValue32` / PyTorch
`stream_write_value32` 会返回 `CUDA_ERROR_NOT_SUPPORTED`，但 device
attribute 显示 64-bit stream memory op 可用。因此本实现把
`rank_flags` 固定为 `int64`，通信侧在当前 CUDA stream 上用
`cuStreamWriteValue64_v2(stream, &rank_flags[r], 1, 0)` 发布 ready flag，
GEMM 侧用 `ld.acquire.sys.global.b64` 读取。注意 lazy loader 必须显式
加载 v2 符号；未带后缀的 legacy export 在该环境中存在但会返回
`CUDA_ERROR_NOT_SUPPORTED`。

不要用“1 个 thread 的 store kernel”作为 32-bit fallback：overlap GEMM
会先启动并在 SM 上 spin 等 flag；大 shape 时这个 spin kernel 可能占满
全部 SM。若 flag writer 也是一个 kernel，它就拿不到 SM 执行，从而形成
循环等待：

```
GEMM spin kernel 等 rank_flags[r] == 1
flag store kernel 等空闲 SM
```

这就是 8-rank full-shape bench 曾经 hang 住的原因。可行解法是使用不占
SM 的 stream memory op（本实现选择 `cuStreamWriteValue64_v2`），或者使用
copy engine 的 stream-ordered H2D 小拷贝；不要在 overlap 期间用需要 SM
的 kernel 来写 flag。

### 2.5 Cross-rank barrier placement

`a_handle.barrier()` / `nvshmemx_barrier_all_on_stream` 这类跨 rank barrier
通常会 enqueue 一个需要 SM 执行的 GPU-side barrier kernel。它不能放在
full-SM grouped GEMM 之后再等待执行，否则 GEMM 中等待远端 `rank_flags` 的
CTA 可能占满 SM，而 barrier kernel 拿不到 SM，形成循环等待。

当前可用的调度方式是把 barrier 放到 GEMM 之前完成，但只让 GEMM 等到
barrier，不等远端 P2P pull 全部完成：

```
comm_stream:    zero rank_flags
comm_stream:    local D2D copy -> a_handle.barrier()
comm_stream:    record comm_ready event
compute_stream: wait comm_ready -> launch full-SM grouped GEMM
comm_stream:    enqueue P2P pulls after GEMM launch; each pull completion writes rank_flags[peer]
```

这样需要 SM 的 barrier 已经在 GEMM 启动前完成，GEMM 可以不预留 SM；后续
远端 P2P pull 预期走 copy engine/NVLink，与 GEMM overlap，并通过
`cuStreamWriteValue64_v2(rank_flags[peer])` 逐 rank 唤醒 GEMM tiles。

`gemm-first` / `after-local-copy` 如果仍把 barrier 放在 GEMM 启动之后，则必须
保留 SM 预留，否则仍有死锁风险。

### 2.6 Gather path SFA layout

普通 1D2D GEMM 会把 SFA 从原始 row-major per-token layout 转成 MN-major
TMA-aligned layout，以便按 `sfa[k_scale, m]` 读取时获得更好的合并访存。
但是 gather-index 路径读取 SFA 时必须先经 `gather_index[logical_m]` 做间接
寻址：

```
source_m = gather_index[logical_m]
scale_a  = sfa[source_m, k_block]
```

此时输出 tile 内的 `source_m` 不再保证连续，MN-major transform 的合并访存收益
不稳定；同时 transform 本身会在 GEMM launch 前引入 `empty_strided` 和
`transpose_fp32` kernel，推迟 overlap GEMM 下发。SFA 数据量也远小于 A，因此
rank-overlap 的 SM90 grouped+gather 路径直接保留原始 row-major SFA，并在 kernel
中按：

```
gmem_sfa + source_m * stride_sfa + sfa_k_idx
```

读取。非 gather 路径仍沿用原来的 MN-major transformed SFA。

### 2.7 SM-free cross-rank barrier

`a_handle.barrier()` 的语义是：所有 rank 都已经把本地 token 写入自己的
symmetric-memory slot，之后任意 rank 才能安全地从 peer 的 symmetric buffer
发起 P2P pull。这个语义不要求 barrier 本身占用 SM；在 overlap 路径里最好
用 CUDA stream memory op 在 symmetric memory 上实现一个 SM-free barrier。

基本思路：

1. 每个 rank 额外分配一个 symmetric `int64` barrier flag（或长度为 1 的
   symmetric tensor），并通过 symmetric-memory rendezvous 拿到所有 peer flag
   的 device pointer。
2. 本 rank 在 `comm_stream` 上完成本地 D2D copy：

   ```
   cudaMemcpyAsync(a_symm[rank], a_local, ..., cudaMemcpyDeviceToDevice, comm_stream)
   cuStreamWriteValue64_v2(comm_stream, local_barrier_flag, epoch, 0)
   ```

   因为 write value 与前面的 D2D copy 在同一 stream 上有序，所以 peer 看到
   `epoch` 时，本 rank 的 local slot 已经可读。
3. 每个 rank 在同一个 `comm_stream` 上依次等待 peer flag，然后发起对应 P2P
   pull：

   ```
   cuStreamWaitValue64_v2(comm_stream, peer_barrier_flag[r], epoch, CU_STREAM_WAIT_VALUE_GEQ)
   cudaMemcpyAsync(a_pool[r], peer_a_symm[r], ..., cudaMemcpyDeviceToDevice, comm_stream)
   cuStreamWriteValue64_v2(comm_stream, rank_flags[r], 1, 0)
   ```

   P2P pull 是 peer-to-peer `cudaMemcpyAsync`，预期由 copy engine/NVLink 推进，
   不需要为空闲 SM 做让步；`rank_flags[r]` 仍然是 GEMM kernel 的 per-rank
   ready flag，用来告诉 producer WG “rank r 的 A rows 已经被 pull 到本地”。

这样 `after-local-copy` overlap 可以变成：

```
comm_stream:    zero rank_flags
comm_stream:    local D2D copy -> write local barrier epoch
compute_stream: wait local-copy event -> launch full-SM GEMM
comm_stream:    wait peer barrier epoch -> P2P pull peer -> write rank_flags[peer]
```

注意事项：

- barrier flag 使用递增 `epoch`，不要反复用 0/1，否则下一轮可能读到上一轮
  的 stale flag。`cuStreamWaitValue64_v2(..., CU_STREAM_WAIT_VALUE_GEQ)` 等到
  当前 epoch 即可。使用 GEQ（而非 EQ）可避免快 rank 的下一轮 publish 覆写
  慢 rank 尚未 poll 到的值导致 hang。
- barrier flag 与 GEMM 读的 `rank_flags` 是两类 flag。前者同步 “peer local
  symmetric slot 已可被 P2P 读取”，后者同步 “本地 `A_pool` 中 rank r 的 rows
  已经可被 GEMM 读取”。
- 这个替换只绕开 symmetric-memory runtime 的 barrier 实现；如果某个 CUDA/driver
  环境不支持 64-bit stream wait/write value，应回退到 `a_handle.barrier()` +
  预留 SM 的保守路径。
- 实测当前 PyTorch symmetric-memory tensor 的 device pointer 不能直接作为
  `cuStreamWriteValue64_v2` / `cuStreamWaitValue64_v2` 的目标地址（driver 返回
  `CUDA_ERROR_INVALID_VALUE`）。因此落地时还需要一个 stream-mem-op 可接受的
  cross-rank flag 存储，例如 CUDA IPC 暴露的普通 device allocation、或者由
  symmetric-memory runtime 提供的 signal pad/doorbell API；在找到该存储前，
  `a_handle.barrier()` 仍是保守 fallback。

#### 2.7.1 Per-rank IPC stream signal prototype

NVSHMEM 的 on-stream signal 文档提到，在 P2P connected GPUs 上可优先使用
`cuStreamWriteValue` 实现 zero-SM stream signaling。对应到本实现，实际可行的
方向不是 “rank A 在自己的 stream 上 wait rank B GPU 上的 flag”，而是：

```
producer rank r:
  local D2D copy -> for each dst: cuStreamWriteValue64(dst.local_ready[r], epoch)

consumer rank c:
  before pulling src r:
    cuStreamWaitValue64(c.local_ready[r], epoch, GEQ)
    cudaMemcpyAsync(local_pool[r], peer_symmetric_slot[r])
    cuStreamWriteValue64(rank_flags[r], 1)
```

也就是 producer 通过 P2P stream write 写入 receiver GPU 上的 signal slot；
receiver 只 wait 自己 GPU 上的本地 slot。这与 NVSHMEM signal 的方向一致。

实测注意点：

- PyTorch symmetric-memory tensor 不能直接作为 `cuStreamWait/WriteValue64`
  的目标地址，driver 返回 `CUDA_ERROR_INVALID_VALUE`。
- 普通 PyTorch CUDA tensor 也不适合直接拿来做 CUDA IPC flag：caching allocator
  可能让 `tensor.data_ptr()` 带有底层 allocation 内 offset，而
  `cudaIpcOpenMemHandle` 返回的是 allocation base，peer 端会写错地址。
- 当前 prototype 使用裸 `cudaMalloc` 分配 `int64 local_ready[num_ranks]`，
  再通过 `cudaIpcGetMemHandle/cudaIpcOpenMemHandle` 交换并映射 base pointer。
  在 2-GPU signal probe 中，P2P `cuStreamWriteValue64` + local
  `cuStreamWaitValue64` 能正确唤醒；8-GPU overlap benchmark 也能跑通。

#### 2.7.2 Profiling caveat for IPC stream signals

**现象**

`ipc-stream` 模式下，8-rank `torch.profiler` timeline capture 曾经
在 profiling warmup 阶段卡住。2-rank / 4-rank 不受影响；8-rank
非 profiler benchmark 也能正常完成。

**根因：epoch 覆写竞争 + `CU_STREAM_WAIT_VALUE_EQ`**

profiling warmup 循环中连续调用两次 `fn()`，每次调用后只做
`torch.cuda.synchronize()`（本地 GPU 同步），**没有跨 rank 的
`dist.barrier()`**。而正常 benchmark 路径 `_bench_ms` 在每次迭代
后都有 `dist.barrier()`。

缺少 barrier 时，快完成的 rank A 可以在慢 rank B 仍在处理
epoch N 的 `cuStreamWaitValue64_v2(ready_flag[A], N, EQ)` 时，就
进入 epoch N+1 并通过 `cuStreamWriteValue64_v2` 把 B 的
`ready_flag[A]` 从 N 覆写为 N+1。当 B 的 GPU 执行到该 wait 时，
看到的值是 N+1，而不是 N。`CU_STREAM_WAIT_VALUE_EQ` 要求精确
匹配：`N+1 != N` → 永远不通过 → **hang**。

rank 数越多，各 rank 完成时间方差越大，竞争越容易触发。profiler
（CUPTI）额外增加了 overhead，使得方差更大，但 profiler 本身不是
根因。

**修复**

1. **`CU_STREAM_WAIT_VALUE_EQ` → `CU_STREAM_WAIT_VALUE_GEQ`**
   （`csrc/apis/comm.hpp`）。epoch 单调递增，GEQ 语义正确且不会
   被覆写竞争卡住。
2. **warmup 循环每次迭代后加 `dist.barrier()`**，与 `_bench_ms`
   保持一致，从根源上消除覆写竞争。

**`--profile-ranks` 参数**

benchmark 脚本额外提供 `--profile-ranks` 参数，可控制哪些 rank
启用 `torch.profiler.profile()`：

- `auto`（默认）：`ipc-stream` + `num_ranks >= 8` 时仅 rank 0
  启用 profiler，减少 CUPTI overhead 对跨 rank 时序的干扰。
- `all`：所有 rank 启用 profiler。
- 逗号分隔的 rank ID：例如 `--profile-ranks 0,1`。

示例：

```bash
# 8-rank ipc-stream profiling（auto 模式，仅 rank 0 生成 trace）
python tests/bench_single_node_allgather_overlap.py \
  --num-local-ranks 8 --cross-rank-sync ipc-stream \
  --profile --profile-iters 1

# 强制所有 rank 生成 trace
python tests/bench_single_node_allgather_overlap.py \
  --num-local-ranks 8 --cross-rank-sync ipc-stream \
  --profile --profile-ranks all
```

#### 2.7.3 ipc-stream 模式下消除 GEMM 与 comm 之间的 event 同步

§2.5 描述的 `allgather-first` 调度在 `barrier` 模式下必须让 GEMM 等待
comm_stream 的 `comm_ready` event（覆盖 `rank_flags.zero_()` + `copy_local`
+ `a_handle.barrier()`），因为 barrier 占 SM，必须先完成。

`ipc-stream` 模式下，GEMM kernel 内部已经通过 `ld.acquire.sys` 逐 rank
spin 等待 `rank_flags[r]`，不需要 event。唯一需要 event 的原因是
**`rank_flags` 使用 0/1 语义，每轮必须 `rank_flags.zero_()`，而 zero 在
comm_stream 上、GEMM 在 compute_stream 上，需要 event 保证 zero 在
GEMM 读 flag 之前完成**。

**解决方案：epoch-based rank_flags**

将 `rank_flags` 从 0/1 语义改为 epoch 语义，彻底消除 zero + event：

| | 旧（0/1 语义） | 新（epoch 语义） |
|---|---|---|
| copy_local 写 | `rank_flags[local_rank] = 1` | `rank_flags[local_rank] = epoch` |
| pull 写 | `rank_flags[remote_rank] = 1` | `rank_flags[remote_rank] = epoch` |
| kernel 等待条件 | `ld.acquire.sys(rank_flags[r]) == 0` → spin | `ld.acquire.sys(rank_flags[r]) < epoch` → spin |
| 每轮开始 | 需要 `rank_flags.zero_()` + event | **不需要 zero，不需要 event** |

epoch 单调递增，上一轮结束时 `rank_flags[r] == epoch-1`。新一轮 GEMM
kernel 检查 `rank_flags[r] < epoch`，`epoch-1 < epoch` 为真 → spin。
当 copy_local/pull 写入 `epoch` 后 → `epoch < epoch` 为假 → 放行。

`s_rank_seen` 缓存逻辑不变：每次 kernel 启动清零，首次看到
`rank_flags[r] >= epoch` 后设为 1，后续 tile 跳过 spin。

改动涉及三层：

1. **kernel**（`sm90_fp8_gemm_1d2d.cuh`）：新增 `uint64_t rank_flag_epoch`
   参数；spin 条件从 `== 0` 改为 `< rank_flag_epoch`。
2. **host JIT**（`sm90_fp8_gemm_1d2d.hpp`）：`Args` 加
   `uint64_t rank_flag_epoch`；`launch_impl` 透传。
3. **host API**（`gemm.hpp`）：加 `rank_flag_epoch` 可选参数；pybind 绑定。

bench 侧改动：
- 去掉 `rank_flags.zero_()`
- 去掉 `comm_ready` event 和 `compute_stream.wait_event`
- 传 `epoch` 给 GEMM（`rank_flag_epoch=epoch`）和 comm（`flag_value=epoch`）
- GEMM 和 comm 完全独立下发到各自 stream，无任何跨 stream 同步

最终时序（`allgather-first` + `ipc-stream`）：

```
comm_stream:    [copy_local] [publish x8] [wait+pull x7]
compute_stream: [──────────── GEMM kernel (spin on rank_flags) ──────]
                 ↑ 无 event，两 stream 完全并行
```

注意：`barrier` 模式仍需 event（barrier 占 SM，必须在 GEMM 之前完成）。

---

## 3. 现有代码梳理（与本特性的接口）

### 3.1 已具备的能力

#### 3.1.1 `gather_index` 间接寻址（已合入）

`sm90_fp8_gemm_1d2d_impl` 里 producer WG 在每个 M tile 开头一次性
预计算每条 cp.async 的源行号，然后在 K loop 里直接用：

```c++
uint32_t source_m_for_a[kAItersPerThread];
#pragma unroll
for (uint32_t i = 0; i < kAItersPerThread; ++i) {
    const uint32_t logical_m = m_global_base + row_of_iter(i);
    source_m_for_a[i] = (has_gather_index and logical_m < shape_m)
        ? __ldg(gather_index + logical_m) : logical_m;
}
// ...
cp_async4(dst_a, gmem_a + source_m_for_a[i] * stride_a + k_idx + col);
```

我们要扩展的是这条 “每 tile 一次” 的初始化阶段，以及 cp.async 本身。

#### 3.1.2 cp.async 路径自带 “src_size 受控置零”

PTX `cp.async.cg.shared.global` 的完整签名是：

```
cp.async.cg.shared.global [dst], [src], cp_size, src_size;
```

当 `src_size < cp_size` 时，剩余字节 **由硬件填 0**。这正是我们做
pad 行需要的：传 `src_size = 0` 即可让 16B 全部清零，barrier 计数
（`cp.async.mbarrier.arrive.noinc`）行为不变。

当前 kernel 里的 helper `cp_async4` 用的是不带 `src_size` 的简写（即默认
`src_size = cp_size = 16`）。本特性需要切到带 `src_size` 的版本，
让 pad 行传 0，非 pad 行传 16。

> SFA 用的是 `cp.async.ca.shared.global ..., 4`（4 字节窄拷贝），
> 同样支持 `src_size` 参数；pad 行传 0 即可。

#### 3.1.3 系统作用域 acquire load

`deep_gemm/include/deep_gemm/ptx/ld_st.cuh` 已提供：

```c++
CUTLASS_DEVICE uint64_t ld_acq_sys(const uint64_t* ptr);
```

直接用于 spin-wait flag。

#### 3.1.4 producer WG 内同步

producer WG（128 线程，4 warp）已经使用 NamedBarrier / `__syncwarp`
做内部同步。新加一个 “elected 线程 spin → NamedBarrier 广播” 的
模式不会引入额外资源。

### 3.2 需要新增的接口


| 入参           | 类型                     | 形状                   | 来源                    | 说明                                 |
| ------------ | ---------------------- | -------------------- | --------------------- | ---------------------------------- |
| `rank_flags` | `uint64_t*` (global)   | `(R,)`               | host 分配，通信侧写入         | 必须用 `ld.acquire.sys` 读             |
| `tile_rank`  | `int*` 或 `uint8_t*`    | `(ceil(M/BLOCK_M),)` | host 端 padding 阶段同步生成 | 每 tile 一次查表                        |
| `M_per_rank` | `uint32_t` (kernel 标量) | scalar               | host 已知               | 仅当不传 `tile_rank` 时由 kernel 反推 rank |


二选一：传 `tile_rank` 还是 `M_per_rank`，详见 §4.1。

### 3.3 不需要改动的部分

- TMA descriptor / TMA 路径：B 矩阵不参与 all-gather，TMA 不动。
- consumer (math) WG：不感知 rank 概念。pad 行 A=0 → A·B = 0，
WGMMA / promotion / epilogue 都自然产生 0；存回 D 也是 0
（pad 输出行的 D 值是 don't-care，host 后处理跳过即可）。
- scheduler：`scheduler.get_next_block` 给的 `m_block_idx` 即可作为
`tile_rank` 的索引，不需要重写调度器。
- empty/full barrier 计数：cp.async 数量不变（pad 行也参与 cp.async），
barrier 语义保持原状。

---

## 4. 设计要点

### 4.1 tile→rank 的映射方式：表 vs 在线推

#### 方案 A：host 提供 `tile_rank[m_block_idx]`（推荐）

- host 在准备 `gather_index` / 做 padding 时已经天然知道 tile 划分，
顺手 emit 一个 `int32` 数组 `tile_rank`。
- 大小：`ceil(M/BLOCK_M)` 个 int32。M=8192, BLOCK_M=128 → 64 项 = 256 B。
- kernel 每 tile 一次 `__ldg(tile_rank + m_block_idx)`，一条 LDG.E.CI
指令，命中 read-only cache，开销 ~10 cycles。
- 缺点：多一个 host→device tensor，多一个绑定参数。

#### 方案 B：kernel 从 `gather_index` 推 rank

- 因为每 rank 发 `T` 个 token，且 host 已经保证 tile 内单一 rank：
  ```c++
  // 取 tile 第一个非 pad 的源行
  int g0 = first_non_pad(gather_index, m_global_base);
  uint32_t rank = g0 / T;     // T 是 kernel 标量参数
  ```
- 缺点：
  - 需要 “找第一个非 pad” 的逻辑（pad 行 g<0），最差情况扫到 `BLOCK_M-1`；
  - 需要 host 把 `T` 作为标量传入；
  - integer divide 在 GPU 上不是单条指令（除非 `T` 编译期常量；运行期需
  `IDIV` 或 fastdiv 转成乘法）。
- 优点：少一个 tensor。

**结论**：选方案 A。对 8K M 来说一张 256 字节的小表换来 “一条 LDG +
零除法” 的简洁代码，划得来。方案 B 留作 “host 不方便 emit 表” 时的
fallback，文档里要双方案都允许，但默认走 A。

#### 方案 C（折衷）：直接复用 grouped_layout

如果 host 愿意把 rank 编码塞进 `grouped_layout` 的高位（低位仍是
expert id），kernel 解一次即可。但这跟现有 grouped contiguous /
psum layout 的语义耦合太重，**不建议**。本特性只在新加一个 `tile_rank`
张量时才落地，不动 `grouped_layout`。

### 4.2 pad 行的标记

**约定**：`gather_index[i] < 0` 表示 pad。

理由：

- 已有的 gather_index 路径里，`source_m_for_a[i]` / `source_m_for_sfa[i]`
是 `uint32_t`。把负值原样塞进去会变成大整数，刚好可以用一个
哨兵值（如 `0xFFFFFFFFu`）把它和合法源行（`< RT < 2^31`）区分开。
- 也可以保留 int32 类型，不强转 uint32，让 producer 先比 `< 0`。

具体做法（kernel 端伪码）：

```c++
constexpr uint32_t kPadSentinel = 0xFFFFFFFFu;
const int g = (has_gather_index and logical_m < shape_m)
    ? __ldg(gather_index + logical_m) : static_cast<int>(logical_m);
source_m_for_a[i] = (g < 0) ? kPadSentinel : static_cast<uint32_t>(g);
```

> **不要**用 “out-of-range source_m” 来表示 pad（譬如 `source_m == RT`），
> 因为我们也允许调用方传 `gather_index = nullptr`（向后兼容老的 gather=
> `arange(M)` 路径），那时 `source_m = logical_m`，没有 pad 概念。哨兵
> 值统一表示 pad，逻辑清楚。

### 4.3 cp.async 路径的 pad 处理

把现有的：

```c++
cp_async4(dst_a + ..., gmem_a + source_m_for_a[i] * stride_a + k_idx + col);
```

替换为带 `src_size` 的版本：

```c++
const uint32_t src_size = (source_m_for_a[i] == kPadSentinel) ? 0u : 16u;
const __nv_fp8_e4m3* src =
    (source_m_for_a[i] == kPadSentinel)
    ? gmem_a                                            // 任意合法地址都行
    : gmem_a + source_m_for_a[i] * stride_a + k_idx + col;
asm volatile(
    "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_a + ...))),
       "l"(src),
       "r"(src_size)
);
```

要点：

- **对 pad 行，`src` 仍必须是合法 cuda 内存指针**（PTX 不要求实际读，
但 cuda-memcheck 会校验地址）。最稳的做法是用 `gmem_a`（一定合法
且永远在 A_pool 范围内）。
- HW 在 `src_size = 0` 时把整 16 B smem 清零；硬件路径仍然计 1 条
在飞 cp.async，`cp.async.mbarrier.arrive.noinc` 的累计正确。

SFA 的 cp.async 走 `cp.async.ca.shared.global`，同样支持 `src_size`
受控（4B 窄拷贝），逻辑同 A：

```c++
const uint32_t src_size = (source_m_for_sfa[i] == kPadSentinel) ? 0u : 4u;
asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
    :: "r"(...), "l"(...), "r"(src_size)
);
```

> 需要核对的细节：`cp.async.ca` 的 `cp_size` 仅允许 4/8/16，
> `src_size` 范围是 `[0, cp_size]`，`src_size = 0` 在 ca 路径同样有效
> （PTX ISA 7.0+ / SM80+ 一致）。这一点已在 cutlass 的 predicate
> async copy 里大量使用，没问题。

### 4.4 flag polling 的位置与同步

**位置**：在 producer WG 的 outer loop（`while scheduler.get_next_block`）
开头、`source_m_for_`* 预计算之后、K loop 之前。

**线程**：用 producer WG 内的 “elected one” 做 spin-load，剩下 127
线程在 NamedBarrier 上等。这避免 128 线程同时刷同一个全局地址。

**伪码**：

```c++
__shared__ uint32_t s_rank_seen[8];   // 0/1 cache，避免重复 spin
if (threadIdx.x == kNumMathThreads and ...) {  // 初始化
    for (int r = 0; r < kNumRanks; ++r) s_rank_seen[r] = 0;
}

// ... 进入 producer WG 主循环 ...
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    const uint32_t m_global_base = ...;
    const uint32_t r = static_cast<uint32_t>(__ldg(tile_rank + m_block_idx));

    // 仅 elected 线程 spin
    if (warp_in_wg == 0 and cute::elect_one_sync()) {
        if (s_rank_seen[r] == 0) {
            while (deep_gemm::ptx::ld_acq_sys(rank_flags + r) == 0) { /* spin */ }
            s_rank_seen[r] = 1;
        }
    }
    // 等 elected 线程通告 “flag 已就绪”
    cutlass::arch::NamedBarrier::sync(kCpAsyncThreads, kBarrierIdRankReady);

    // ... 原本的 source_m 预计算 + K loop ...
}
```

要点：

1. `**s_rank_seen` 缓存**：flag 单调，已经看到 1 的 rank 不再 spin。
  每个 SM 上每个 rank 至多 spin 一次（其余 tile 直接走 cache hit）。
2. **elected 线程**：`cute::elect_one_sync()` 在 producer WG 4 个 warp
  中只让一个 warp 的一条 lane 进。也可以更省地选 warp 0 + lane 0。
3. **NamedBarrier ID**：要避开 kernel 中已有的 ID（目前用了 `0` 和 `1`），
  选 `2` 即可。
4. **acquire 语义**：必须 `ld.acquire.sys`，不能裸 `ld.global` 或
  `volatile`。前者会被乱序，后者只阻止编译器重排，不阻止 HW 重排，
   会导致 “看到 flag=1 但读到旧 A_pool” 的踩坑。
5. **memory model 边界**：`s_rank_seen[r]` 进入 1 后被 elected 线程
  写，其它线程读时只关心 “是不是已经过了 NamedBarrier”。NamedBarrier
   提供了发布点，所以 `s_rank_seen` 不需要 `volatile`。
6. **降级为 nullptr**：`rank_flags == nullptr` ⇒ 整个 spin 段 bypass，
  行为退化为现有 gather_index 路径。这是兼容性兜底。

### 4.5 与 multicast / cluster 的交互

SM90 1d2d 在某些形状下会启用 TMA multicast（cluster 内 2 个 CTA 共享
B 或 A）。需要确认本特性不破坏 multicast：

- **multicast on B**：两个 CTA 处理 `(m_block_a, n_block)` 和
`(m_block_b, n_block)`，A 各自加载，B 复用。两个 CTA 的
`tile_rank` 可以不同，各自 spin 自己的 flag。**不冲突**。
- **multicast on A**：两个 CTA 处理 `(m_block, n_block_a)` 和
`(m_block, n_block_b)`，A 复用、B 各自。两个 CTA 共享 m_block
⇒ 共享 rank ⇒ 各自 spin 同一个 flag，结果一致。**不冲突**。

cluster 内的 producer WG 各自独立 spin，不需要走 cluster barrier。
flag 的 scope 是 system，对 cluster 内的多 CTA 自然可见。

### 4.6 与 grouped GEMM 的协同

#### 4.6.1 normal GEMM (`GemmType::Normal`)

最简单：tile 划分只看 M。`tile_rank[m_block_idx]` 直接对齐
`m_block_idx ∈ [0, ceil(M/BLOCK_M))`。

#### 4.6.2 m-grouped contiguous (`GemmType::MGroupedContiguous`)

每个 tile 已经被 host 强制对齐到 expert 边界（现有约束：tile 内行
的 `grouped_layout[row]` 必须同值）。**新约束**叠加：tile 内的非
pad 行必须同 rank。

host 侧的 padding 策略变成：

```
对每个 (rank, expert) 二元组:
    rows_count = count_tokens_belonging_to(rank, expert)
    rows_padded = align_up(rows_count, BLOCK_M)
    在 layout 中分配 rows_padded 行
        前 rows_count 行：gather_index[i] = real_source_row,  grouped_layout[i] = expert
        后 (rows_padded - rows_count) 行：gather_index[i] = -1,  grouped_layout[i] = expert
    ※ tile_rank 对应的所有 ceil(rows_padded/BLOCK_M) 个 m_block 都标 rank
```

注意：`grouped_layout` 在 pad 行也要写正确的 expert 值（不能是 -1），
因为 scheduler 用它做 `is_computation_valid` 判定，只能是 `>= 0`
才算这块 tile “要算”。pad 行不参与最终结果，但仍走 GEMM 计算 0·B = 0。

> 老 gather 文档里 “gather_index = -1 直接当 invalid” 不再适用，因为
> -1 会被新 pad 语义占用。两种 invalid 含义并存，但 grouped_layout 的
> -1 仍然是 “整 tile skip”，gather_index 的负值是 “行级 zero pad”。
> 它们粒度不同、互不冲突。

#### 4.6.3 m-grouped contiguous with psum layout

跟上面同理。`grouped_layout` 改成 `current_psum_m`（cumulative m），
不影响本特性。

#### 4.6.4 m-grouped masked

masked layout 没有 gather_index 入口，也没有 all-gather 需求（一般
是 dynamic batch 的内部 padding）。**本特性不暴露 masked 入口**。

---

## 5. 推荐方案概览

```
┌──────────────────────────────────────────────────────────────┐
│  HOST PRE-PROCESS                                            │
│  ----------------                                            │
│  1) 从 expert routing 决定每个输出行属于哪个 (rank, expert)  │
│  2) 按 (rank, expert) 把行 padding 到 BLOCK_M 倍数            │
│  3) 输出:                                                    │
│       - gather_index[i]  : 源行号 ≥ 0；pad 行 = -1            │
│       - grouped_layout[i]: expert id（pad 行也填实际 expert） │
│       - tile_rank[mb]    : 该 tile 依赖的 rank                │
│       - rank_flags[r]    : 由通信侧写入，gemm 端只读          │
└──────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  KERNEL  (Producer WG, per tile)                             │
│  --------                                                    │
│  for each tile (m_block_idx, n_block_idx):                   │
│      r = tile_rank[m_block_idx]                              │
│      if !s_rank_seen[r]:                                     │
│          while ld.acquire.sys(flags[r]) == 0: spin           │
│          s_rank_seen[r] = 1                                  │
│      NamedBarrier.sync(producer WG)                          │
│                                                              │
│      precompute source_m_for_a / source_m_for_sfa            │
│        - if gather_index[i] < 0  ->  source_m = PAD_SENT     │
│                                                              │
│      for each k_block:                                       │
│          for each cp.async chunk:                            │
│              src_size = (source_m == PAD_SENT) ? 0 : 16      │
│              cp.async.cg [smem], [gmem], 16, src_size        │
│          ... TMA B + barrier arrive ...                      │
└──────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  KERNEL  (Math WG)  -- 不动                                   │
│  --------                                                    │
│  WGMMA + scale + epilogue 完全沿用现有逻辑                   │
│  (pad 行的 A=0  ⇒  A·B=0  ⇒  D[pad_row]=0)                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. 实现细节

### 6.1 kernel 端：`sm90_fp8_gemm_1d2d.cuh`

签名扩展：

```c++
template <... 现有模板参数 ..., uint32_t kNumRanksMax = 8>
__global__ void sm90_fp8_gemm_1d2d_impl(
    float* sfb, int* grouped_layout,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
    const __nv_fp8_e4m3* __restrict__ gmem_a,
    uint32_t stride_a,
    const float* __restrict__ gmem_sfa,
    uint32_t stride_sfa,
    const int* __restrict__ gather_index,
    // ↓ 新增三项
    const uint64_t* __restrict__ rank_flags,      // (R,)
    const int* __restrict__ tile_rank,            // (ceil(M/BLOCK_M),)
    uint32_t   num_ranks,                         // R, 运行期标量
    // ↑
    const __grid_constant__ cute::TmaDescriptor tensor_map_a, ...);
```

主要改点：

1. **shared mem**：在现有 `barrier_start_ptr` 之后追加
  `int s_rank_seen[kNumRanksMax]`（≤ 32 B，可忽略）。warp 1 elected
   线程在 barrier 初始化的同时把它清零。
2. **producer WG outer loop 头部**：插入 §4.4 的 spin-wait + NamedBarrier。
  `rank_flags == nullptr` ⇒ 整段 bypass。
3. `**source_m_for_`* 预计算**：把 `gather_index[logical_m]` 的负值
  翻译成 `kPadSentinel`。
4. **A / sfa cp.async**：换成带 `src_size` 的 PTX，按 `kPadSentinel`
  决定 0 或 16/4。
5. **TMA B 路径**：完全不动。

### 6.2 host JIT 端：`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp`

`SM90FP8Gemm1D2DRuntime::Args` 追加：

```c++
struct Args {
    ...
    void* gather_index;         // 现有
    void* rank_flags;           // 新增, int64* on device
    void* tile_rank;            // 新增, int32* on device
    uint32_t num_ranks;         // 新增, R
};
```

`launch_impl` 把它们按 kernel 签名顺序传进去。

`generate_impl` 模板字符串里把 `kNumRanksMax = 8`（编译期上限）作为
模板实参，运行期 `num_ranks ≤ 8`。

新增校验（在 `sm90_fp8_gemm_1d2d` 入口）：

```c++
if (rank_flags.has_value()) {
    DG_HOST_ASSERT(tile_rank.has_value());                 // 二者必须同时存在
    DG_HOST_ASSERT(gather_index.has_value());              // overlap 仅在 gather 路径
    DG_HOST_ASSERT(rank_flags->scalar_type() == torch::kLong);
    DG_HOST_ASSERT(rank_flags->is_cuda() and rank_flags->is_contiguous());
    DG_HOST_ASSERT(rank_flags->numel() >= num_ranks);
    DG_HOST_ASSERT(tile_rank->scalar_type() == torch::kInt);
    DG_HOST_ASSERT(tile_rank->is_cuda() and tile_rank->is_contiguous());
    DG_HOST_ASSERT(tile_rank->numel() >= ceil_div(m, BLOCK_M));
    DG_HOST_ASSERT(num_ranks > 0 and num_ranks <= 8);
}
```

> tip：`BLOCK_M` 在 host 侧只有 `get_best_config` 之后才确定，所以
> `tile_rank` 大小校验也要放在 config 拿到之后。

### 6.3 Python / pybind 端：`csrc/apis/gemm.hpp`

`fp8_fp4_gemm_nt` 和 `m_grouped_fp8_fp4_gemm_nt_contiguous` 各自追加：

```c++
const std::optional<torch::Tensor>& rank_flags          = std::nullopt,
const std::optional<torch::Tensor>& tile_rank           = std::nullopt,
const std::optional<int>&            num_ranks          = std::nullopt,
```

校验逻辑在 `gemm.hpp` 这一层做（参考 §6.2 的 host 校验复制一份），
然后透传到 sm90 入口；非 SM90 / 1d1d / SM100 路径如果传了 `rank_flags`
要直接 `DG_HOST_ASSERT(false)` 拒绝。

pybind 绑定：

```c++
m.def("fp8_fp4_gemm_nt", &fp8_fp4_gemm_nt,
      ...,
      py::arg("gather_index") = std::nullopt,
      py::arg("rank_flags")   = std::nullopt,
      py::arg("tile_rank")    = std::nullopt,
      py::arg("num_ranks")    = std::nullopt);
```

Python 端使用：

```python
deep_gemm.fp8_gemm_nt(
    a_pool_fp8, b_fp8, d,
    recipe=...,
    gather_index=gather_index,        # int32, 可含 -1 pad
    rank_flags=rank_flags,            # int64, shape (num_ranks,)
    tile_rank=tile_rank,              # int32, shape (ceil(m / BLOCK_M),)
    num_ranks=num_ranks,
)
```

---

## 7. 正确性与边界情况


| 情景                                      | 期望行为                                                                                    |
| --------------------------------------- | --------------------------------------------------------------------------------------- |
| 不传 `rank_flags`                         | 完全等价于现有 `gather_index` 路径                                                               |
| 传了 `rank_flags` 但 `gather_index = None` | host 校验失败（这是 misuse）                                                                    |
| 某 rank 始终未发 flag                        | spin 直到超时；建议 host 侧加 watchdog 或在 kernel 里加 cycle 计数后 trap（参考 `nvlink_barrier` 的 30s 超时） |
| `gather_index[i] < 0` 而 `i < shape_m`   | pad 行，A/sfa cp.async src_size=0；output 写 0                                              |
| `tile_rank[mb]` 越界（非负且 `>= num_ranks`） | trap；host 必须保证。建议 host 用 `assert` 兜底                                                      |
| `tile_rank[mb] < 0`                       | mixed/unknown rank tile，kernel 保守等待所有 rank flags                                             |
| tile 内全部是 pad（不该出现）                     | `tile_rank >= 0` 时 spin 当前 rank；`tile_rank < 0` 时等待所有 rank。host 不应该调度这种 tile                    |
| `num_ranks = 1`                         | 退化：每 tile spin 同一个 flag（且只 spin 一次因 cache），相当于一次性等 all-gather 完成                        |


### 7.1 数值正确性论证

记 logical 输出行 `i`：

- 非 pad 行：`source_m = gather_index[i]`，A 切片来自 A_pool 的真实行，
与显式 `A_perm = A_pool[gather_index]` 的结果**逐字节一致**
（和现有 gather_index 一样，只是多了 “等 flag” 这层时序保证）。
- pad 行：A 切片 = 0；sfa 切片 = 0；A·B = 0；
promotion = `scale_a * scale_b * 0 = 0`；
存回 D 时 `D[pad_row] = 0`。
注意 `sfa = 0` 的时候 `scale_a` 也是 0，与 `accum = 0` 相乘还是 0，
数值无溢出无 NaN（`0.0 * finite = 0.0`）。
**唯一需要小心的是**：如果 `sfa` cp.async 跑出来不是严格 0，
例如硬件 src_size=0 没真把 smem 清零，那 `scale_a * 0 = 0` 这条
保证不再成立。这一点要在测试里实测，或在 first stage 之前加一次
`__syncwarp + zero-init smem_sfa`（最小代价的兜底）。
  > 在 PTX 规范里，`cp.async.{cg,ca}` 的 `src_size < cp_size` 时，
  > 剩余字节由 hw zero-fill，**这是规范保证的**，不是实现细节。
  > 所以正常情况无需兜底。但建议在第一次落地时打开 debug 测一遍。

### 7.2 与现有兼容性

兼容矩阵（在现有 [gather_index 文档](./sm90_fp8_gemm_1d2d_gather_index.md)
表格上扩展）：


| 调用入口                                           | gather_index | rank_flags / tile_rank |
| ---------------------------------------------- | ------------ | ---------------------- |
| `fp8_gemm_nt` (SM90 1d2d)                      | 已支持          | **本特性新增**              |
| `fp8_gemm_nt` (SM90 1d1d)                      | 不支持          | 不支持                    |
| `m_grouped_fp8_gemm_nt_contiguous` (SM90 1d2d) | 已支持          | **本特性新增**              |
| `m_grouped_fp8_gemm_*_masked`                  | 不支持          | 不支持                    |
| `fp8_gemm_{nn,tn,tt}`                          | 不支持          | 不支持                    |
| 任何 SM100 / cublasLt / bmm 路径                   | 不支持          | 不支持                    |


`rank_flags = None` 时所有路径行为与改前一致。

---

## 8. 性能影响估算

### 8.1 没有 overlap 时的 overhead

不传 `rank_flags`：

- Args 多两个 nullptr 字段，host launch 增加 ~16 B 拷贝；可忽略。
- kernel 编译产物多一段 “if(rank_flags) {...}” dead code（运行期常量
分支，编译器会消掉一部分）；register pressure 不变。
- **无运行期 overhead**。

### 8.2 有 overlap 时的额外开销

传了 `rank_flags`：

- 每 tile 多 1 次 `__ldg(tile_rank+mb)`：~10 cycle，命中 read-only cache。
- 每 tile 多 1 次 NamedBarrier sync（producer WG 内）：~20 cycle。
- 每个 (SM, rank) 第一次 spin：受 flag 写入延迟主导，**这正是
overlap 的本质收益所在** — 我们要的就是 spin 等待时长 < 全 all-gather
等待时长。
- 每 cp.async chunk 多 1 个 src_size 操作数：编译期常量分支
（pad/non-pad），实际生成两条 ld.global.u32 选择 src_size + cp.async；
~1 cycle 增量。

总体：相对纯 GEMM kernel，rank-aware path 的稳态额外开销 < 1%。

### 8.3 收益估算

假设 all-gather 总用时 `T_ag`，按 rank 均分；GEMM 总用时 `T_gemm`，
按 m_block 均分。理想情况下 `T_total = max(T_ag, T_gemm)`，相比串行
`T_ag + T_gemm` 可省下 `min(T_ag, T_gemm)`。

实测要看：

- per-rank 数据量 vs per-rank 算力分配。
- pad 比例（pad 越多有效 FLOPS 越少）：高 pad 比例下 GEMM 端浪费算力，
但仍维持 latency 优势。
- flag 写入延迟和 SM scheduling 的耦合。

---

## 9. 测试与验证方案

### 9.1 单元正确性（无 overlap，只测 pad + tile_rank 通路）

在 `tests/test_gather_index.py` 基础上加：

1. **pad-only 测试**：把 `rank_flags` 全置 1（host 侧预设），
  只验证 pad 行 → 0、非 pad 行结果与 baseline 一致。
2. **rank=1 退化**：`num_ranks=1`，flag[0]=1，等价于现有 gather_index 路径。
3. **多 rank + 全部 flag 预置 1**：模拟 “通信已经全部完成”，只测 GEMM
  的 pad / gather 正确性。

### 9.2 端到端 overlap 测试

新增 `tests/test_gather_rank_overlap.py`：

1. **单 GPU 模拟**：在另一个 stream 上启动 `cudaMemcpy` 模拟 all-gather
  分阶段写入，配合 `cuStreamWriteValue64_v2` 写 flag。GEMM 端在主 stream
   启动，验证：
  - 结果与 baseline 数值一致；
  - GEMM kernel 启动时机比 “等所有 flag 都到再 launch” 提前；
  - 用 nsight-compute 看 producer WG spin 的 cycle 占比。
2. **多 GPU 实测**：在多 rank（NCCL/NVSHMEM）下，把 GEMM 接到真实
  all-gather 之后，对比：
  - Path 1: `all_gather; barrier; gemm(gather_index)`
  - Path 2: `all_gather (with flag publishing); gemm(rank_flags=...)`
   端到端 latency。

### 9.3 mock 通信端

为方便单 GPU 调试，提供一个 host fn 把 `rank_flags` 直接预置成 1 数组。
这条路径的 GEMM 行为应当与 `rank_flags = nullptr + gather_index` 完全
一致（除 spin overhead）。

### 9.4 Stress / fuzzing

随机生成：

- `num_ranks ∈ [1, 8]`
- 每 rank 的真实 token 数 ∈ `[0.5*M_per_rank, M_per_rank]`
- 各 rank 内的 token 路由到 `num_experts` 个 expert
- pad 比例 0% / 25% / 50% / 75%

对每组配置跑 baseline `index_select + gemm` vs `gemm(rank_flags=...)`，
diff 必须在 FP8 量化误差 (`1e-2`) 以内。

---

## 10. 文件改动清单（实施时）


| 文件                                                         | 类型       | 主要改动                                                                                                                                                |
| ---------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` | kernel   | 签名加 `rank_flags / tile_rank / num_ranks`；producer WG outer loop 加 spin-wait + cache；`source_m_for`_* 处理 `< 0` pad；A / sfa 改成带 `src_size` 的 cp.async |
| `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp`            | host JIT | `Args` 加 3 字段，`launch_impl` 透传，`sm90_fp8_gemm_1d2d` / `sm90_m_grouped_fp8_gemm_contiguous_1d2d` 入口加形参 + 校验                                          |
| `csrc/apis/gemm.hpp`                                       | py API   | `fp8_fp4_gemm_nt` 和 `m_grouped_fp8_fp4_gemm_nt_contiguous` 加形参 + 分支校验，pybind 绑定                                                                     |
| `tests/test_gather_index.py`                               | 测试       | 加 rank=1 / 全 flag=1 / pad-only 三组单 GPU 用例                                                                                                           |
| `tests/test_gather_rank_overlap.py`（新增）                    | 测试       | 用 `cuStreamWriteValue64_v2` 模拟分阶段 flag publish 的 overlap 场景                                                                                         |
| `docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md`（本文） | 文档       | —                                                                                                                                                   |


---

## 11. tile_rank / gather_index 生成 kernel 设计

§1–§10 把 GEMM 一侧的 `(gather_index, tile_rank, rank_flags)` 入参约定下来
之后，下一个工程问题是：**这些 tensor 怎么从 MoE routing map 算出来**。
本节描述生成它们的 CUDA kernel 设计，包括关键 layout 决策
（**expert-major outer + ring-order rank-minor inner**，下文称 layout A）、
per-(expert, rank) padding 规则，以及具体的四阶段 CUDA kernel 实现。

### 11.1 上下游

```
                 step                 step                  step
   per-rank   ─allgather─►  global  ─generator─►   gather_index    ─►   GEMM
  routing_map              routing_map           tile_rank          (overlap)
                                                  grouped_layout
```

- 每个 rank 通过 token gating 拿到本地 routing map（每个 token 选 top-k expert）。
- 全部 rank 在单机内做一次 routing-map 的 all-gather，每个 rank 都拿到全局
routing map（`(num_ranks * tokens_per_rank, top_k)` int32）。
- 本节的 **生成 kernel** 把全局 routing map 翻译成下游 GEMM 需要的
`(gather_index, tile_rank, grouped_layout, m_logical)` 四元组。

> 和 token activation 的 all-gather（即 `A_pool`）不同，routing map 的体积小
> （`T * K * 4 B`），可以提前完成；本节假设 routing map 已经全局可见。

### 11.2 输入与输出契约

#### 11.2.1 输入


| 名称                | 形状 / 类型                                      | 说明                              |
| ----------------- | -------------------------------------------- | ------------------------------- |
| `routing_topk`    | `(num_ranks * tokens_per_rank, top_k)` int32 | 每行是这个 token 选中的 top-k expert id |
| `local_rank`      | int                                          | 本 rank 的编号（决定 ring 起点）          |
| `num_ranks`       | int (≤ 8)                                    | 单机 rank 总数                      |
| `tokens_per_rank` | int                                          | 每个 rank 发送的 token 数（约定相同）       |
| `num_experts`     | int                                          | 总 expert 数                      |
| `top_k`           | int                                          | 每个 token 选的 expert 个数           |
| `block_m`         | int                                          | 必须与下游 GEMM 实际 `BLOCK_M` 匹配      |


> **routing_topk vs routing_mask**：另一种常见格式是 `(T, num_experts)`
> 的二值 mask（gating sigmoid 之后再做 top-k 通常会同时输出二者）。
> mask 的好处是无需保证 top_k 列里 expert 唯一；缺点是每 token 要扫描
> `num_experts` 列。生成 kernel 的实现以 **routing_topk 为主**（迭代次数
> 是 `T * top_k`，比 `T * num_experts` 少一个量级），如果 caller 只有
> mask，可以先 `torch.nonzero(mask)` 转一遍再调本 kernel。

#### 11.2.2 输出


| 名称               | 形状 / 类型                      | 含义                                        |
| ---------------- | ---------------------------- | ----------------------------------------- |
| `gather_index`   | `(M_logical_max,)` int32     | 输出行 → A_pool 行；pad 行 = `-1`               |
| `tile_rank`      | `(num_m_tiles_max,)` int32   | 每个 m-tile 依赖的 rank                        |
| `grouped_layout` | `(M_logical_max,)` int32     | 输出行 → expert id（pad 行也填该 chunk 所属 expert） |
| `m_logical`      | scalar int32 (device + host) | 实际有效的 M（含 pad），用于 GEMM 的 `m_d`            |


`M_logical_max` / `num_m_tiles_max` 上界由 caller 在 host 端预估并预分配
（详见 §11.4.4）。Kernel 不会 reallocate，越界由 host 负责保证不发生。

### 11.3 layout 关键设计：**expert-major outer + ring-order rank-minor inner**

#### 11.3.1 为什么 expert-major outer（即 “每个 expert 一整段连续”）

直接遵循上层 spec 的字面要求：

> 每个 expert 的所有 token 中，来自同一个 rank 的 token 的 gather index
> 必须放在一起；padding 也是每个 expert 的每个 rank 的 token 各自做 padding。

把 m-tile 排成 **expert-major outer**：

```
m-tile 顺序：
  (expert 0, rank R)    (expert 0, rank R+1)    ...    (expert 0, rank R-1)
  (expert 1, rank R)    (expert 1, rank R+1)    ...    (expert 1, rank R-1)
  ...
  (expert G-1, rank R)  (expert G-1, rank R+1)  ...    (expert G-1, rank R-1)
```

可视化（2 rank、3 expert，每段长度示意）：

```
     ┌── expert 0 ────────────────────────────┐ ┌── expert 1 ────────────────────────────┐ ┌─ expert 2 ─...
     │ R (real ▮▮▮ pad □) │ R+1 (real ▮▮▮▮)  │ │ R (real ▮▮▮▮)    │ R+1 (real ▮▮ pad □□)│ │ R (real ▮)  ...
     │←──── BLOCK_M ─────→│←──── BLOCK_M ───→│ │←──── BLOCK_M ───→│←──── BLOCK_M ──────→│ │
     │ tile_rank = R      │ tile_rank = R+1  │ │ tile_rank = R    │ tile_rank = R+1     │ │
m=0                                                                                                       M_logical
```

这么排的关键收益是 **下游 expert-contiguity**：grouped GEMM 的输出 `D`
天然按 expert 排好序，下一步（典型 MoE 里的 SiLU/down-projection grouped
GEMM/expert-wise reduce）可以**直接读 D**，不需要再跑一个 “按 expert
重排” 的 kernel。MoE forward 通常是 up-proj → silu → down-proj 三段
grouped GEMM 串起来，up-proj 输出的 layout 必须就是 down-proj 输入需要
的 layout，否则得插一层 permute；layout A 让两段 GEMM 直接对接。

#### 11.3.2 ring order = `(local_rank + s) % num_ranks`

每个 expert 内 rank 的排序遵循 NCCL ring all-gather 的数据到达顺序。
NCCL ring 在 `step s` 时 rank R 收到来自 `(R + s) % num_ranks` 的数据
（这里取 `+1` 方向；`-1` 方向把公式里 `+s` 换成 `-s` 即可）。

每个 expert 内的 rank 序列：

```
step 0 → local_rank                       (本地数据，无须等)
step 1 → (local_rank + 1) % num_ranks
...
step num_ranks-1 → (local_rank + num_ranks - 1) % num_ranks
```

例子（`num_ranks = 8`）：


| local_rank | 每个 expert 内的 rank 序列 (= tile_rank 重复段) |
| ---------- | -------------------------------------- |
| 0          | 0, 1, 2, 3, 4, 5, 6, 7                 |
| 1          | 1, 2, 3, 4, 5, 6, 7, 0                 |
| 7          | 7, 0, 1, 2, 3, 4, 5, 6                 |


整个 `tile_rank` 数组就是把这个 8 元序列**重复 num_experts 次**：

```
local_rank = 1, num_experts = 3:
  tile_rank = [1, 2, 3, 4, 5, 6, 7, 0,    ← expert 0
               1, 2, 3, 4, 5, 6, 7, 0,    ← expert 1
               1, 2, 3, 4, 5, 6, 7, 0]    ← expert 2
```

（实际每个 (expert, rank) chunk 可能跨多个 m-tile，每个 m-tile 都标
同一个 rank id；上面是每 (expert, rank) 只占 1 个 m-tile 的简化形式。）

#### 11.3.3 完整 layout 公式

```text
start = 0
for e in [0, num_experts):              # ← 外层 expert
    for s in [0, num_ranks):            # ← 内层 ring step
        rank_id = (local_rank + s) % num_ranks
        n_real = count[e][rank_id]                       # 第 11.5 节算出
        n_pad  = (-n_real) mod block_m                   # 向上对齐到 block_m
        n_slot = n_real + n_pad

        # 把 [start, start + n_real) 写成 source token id（在 A_pool 里的行号）
        # 把 [start + n_real, start + n_slot) 写成 -1（pad）
        # grouped_layout 整段都填 e
        # tile_rank 把这一段的 (n_slot / block_m) 个 m-tile 都填 rank_id

        start += n_slot

m_logical = start
```

#### 11.3.4 与现有 m-grouped contiguous 约束的兼容性

m-grouped contiguous scheduler 在两个地方读 `grouped_layout`，都只看
**每个 m-tile 第一行** 的值：

```c++
// scheduler/gemm.cuh
const auto offset = grouped_layout[m_block_idx * BLOCK_M];   // → expert id
return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0; // is_computation_valid
```

Layout A 满足：

1. **每个 m-tile 单一 expert**：每 (expert, rank) chunk 都是 `block_m` 的
  整数倍，m-tile 不会跨 expert。✓
2. **每个 m-tile 单一 rank**：同样的对齐保证 m-tile 也不会跨 rank（这是
  新 GEMM 路径需要的）。✓
3. **grouped_layout 在 expert 边界单调**：layout A 下 `grouped_layout`
  是 `0, 0, ..., 0, 1, 1, ..., 1, ..., G-1, G-1` 的形式，**严格单调
   非递减**，与原始 m-grouped contiguous 的语义最一致。✓
4. **TMA multicast** 看相邻 m-tile pair：`is_tma_multicast_valid` 比较
  `m_block_idx` 和 `m_block_idx ^ 1`。
  - **expert 内、相邻 rank 边界**（如 expert 0 的 rank-R 末 m-tile vs
  rank-(R+1) 首 m-tile）：两个 m-tile expert 相同 → multicast 仍然
  有效，**不退化**，比 layout B 更好。
  - **expert 边界**（如 expert 0 末 m-tile vs expert 1 首 m-tile）：
  expert 不同 → multicast 退化，与原 m-grouped contiguous 行为一致。

#### 11.3.5 Layout A vs Layout B 的取舍（设计决策记录）

我们考虑过另一种 layout B（rank-major outer + expert-minor inner），
这里记录一下取舍过程，方便后续读者理解。


|                                         | Layout A（本节选用）         | Layout B（备选）                 |
| --------------------------------------- | ---------------------- | ---------------------------- |
| 顶层结构                                    | expert-major outer     | rank-major outer             |
| `D` 中 expert 的连续性                       | ✅ 整段连续                 | ❌ 被 rank 边界切成 num_ranks 段    |
| 下游 down-proj / silu / scatter           | ✅ 直接读 D                | ❌ 需要先按 expert 重排             |
| `tile_rank` 重复模式                        | `[ring]` × num_experts | `[R, R, ..., R+1, R+1, ...]` |
| 每个 (CTA, rank) 第一次 spin 在哪              | expert 0 内部            | rank 段切换处                    |
| spin 间隔的 GEMM 工作量                       | 1 m-tile (expert 0 内部) | G 个 m-tile (一个 rank 段)       |
| AG 单步延迟 `X` 的容忍上界                       | `T_blk`                | `G * T_blk`                  |
| 预期 stall（典型 MoE: `T_blk≈50µs`，`X≈10µs`） | 0（X < T_blk）           | 0（X < G·T_blk）               |


**结论**：在 NVL8 + intra-node NVLink AG 的典型场景下，AG 单步延迟
`X` 远小于单 m-tile GEMM 时间 `T_blk`，**两种 layout 实际都不会 stall**，
overlap 收益相同。这时 layout A 的下游 expert-contiguity 优势是绝对的，
所以我们选 layout A。

只有在 “小 GEMM + 慢 AG”（例如 cross-node 跨 NIC 的 AG）这种 corner case
下 layout B 才能跑赢；如果未来真碰到这种场景，可以在 host 端加一个
`layout="rank_major"` 选项切换，kernel 端无须改动（kernel 只看
`gather_index` / `tile_rank`，不假设它们是哪种 layout 生成的）。

### 11.4 padding 规则

#### 11.4.1 per-(rank, expert) padding

```
n_pad = ceil(n_real / block_m) * block_m - n_real
```

每个 (rank, expert) chunk 各自向上 pad，**不跨 chunk 共享**。这是为了
保证：

- 每个 m-tile 只属于一个 (rank, expert) 二元组；
- pad 行的 `gather_index = -1` 触发 §4.3 的 `cp.async ... src_size = 0`
零填路径，下游 WGMMA 自然产生 0，不污染相邻有效行。

#### 11.4.2 grouped_layout 在 pad 行的取值

**填该 chunk 所属的 expert id**，而不是 `-1`。

理由：`is_computation_valid` 看 `grouped_layout[first_row] >= 0`。如果
首行恰好是 pad 行（不会发生，因为 real 行排在前面），值是 -1 的话整个
m-tile 都被 skip 不参与计算。我们希望 pad 行也走 GEMM、产生 0（这样
不需要在 host 端再做 “跳过 pad 输出行” 的特判，下游可以一视同仁）。

实际上：

- chunk 第一行一定是 real 行（除非 `n_real = 0`，详见 §11.4.3）；
- chunk 中间或末尾的 pad 行的 `grouped_layout` 值取自 chunk 的 expert，
下游 WGMMA 仍然算 `0 · B[expert] = 0`，对 D 的写入也是 0。

#### 11.4.3 退化情况：count[e][r] = 0

某 rank 的 token 完全没人选某 expert，常见情况。处理方式：

```
n_real = 0  →  n_pad = 0  →  n_slot = 0   # chunk 干脆不分配
```

这时该 (rank, expert) 不占任何 m-tile，layout 中 “无缝跳过”。注意因此
**真实 m_logical 是动态的**，需要 kernel 计算并返回。

#### 11.4.4 M_logical 上界（host 预分配用）

实现采用更紧的解析上界：

```
T            = num_ranks * tokens_per_rank
total_pairs  = T * top_k                                # Σ_chunks n_real 的精确上界
num_chunks   = num_experts * num_ranks
max_pad      = min(num_chunks, total_pairs) * (block_m - 1)
M_logical_max = total_pairs + max_pad
```

推导：

- 每个 token 贡献 `top_k` 个 (rank, expert) pair，所以 `Σ_chunks n_real ≤ T * K`
（OOB expert id 会被 host kernel filter 掉，所以实际是 `≤`）。
- 每个 chunk 的 pad 行 `n_pad < block_m`；最多有
`min(num_chunks, total_pairs)` 个非空 chunk（chunk 数受限于总
pair 数和总 chunk 槽位数中的较小者）。
- 因此 `Σ_chunks n_pad ≤ min(num_chunks, T·K) * (block_m - 1)`。

对照旧的宽松上界（`num_ranks * num_experts * ceil(tokens_per_rank, block_m)`）：


| 形状 (`num_ranks` / `num_experts` / `tokens_per_rank` / `top_k` / `block_m`) | 旧上界        | 新上界       | 节省        |
| -------------------------------------------------------------------------- | ---------- | --------- | --------- |
| 8 / 64 / 256 / 4 / 128                                                     | 524 288    | 188 416   | 2.8×      |
| 8 / 64 / 1024 / 4 / 128                                                    | 524 288    | 326 656   | 1.6×      |
| 8 / 512 / 7351 / 16 / 128                                                  | 30 408 704 | 1 461 120 | **20.8×** |


`gather_index` / `grouped_layout` 都是 `int32` 张量，按 M_logical_max
分配。tighter bound 让超大 expert 数 + 大 token 数的配置（典型
DeepSeek-V3 量级，`num_experts=512` / `top_k=16`）从 ~120 MB / 张量降到
~6 MB / 张量。

caller 用此上界 allocate；kernel 写 `m_logical` 标量；GEMM 调用时用
真实 `m_logical` 作为 `m_d`。

> 如果 caller 已经在 host 端持有路由统计，可以再额外收紧到
> `M_logical = Σ_chunks ceil(count[e][r], block_m) * block_m`，但需要把
> Phase 1 的 `counts` 同步回 host 并重算上界，会引入一次 device→host
> 同步。当前 generator 选择"一次性 launch + 解析上界"换简洁性。

### 11.5 算法概览（四阶段）

```
                  ┌──────────────────┐
 routing_topk ──► │ 1. histogram     │ ──► count[e][r]            (atomic)
                  │   (multi-block)  │
                  └──────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ 2a. prefix       │ ──► padded_starts[e][s]
                  │   (single block, │     m_logical
                  │   block-wide      │
                  │   parallel scan) │
                  └──────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ 2b. fill tables  │ ──► grouped_layout[*]
                  │   (multi-block,  │     tile_rank[*]
                  │   1 block /     │
                  │   chunk)         │
                  └──────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
 routing_topk ──► │ 3. scatter       │ ──► gather_index           (pad 行 = -1)
                  │   (multi-block,  │
                  │    atomic cursor)│
                  └──────────────────┘
```

每个阶段都是独立的 CUDA kernel；阶段之间通过 device-side stream
顺序保证可见性，**不需要 cooperative launch**。

> **设计演进**：早期版本是三阶段——Phase 2 是单 block，做 prefix + tile_rank
> 填充 + grouped_layout 整体填充；后两者占了 ~97% 的总时间（单 SM 顺序写
> ~1 M 个 int32 → ~2.4 ms）。当前版本把 Phase 2 拆成 prefix（仍是单 block，
> 但用 warp-shuffle 的 block-wide scan 替代 thread-0 串行循环）和 fill_tables
> （`(num_experts, num_ranks)` 个 block，每个 block 处理一个 chunk）两步。
> 详细分析见 §11.8。

> 索引约定：`count[e][r]` / `padded_starts[e][s]` / `chunk_cursor[e][s]`
> 都是 `expert-major`，与 §11.3 选定的 layout A 保持一致。`s` 是 ring step，
> 对应 `rank_id = (local_rank + s) % num_ranks`。

#### Phase 1 — Histogram

```c++
// 每线程处理一个 token；遍历它的 top_k 选项
template <uint32_t kNumThreads>
__global__ void histogram_for_gather_layout(
    const int* __restrict__ routing_topk,   // (T, K)
    int* __restrict__ counts,                // (num_experts, num_ranks), zero-init by host
    uint32_t T, uint32_t K,
    uint32_t num_experts, uint32_t num_ranks, uint32_t tokens_per_rank)
{
    const uint32_t t = blockIdx.x * kNumThreads + threadIdx.x;
    if (t >= T) return;
    const uint32_t r = t / tokens_per_rank;       // rank of this token

    #pragma unroll 1
    for (uint32_t j = 0; j < K; ++j) {
        const int e = __ldg(routing_topk + t * K + j);   // read-only cached load
        if (static_cast<uint32_t>(e) < num_experts)      // defensive: drop OOB
            atomicAdd(&counts[static_cast<uint32_t>(e) * num_ranks + r], 1);
    }
}
```

- 复杂度 `T * K` 个 atomicAdd，热点 `≤ num_experts * num_ranks` 个 cell。
对于参考形状（`num_experts=512`, `num_ranks=8` ⇒ 4 096 个 cell，
`T*K ≈ 940 K`），平均每个 cell ~230 次 atomicAdd，L2 上的争用可控。
- `__ldg` 走只读 cache（与 L1 D-cache 独立），让 Phase 1 / Phase 3 各自
独立读取 `routing_topk` 时不会互相挤占 L1 D。
- 全部 `counts` 由 host 在调度时 zero-init（`torch.zeros`）一次即可。
- 与 layout 选择无关：counts 永远是 (num_experts, num_ranks) 的纯统计。

> **为什么不做 smem block-local histogram + flush**：理论上能把"hot
> loop atomic"从 gmem 搬到 smem（~5× 更快），但 flush 阶段每个 block
> 仍要把所有非零 cell atomic 加到 gmem，gmem ops 总量不降反升（因为
> 每个 cell 现在被 ~num_blocks 次 atomicAdd 而不是 ~每 token 一次）。
> 当前 Phase 1 在 H800 上 ~30 µs/940 K ops ≈ 32 ns/op，已接近硬件 atomic
> 吞吐上限，没必要再压。

#### Phase 2a — Prefix sum (单 block，并行 block-wide scan)

输入 `counts[e][r]`，输出 `padded_starts[e][s]`（layout A：expert-major
外层 + ring step 内层）和 `m_logical`。**不**填 tile_rank / grouped_layout
——那两个分到 Phase 2b 多 block 并行做。

实现关键：用 warp-shuffle 的 block-wide exclusive scan 替代 thread-0
串行循环。

```c++
template <uint32_t kNumThreads>      // 256
__global__ void prefix_for_gather_layout(
    const int* __restrict__ counts,
    int* __restrict__ padded_starts,
    int* __restrict__ m_logical_out,
    uint32_t local_rank, uint32_t num_experts, uint32_t num_ranks, uint32_t block_m)
{
    constexpr uint32_t kChunksPerThread = 16;
    extern __shared__ int s_buf[];
    int* s_counts = s_buf;        // 一份 staging 输入

    // 1) cooperative load of counts → smem (coalesced + parallel)
    for (uint32_t i = threadIdx.x; i < num_experts * num_ranks; i += kNumThreads)
        s_counts[i] = counts[i];
    __syncthreads();

    // 2) 每线程处理 kChunksPerThread 个 chunk (e, s) (linear-idx blocked layout):
    //    idx = threadIdx.x * kChunksPerThread + i
    //    e   = idx / num_ranks
    //    s   = idx % num_ranks
    //    r   = (local_rank + s) % num_ranks
    //    n_slot = ceil(s_counts[e * num_ranks + r], block_m) * block_m
    //    →  local_total = Σ_i n_slot
    //
    // 3) block-wide exclusive scan over local_total:
    //    3a. warp inclusive scan via __shfl_up_sync
    //    3b. warp totals → smem; warp 0 scans them; broadcast back
    //    →  每个 thread 得到 my_excl_prefix
    //
    // 4) 顺序写自己负责的 padded_starts 段:
    //    for i: padded_starts[base + i] = prefix; prefix += local_n_slot[i];
    //
    // 5) thread 0 写 m_logical_out。
}
```

- 解决了之前 thread-0 串行循环的痛点：4096 chunk × ≥2 个 integer division
（`% num_ranks`、`/ block_m`）= 几十 K cycle，单线程 latency-bound 跑到
~~300 µs。block-wide scan 把这部分摊到 256 个线程，~~9 µs。
- smem 只需要一份 `num_experts * num_ranks * sizeof(int)` 的 staging
（即 64×8×4 = 2 KB / 512×8×4 = 16 KB）；warp-of-warp 用的小 smem
是 `static __shared__`，不占 dynamic smem。
- 编译期常量 `kChunksPerThread = 16` 决定了支持的最大 chunk 数：
`kNumThreads * kChunksPerThread = 4096`。device 上有 trap-style assert
保护越界配置。

#### Phase 2b — grouped_layout + tile_rank fill (multi-block, 1 block / chunk)

```c++
template <uint32_t kNumThreads>      // 128
__global__ void fill_layout_tables_for_gather_layout(
    const int* __restrict__ counts,
    const int* __restrict__ padded_starts,
    int* __restrict__ tile_rank,
    int* __restrict__ grouped_layout,
    uint32_t local_rank, uint32_t num_experts, uint32_t num_ranks, uint32_t block_m)
{
    // grid = (num_experts, num_ranks)
    const uint32_t e = blockIdx.x, s = blockIdx.y;
    const uint32_t r = (local_rank + s) % num_ranks;
    const int n_real = counts[e * num_ranks + r];
    const int n_slot = ((n_real + block_m - 1) / block_m) * block_m;
    if (n_slot == 0) return;                           // empty chunk: skip
    const int start = padded_starts[e * num_ranks + s];

    // grouped_layout: 每个 thread 写若干 int (合并 + 全局存储)
    for (int i = threadIdx.x; i < n_slot; i += kNumThreads)
        grouped_layout[start + i] = static_cast<int>(e);

    // tile_rank: n_tiles 通常很小（≤ tokens_per_rank/block_m），少量线程参与
    const int n_tiles    = n_slot / block_m;
    const int tile_base  = start / block_m;
    for (int t = threadIdx.x; t < n_tiles; t += kNumThreads)
        tile_rank[tile_base + t] = static_cast<int>(r);
}
```

- 每个 chunk 一个 block，不同 chunk 之间的 grouped_layout 写入完全独立，
所以总耗时 ≈ HBM 写吞吐 × 总写入字节数。对于参考形状 `m_logical=1.07 M`
（4 MB），132 SM 上 ~31 wave 完成，实测 ~4 µs。
- 没有了 tile_rank 的串行写（之前 Phase 2 thread 0 干的那堆），所以
该 phase 的 wall-time 受 HBM 写吞吐 + launch overhead 主导。
- chunk 之间独立写不同 region，不会有 atomic race；Phase 3 之后无需
等待这一 phase（它们写的是不同张量）。

#### Phase 3 — Scatter

```c++
template <uint32_t kNumThreads>
__global__ void scatter_for_gather_layout(
    const int* __restrict__ routing_topk,    // (T, K)
    const int* __restrict__ padded_starts,   // (num_experts, num_ranks) - [e][s]
    int* __restrict__ chunk_cursor,          // (num_experts, num_ranks), zero-init by host
    int* __restrict__ gather_index,          // (M_logical_max,) pre-init = -1
    uint32_t T, uint32_t K, uint32_t local_rank,
    uint32_t num_experts, uint32_t num_ranks, uint32_t tokens_per_rank)
{
    const uint32_t t = blockIdx.x * kNumThreads + threadIdx.x;
    if (t >= T) return;
    const uint32_t r = t / tokens_per_rank;
    const uint32_t s = (r + num_ranks - local_rank) % num_ranks;     // ring step

    #pragma unroll 1
    for (uint32_t j = 0; j < K; ++j) {
        const int e = __ldg(routing_topk + t * K + j);
        if (static_cast<uint32_t>(e) >= num_experts)
            continue;                                                // matches Phase 1
        const uint32_t chunk = static_cast<uint32_t>(e) * num_ranks + s;
        const int off = atomicAdd(&chunk_cursor[chunk], 1);
        const int pos = padded_starts[chunk] + off;
        gather_index[pos] = static_cast<int>(t);
    }
}
```

- 注意这里**只写 gather_index 不写 grouped_layout**——后者整段 chunk
（real + pad）都由 Phase 2b 统一填好了。这样 Phase 3 退化成"每 token
一次 atomic + 一次 scatter store"，chunk 内的 token 顺序仍然由 atomic
race 决定（test 用 multiset 比较）。
- gather_index 的 scatter 写虽然乱序，但写入区域是连续的 `[0, m_logical)`
（紧上界后只有 ~5.8 MB），cache locality 友好。
- 与 Phase 1 同样的 `__ldg` 优化；同样的 `atomicAdd` 争用模型，约
0.7 µs / 1 K tokens。

#### Phase 3 之前的 buffer 初始化

- `gather_index`：host 用 `torch.full((M_max,), -1, dtype=int32)` 预填 -1；
scatter 只覆盖 real 行的位置，剩下的天然是 -1（pad signal）。
- `grouped_layout`：**整段 chunk** 由 Phase 2b 写入对应的 expert id（real
和 pad 行都写同一个 e），无需 host 端 init，也无需后置 fill_pad
kernel。
- `chunk_cursor`：host 用 `torch.zeros` 初始化。
- `tile_rank`：Phase 2b 完整覆盖（每个 chunk 内的 `n_tiles` 个 entry），
无需预 init。
- `padded_starts`：Phase 2a 完整覆盖（block-wide scan 的输出），无需预 init。
- `m_logical_out`：标量，`torch.empty(1, dtype=int32)` 即可。

> Phase 3 已经把 real 行的 grouped_layout 写成 `e`，所以这一步事实上
> 也覆盖了 real 行（写同样的值）。这不影响正确性，只是冗余写一次。
> 如果想精确 “只写 pad 行”，把循环改成 `for (i = n_real; i < n_slot; ...)`。

- `chunk_cursor`：host 用 `torch.zeros` 初始化。
- `tile_rank`：phase 2 里全部覆盖完，无需预 init。
- `m_logical_out`：标量，`torch.empty(1, dtype=int32)` 即可。

#### 同步与 stream 调度

```
host: launch phase 1
host: launch phase 2 (depends on phase 1's counts)
host: launch phase 3 + fill_pad (depends on phase 2's padded_starts)
host: m_logical = m_logical_out.item()        # 一次 implicit sync
host: launch GEMM(rank_flags=..., gather_index=..., tile_rank=..., m=m_logical)
```

`m_logical_out.item()` 是必须的 host-device 同步点（GEMM 入口需要 Python
int 形式的 m）。把 generator 整体放在 GEMM 之前的同一个 stream，CUDA
graph 串起来即可；如果 routing map 早就到达，也可以把 generator
放到一个独立的 stream，与上游 token activation 的 all-gather 重叠。

### 11.6 host / Python API 草案

#### C++ 入口（建议挂在 `csrc/apis/layout.hpp` 或新文件 `csrc/apis/moe_gather_layout.hpp`）

```c++
// returns (gather_index, tile_rank, grouped_layout, m_logical_tensor)
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_gather_layout_for_rank_overlap(
    const torch::Tensor& routing_topk,    // (T, K) int32, T = num_ranks * tokens_per_rank
    const int& local_rank,
    const int& num_ranks,                 // must equal T / tokens_per_rank
    const int& tokens_per_rank,
    const int& num_experts,
    const int& block_m);
```

校验：

- `routing_topk.dim() == 2`、`scalar_type == kInt`、CUDA、contiguous；
- `routing_topk.size(0) == num_ranks * tokens_per_rank`；
- `0 <= local_rank < num_ranks ≤ 8`；
- `block_m` 必须与下游 GEMM 实际选到的 `BLOCK_M` 相同（caller 责任）。

#### Python 侧

```python
gather_index, tile_rank, grouped_layout, m_logical_t = \
    deep_gemm.build_gather_layout_for_rank_overlap(
        routing_topk,            # (T, K) int32
        local_rank=local_rank,
        num_ranks=num_ranks,
        tokens_per_rank=tokens_per_rank,
        num_experts=num_experts,
        block_m=BLOCK_M,
    )
m_logical = int(m_logical_t.item())

# 然后接到 GEMM
deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
    a_pool_fp8, b_fp8, d, grouped_layout,
    gather_index=gather_index[:m_logical],   # 也可以直接传整张，kernel 只用前 m_logical 行
    rank_flags=rank_flags,
    tile_rank=tile_rank,
    num_ranks=num_ranks,
    ...,
)
```

### 11.7 边界情况与正确性


| 情景                               | 处理                                                                                       |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| `count[e][r] == 0`               | chunk 不分配，跳过                                                                             |
| 整个 rank 的 token 都选同一个 expert     | chunk 最多 `tokens_per_rank` 行，`M_logical_max` 上界仍成立                                       |
| `top_k` 列里出现重复 expert（罕见）        | host 责任去重；否则同一行被 scatter 两次，gather_index 的 race 行为不可定义                                   |
| `tokens_per_rank % block_m != 0` | 没问题，每 chunk 各自 pad                                                                       |
| `local_rank >= num_ranks`        | host 校验失败                                                                                |
| `block_m` 与 GEMM 实际 BLOCK_M 不一致  | 行为未定义；建议 host 先用 `get_best_config` 拿到 BLOCK_M 再调 generator                               |
| `M_logical` 超过预分配 buffer         | UB；caller 用 §11.4.4 的紧上界 `T*K + min(num_chunks, T*K)*(block_m-1)` 预分配（generator 内部已自动处理） |
| GPU 不支持 atomic（很老的卡）             | atomicAdd on int32 在 SM61+ 全部 OK，无 issue                                                 |


### 11.8 性能数据（H800 实测）

参考形状：`num_ranks=8`, `tokens_per_rank=7351`, `top_k=16`,
`num_experts=512`, `block_m=128` ⇒ `T=58 808`, `T*K=940 928`,
`m_logical ≈ 1.07 M`（约 28% padded）。bench 脚本：
`tests/bench_gather_layout_generator.py`，每个 kernel 跑 30 次取平均
（`bench_kineto`），wall time 跑 20 次（`bench`）。

#### 11.8.1 优化前（早期单 block Phase 2 实现）


| 阶段                          | GPU 耗时        |
| --------------------------- | ------------- |
| Phase 1 (hist)              | 30.3 µs       |
| Phase 2 (prefix + fill)     | 2363.0 µs     |
| Phase 3 (scatter)           | 46.9 µs       |
| **GPU 总计**                  | **2440.2 µs** |
| Wall (alloc + launch + GPU) | 2488.2 µs     |


Phase 2 占 ~97%。原因：单 block 顺序写 `m_logical ≈ 1 M` 个 int32
到 `grouped_layout`（单 SM HBM 写吞吐瓶颈），thread 0 又串行扫
4096 个 chunk 算 prefix sum + 写 tile_rank（4 K 次 gmem load latency-bound）。

#### 11.8.2 优化后（当前实现）


| 阶段                                  | GPU 耗时      | 备注                                            |
| ----------------------------------- | ----------- | --------------------------------------------- |
| Phase 1 (hist)                      | 30.3 µs     | atomic-throughput bound（~32 ns/op）            |
| Phase 2a (prefix, parallel scan)    | 9.3 µs      | 256 线程 warp-shuffle scan                      |
| Phase 2b (fill grouped + tile_rank) | 4.4 µs      | `(num_experts, num_ranks)` 个 block 并行         |
| Phase 3 (scatter)                   | 44.8 µs     | 比 Phase 1 多一个 scatter store + 紧 5.8 MB region |
| **GPU 总计**                          | **88.7 µs** | **27.5× ↓**                                   |
| Wall                                | **97.5 µs** | **25.5× ↓**                                   |


附 `gather_index` / `grouped_layout` 内存占用对比（每张卡）：


| 量                       | 优化前（loose bound） | 优化后（tight bound） | 节省       |
| ----------------------- | ---------------- | ---------------- | -------- |
| `M_max`                 | 30 408 704       | 1 461 120        | 20.8×    |
| `gather_index` 大小       | 121.6 MB         | 5.8 MB           | 20.8×    |
| `grouped_layout` 大小     | 121.6 MB         | 5.8 MB           | 20.8×    |
| **总（gather + grouped）** | **243 MB**       | **11.6 MB**      | **~21×** |


#### 11.8.3 与不同 `tokens_per_rank` 的伸缩关系

固定 `num_ranks=8`, `top_k=16`, `num_experts=512`, `block_m=128`：


| `tpr`  | T      | Phase 1 | Phase 2a | Phase 2b | Phase 3 | 总            |
| ------ | ------ | ------- | -------- | -------- | ------- | ------------ |
| 1 024  | 8 192  | 7.0 µs  | 9.3 µs   | 4.4 µs   | 10.9 µs | **31.7 µs**  |
| 2 048  | 16 384 | 10.2 µs | 9.3 µs   | 4.4 µs   | 15.5 µs | **39.4 µs**  |
| 4 096  | 32 768 | 17.3 µs | 9.3 µs   | 4.4 µs   | 26.3 µs | **57.3 µs**  |
| 7 351  | 58 808 | 30.2 µs | 9.3 µs   | 4.4 µs   | 47.0 µs | **90.9 µs**  |
| 12 000 | 96 000 | 48.0 µs | 9.2 µs   | 4.5 µs   | 73.1 µs | **134.9 µs** |


Phase 1 / Phase 3 与 `T = num_ranks * tokens_per_rank` 完全线性
（atomic-throughput bound），Phase 2a / 2b 只取决于 chunk 总数 `num_experts * num_ranks`，对 token 数不敏感。

#### 11.8.4 进一步优化方向（暂未实施）

- **Phase 1/3 的 atomicAdd 争用**：当前在 ~32 ns/op，已经接近 H800 的 atomic
吞吐上限。若要进一步压缩，可以做 block-local smem histogram + flush，
但 gmem ops 总量基本不变（flush 阶段 ~num_blocks × num_chunks 次），
预期收益 < 30%。
- **更紧的 host-side bound**：先 sync `counts` 回 host 再算精确
`Σ ceil(count[e][r], block_m) * block_m`，能把上界再压 1–2×，但需要
device → host 同步。
- **合并所有 phase 到一个 grid-sync kernel**：去掉 launch 间隙的 ~2 µs ×
4 launches = ~8 µs；占当前 wall time ~10%，对 e2e 影响有限。

### 11.9 测试方案

#### 11.9.1 Python 参考实现

写一份纯 Python / numpy 实现 **layout A**（expert-major outer + ring-order
rank-minor inner）：

```python
def reference_build_gather_layout(routing_topk, local_rank, num_ranks,
                                  tokens_per_rank, num_experts, block_m):
    """Reference (layout A) implementation. Matches GPU generator's output as
    a multiset; per-chunk ordering may differ due to atomicAdd race."""
    T, K = routing_topk.shape
    assert T == num_ranks * tokens_per_rank

    # 1. histogram counts[e][r]
    counts = torch.zeros(num_experts, num_ranks, dtype=torch.int32)
    for t in range(T):
        r = t // tokens_per_rank
        for j in range(K):
            counts[routing_topk[t, j], r] += 1

    # 2. prefix: outer expert, inner ring step (layout A)
    starts = torch.zeros(num_experts, num_ranks, dtype=torch.int32)
    tile_rank_list = []
    cum = 0
    for e in range(num_experts):                          # ← 外层 expert
        for s in range(num_ranks):                        # ← 内层 ring step
            r = (local_rank + s) % num_ranks
            starts[e, s] = cum
            n_real = counts[e, r].item()
            n_slot = ((n_real + block_m - 1) // block_m) * block_m
            tile_rank_list.extend([r] * (n_slot // block_m))
            cum += n_slot
    m_logical = cum

    # 3. scatter
    gather = torch.full((m_logical,), -1, dtype=torch.int32)
    glayout = torch.zeros(m_logical, dtype=torch.int32)
    cursor = torch.zeros_like(starts)
    for t in range(T):
        r = t // tokens_per_rank
        s = (r - local_rank + num_ranks) % num_ranks
        for j in range(K):
            e = routing_topk[t, j].item()
            off = cursor[e, s].item(); cursor[e, s] += 1   # layout A: (e, s)
            pos = starts[e, s].item() + off
            gather[pos] = t
            glayout[pos] = e

    # fill pad's grouped_layout (no-op for real rows since they were already
    # written above; this just paints the pad-row tail of each chunk with `e`)
    for e in range(num_experts):
        for s in range(num_ranks):
            r = (local_rank + s) % num_ranks
            n_real = counts[e, r].item()
            n_slot = ((n_real + block_m - 1) // block_m) * block_m
            glayout[starts[e, s].item() + n_real : starts[e, s].item() + n_slot] = e
    return gather, torch.tensor(tile_rank_list, dtype=torch.int32), glayout, m_logical
```

#### 11.9.2 GPU vs reference 等价性

随机 `routing_topk`（top-k 去重），跑 GPU generator 和 reference，比较：

- `m_logical` 一致；
- `tile_rank` 完全相同；
- `gather_index` 在每个 chunk 内可能 **顺序不同**（atomicAdd 顺序非确定），
但作为 multiset 应当相同；如果想 bit-exact，可以在 host 端 sort 每个
chunk 之后再比较；
- `grouped_layout` 完全相同。

#### 11.9.3 端到端

把 generator → GEMM(rank_flags=...) 串起来，flag 全置 1，对比：

- baseline：host-side 显式 `index_select(A_pool, gather_index_ref) + gemm`；
- new：generator + gemm with overlap。

输出 `D` 应当（除 fp8 量化误差外）一致。在多 rank 真实通信下还要测端到
端 latency 是否真的下降，参考 §9.2。

### 11.10 文件改动清单


| 文件                                                        | 类型       | 主要改动                                                                                                                                                                                            |
| --------------------------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deep_gemm/include/deep_gemm/impls/moe_gather_layout.cuh` | kernel   | 四个 device 函数：`histogram_for_gather_layout`、`prefix_for_gather_layout`（block-wide parallel scan）、`fill_layout_tables_for_gather_layout`（multi-block，1 block / chunk）、`scatter_for_gather_layout` |
| `csrc/jit_kernels/impls/moe_gather_layout.hpp`            | host JIT | 四个 `LaunchRuntime` (Histogram / Prefix / FillTables / Scatter) + `build_gather_layout_for_rank_overlap` 入口；其中 host 端实现紧 `M_max` 上界（`T*K + min(num_chunks, T*K)*(block_m-1)`）                    |
| `csrc/apis/layout.hpp`                                    | py API   | pybind 暴露 `build_gather_layout_for_rank_overlap`                                                                                                                                                |
| `tests/test_gather_layout_generator.py`                   | 测试       | reference Python vs GPU 等价性 + 集成 e2e（`-t unit/e2e/all`）                                                                                                                                         |
| `tests/bench_gather_layout_generator.py`                  | bench    | 4 个 phase 用 `bench_kineto` 单独计时 + wall time，默认参数即文档 §11.8 形状                                                                                                                                    |
| 本文档                                                       | 文档       | §11.4.4（紧上界推导 + 对照表）、§11.5（四阶段流程图）、§11.6（Phase 详解）、§11.8（实测性能数据）、§11.10、§11.11                                                                                                                  |


### 11.11 与现有 mega_moe layout 的关系

仓库里已有 `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` 处理 MoE
token 路由相关的 metadata 布局，且 `csrc/jit_kernels/heuristics/mega_moe.hpp`
有对应的调度。本节描述的 generator **可以直接复用 mega_moe 中的 dispatch
counter / cumulative offset 逻辑**，避免重新发明轮子；具体复用点在实施
时再细化（`get_dispatch_count_ptr`、`get_expert_recv_count`_* 等结构体
的语义已经接近本节的 `count[e][r]` 与 `padded_starts`）。

如果直接复用 mega_moe 的输出能凑齐 `(gather_index, tile_rank, grouped_layout)`，
本节描述的 phase 1 + phase 2a + phase 2b + phase 3 四阶段 kernel 就只剩
Phase 3（scatter + tile_rank 一致化）。这是后续工程实施的优先调研方向。

---

## 12. 已知限制与后续工作

1. **仅 SM90 1d2d**：和 gather_index 一致。SM100 / 1d1d / bf16 / cublasLt
  都不在本次范围。
2. **flag 类型固定 int64**：当前环境的 32-bit stream mem op 不可用，
  因此使用 `cuStreamWriteValue64_v2` + `ld.acquire.sys.global.b64`。如果未来
  要把 flag 压回 int32，需要先确认 `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1`
  或等价能力可用，且不能退化为需要 SM 的 flag-store kernel。
3. `**num_ranks` 上限 = 8**：对应 NVL8。如果跨节点的 flag 检查（带
  IB/NVLink-Sharp 等）需要更多 rank，这个上限要调高，并且 spin 模型
   需要重新评估（跨节点写 flag 的延迟分布更尾部）。
4. **pad 行仍然 cp.async + WGMMA 计算 0**：pad 越多有效算力浪费越大。
  后续优化方向：
  - 对全 pad 的 m_block，scheduler 直接跳过（需要 host 别 schedule
  这种 tile，或 scheduler 用 `tile_rank == kSkipSentinel` 跳过）。
  - 对 “最后半个 tile” 这种部分 pad，能否让 WGMMA 只算前 `valid_m`
  行？现行 WGMMA 是 `BLOCK_M` 整体计算，要做这一步需要改 wgmma
  维度选择，工程量较大，留作后续。
5. **超时检测**：当前 spin 没有 watchdog。建议参考 `comm/barrier.cuh`
  的 30s 超时模式，elected 线程在 spin 循环里加 `clock64()` 检查。
6. **多迭代 buffer rotation**：本设计假设 flag 单调 0→1。多次 launch
  复用 buffer 时，host 必须在每次 launch 之前清零 flag，并保证
   通信端的写不会跨 launch。这是上层契约，不在 kernel 范围。
7. `**tile_rank` 与 `gather_index` 一致性**：generator（§11）保证两者
  同源；如果 caller 不走 generator 而是手工拼，建议在 debug build 加
   `DG_TRAP_ONLY_DEVICE_ASSERT` 抽查 “tile 内首个非 pad 的
   `gather_index[i] / tokens_per_rank == tile_rank[mb]`”。
8. **routing_topk 去重**：generator 假设每个 token 的 top_k 列里 expert
  id 互不相同。若 host 不能保证（罕见），需要在 phase 3 之前加一遍去重，
   或在 atomic scatter 时用 bitmask 标记已处理 expert。

---

## 14. Overlap GEMM 性能优化：跳过已缓存 rank 的 sync+fence

### 14.1 瓶颈定位

在当前实现（`sm90_fp8_gemm_1d2d.cuh` producer WG 的 per-M-tile rank check）中，
`NamedBarrier::sync(128, 2)` 和 `membar.sys` 在 `has_rank_flags == true` 时
**对每个 M-tile 无条件执行**：

```cpp
if (has_rank_flags) {
    if (warp_in_wg == 0 and cute::elect_one_sync()) {
        // spin on rank_flags[r] — 仅在 s_rank_seen[r]==0 时执行
        // s_rank_seen[r] = 1;
    }
    cutlass::arch::NamedBarrier::sync(kCpAsyncThreads, 2);  // ← 无条件
    asm volatile("membar.sys;" ::: "memory");                // ← 无条件
}
```

spin 本身已通过 `s_rank_seen` 做了 per-CTA per-rank 的缓存（只 fire 一次），
但 **NamedBarrier + membar.sys 对全部 ~1045 tiles 都执行**。真正需要它们的
只有 ~8 tiles（每个 unique rank 首次遇到时）。

实测开销（ng=64, n=2560, k=2048, m_logical≈134K, H800）：


| 路径                                          | 耗时         | TFLOPS |
| ------------------------------------------- | ---------- | ------ |
| gather-only [K] (test_gather_index, psum=0) | 1926 us    | 736    |
| overlap GEMM-only (rank_flags 全部预设为 1)      | 2317 us    | —      |
| **差值（rank_flags 开销）**                       | **391 us** | —      |


开销逐 tile 分解（临界路径 CTA 约处理 ~8 tiles）：

- `membar.sys`：**~40 us/tile**。system-scope fence 需等待 NVLink 可见性，
是最大开销源。
- `NamedBarrier::sync(128, 2)`：~2–5 us/tile。CTA-scope barrier，128 线程。
- `__ldg(tile_rank)` + smem check：~0.1 us/tile。可忽略。

### 14.2 优化方案：全线程 `ld.acquire.sys` 替代 `membar.sys`

原始实现的根本问题是：只有 elected 线程执行了 `ld.acquire.sys`，其余
127 个线程需要通过 `NamedBarrier` + `membar.sys` 这条间接路径获得
system-scope 可见性。而 `membar.sys` 是一个 drain 整个 memory pipeline
的 system-scope fence，开销极大（~40 us/tile）。

更好的做法是 **让全部 128 个 producer-WG 线程各自直接做
`ld.acquire.sys(rank_flags + r)`**。每个线程独立获得 system-scope
可见性，后续该线程发起的 `cp.async` 自然能看到远端数据。
这样就完全不需要 `NamedBarrier` 和 `membar.sys`。

```cpp
if (has_rank_flags) {
    const uint32_t r = static_cast<uint32_t>(__ldg(tile_rank + m_block_idx));
    DG_TRAP_ONLY_DEVICE_ASSERT(r < num_ranks and r < kNumRanksMax);
    if (*reinterpret_cast<volatile uint32_t*>(s_rank_seen + r) == 0) {
        // 128 线程各自 spin：同地址 L2 broadcast，一次 memory transaction
        while (deep_gemm::ptx::ld_acq_sys(rank_flags + r) == 0) {
        }
        // 一个线程更新 smem 缓存供后续 tile fast-path 使用
        if (warp_in_wg == 0 and cute::elect_one_sync()) {
            s_rank_seen[r] = 1;
        }
        // 无 NamedBarrier：没有需要在线程间传播的 smem 状态
        // 无 membar.sys：每个线程自己的 ld.acquire.sys 已保证
        //   后续 cp.async 对远端 gmem_a 的可见性
    }
}
```

**关键对比**：

| | 原始实现 | 全线程 acquire |
| --- | --- | --- |
| 谁执行 acquire | 1 个 elected 线程 | 128 个线程各自执行 |
| 如何传播可见性 | NamedBarrier + membar.sys | 不需要传播，每线程独立 |
| new-rank 开销 | spin + barrier(~3us) + membar(~40us) | 128 线程 spin (broadcast, ~1us) |
| cached-rank 开销 | barrier(~3us) + smem read + branch | volatile smem read + branch (~0.1us) |
| deadlock 风险 | NamedBarrier 内 diverge 可能 | 无 barrier，无 deadlock 可能 |

### 14.3 正确性论证

#### 14.3.1 `ld.acquire.sys` 对后续 `cp.async` 的 ordering

PTX 内存模型规定 `ld.acquire` 对**执行它的线程**后续所有内存操作建立
happens-before 关系。`cp.async` 是由线程发起的异步拷贝指令，属于该线程
的内存操作，因此受 acquire 的 ordering 保证约束。128 个线程各自执行了
`ld.acquire.sys(rank_flags + r)` 并看到返回值 1（与远端
`st.release.sys` 配对），每个线程后续发起的 `cp.async` 都能看到远端
rank 写入 `gmem_a` 的全部数据。

#### 14.3.2 128 线程读同一地址的性能

128 个线程读同一个 `rank_flags[r]`（8 bytes，在同一条 cache line 内）。
L2 cache 将其作为 broadcast 处理——只产生一次 memory transaction，
结果广播给所有请求线程。比 `membar.sys`（drain 全局 memory pipeline）
轻量几十倍。

#### 14.3.3 无 NamedBarrier → 无 deadlock

不使用任何 barrier，因此不存在"部分线程进入 barrier、部分跳过"的
死锁场景。即使 volatile smem read 在极端情况下有的线程读到 0
（进入 spin）、有的读到 1（跳过 spin），结果也是安全的：
- 进入 spin 的线程：flag 已经是 1（cached rank），spin 立即返回，
  多做一次冗余 `ld.acquire.sys`，无副作用。
- 跳过 spin 的线程：在首次 acquire 的同一 tile 或之前 tile 已经
  执行过 `ld.acquire.sys`，远端数据已可见。

#### 14.3.4 `s_rank_seen[r]` 的跨 tile 可见性

`s_rank_seen[r] = 1` 由 elected 线程在 spin 之后写入 shared memory。
虽然没有显式 NamedBarrier，但 K-loop 中的 `empty_barriers->wait()`
（CTA-scope transaction barrier）提供了充分的 CTA-scope fence：
后续 M-tile 迭代时，128 线程的 volatile smem read 都能看到值 1，
走 fast path 跳过 spin。

### 14.4 预期收益

| 阶段 | 预计 GEMM-only 耗时 | 说明 |
| --- | --- | --- |
| 优化前 | 2317 us | NamedBarrier + membar.sys 对每个 tile 无条件执行 |
| 条件 membar 优化 | 2091 us | 仅 new-rank tiles 执行 membar；cached 仍付 NamedBarrier 开销 |
| **全线程 acquire** | **~1930 us** | 去掉 NamedBarrier 和 membar.sys；cached tiles 仅 volatile read |
| gather-only 参考 | 1926 us | 不带 rank_flags 的 gather GEMM |

overlap bench 预期从 `speedup=1.046x` 提升至 ~1.10–1.12x。

---

## 13. 参考

- 现有 gather_index 设计：`[sm90_fp8_gemm_1d2d_gather_index.md](./sm90_fp8_gemm_1d2d_gather_index.md)`
- 现有 cp.async WG 资源分析：`[cpasync_wg_optimization_analysis.md](./cpasync_wg_optimization_analysis.md)`
- PTX `cp.async` 与 `src_size` 语义：PTX ISA 8.x §
*Data Movement and Conversion Instructions: cp.async*
- 跨 rank acquire / release 模式参考：
`deep_gemm/include/deep_gemm/comm/barrier.cuh` 中的 `nvlink_barrier`
- system-scope load 原语：
`deep_gemm/include/deep_gemm/ptx/ld_st.cuh` 的 `ld_acq_sys`
