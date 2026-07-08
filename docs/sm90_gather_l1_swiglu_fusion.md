# SM90 Gather L1 + SwiGLU Fusion

## Scope

This experiment adds an explicit SM90 specialization to
`m_grouped_fp8_gemm_nt_contiguous` that fuses:

1. gather-index activation loading;
2. FP8 L1 GEMM;
3. BF16 rounding at the original L1 boundary;
4. SwiGLU;
5. FP8 E4M3 quantization with one FP32 scale per row and K128 output block.

The existing BF16 GEMM path is unchanged. The specialization is selected only
when `swiglu_output_scale` is provided and `d` is an FP8 tensor.

## API and weight preparation

L1 weights start in logical `[gate, up]` order and use compact FP32 K128 block
scales. Prepare them once, outside the timed region:

```python
fused_weights = deep_gemm.transform_l1_weights_for_swiglu(l1_weights)
```

The helper interleaves FP8 weight rows in 8-channel atoms:

```text
gate[0:8], up[0:8], gate[8:16], up[8:16], ...
```

It intentionally leaves the scale tensor unchanged. Gate and up rows originally
belong to different N128 weight-scale blocks. The fused kernel loads both scales
and applies them to alternating 8-column accumulator atoms.

Allocate FP8 output data and MN-major scales, then opt in through the existing
grouped GEMM API:

```python
output = torch.empty((m, intermediate), device="cuda", dtype=torch.float8_e4m3fn)
aligned_m = deep_gemm.get_tma_aligned_size(m, 4)
output_scale = torch.empty_strided(
    (m, intermediate // 128),
    (1, aligned_m),
    device="cuda",
    dtype=torch.float32,
)

deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
    activations,
    fused_weights,
    output,
    psum_layout,
    recipe=(1, 128, 128),
    disable_ue8m0_cast=True,
    use_psum_layout=True,
    expected_m_for_psum_layout=expected_m,
    gather_index=gather_index,
    swiglu_output_scale=output_scale,
)
```

## Kernel organization

The default specialization uses a single-CTA M128/N256/BK128 macro-tile. N256 is
the scheduling and B shared-memory shape; the kernel still issues two native
`m64n128k32` WGMMAs. Both N128 subtiles reuse the same gathered A tile and their
64-channel SwiGLU results form one local K128 quantization block.

For every K128 block, the kernel computes the two N128 subtiles sequentially. It
keeps two scaled final fragments and reuses one raw partial fragment. Each
partial is promoted with the matching gate/up weight scale before the next
subtile reuses it. This preserves the original accumulation order without a
large prescale/postscale numerical error. Although the source-level arrays total
192 floats per thread, their lifetimes do not fully overlap; ptxas keeps the
kernel spill-free.

SwiGLU values remain in the two final fragments while the CTA reduces amax over
both 64-channel halves. Math threads then quantize into a 16 KB FP8 shared tile,
followed by vectorized 16-byte global stores. The wider B tile allows four A/B
pipeline stages. For the target specialization ptxas reports:

```text
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 168 registers, used 10 barriers
```

The previous M128/N128 N2-cluster implementation remains available with
`DG_SM90_SWIGLU_BLOCK_N=128`. It uses six pipeline stages and exchanges 512 bytes
of row amax through DSM transaction barriers. Keeping it selectable provides a
direct A/B reference for the macro-tile.

Current restrictions are:

- SM90 only;
- gather index plus M-grouped contiguous psum layout;
- FP8 E4M3 output and FP32 MN-major scales;
- hidden K must be divisible by 128;
- raw L1 N (`2 * intermediate`) must be divisible by 256;
- no combine-scatter epilogue in the same invocation.

## Correctness

The reference is the existing gather-index L1 BF16 GEMM followed by a standalone
Triton SwiGLU + per-K128 FP8 quantization kernel. The fused path explicitly rounds
gate and up to BF16 before applying SwiGLU, preserving that boundary.

The tests cover multiple M/N/K and expert counts, shuffled gather indices, and
padding rows. The H200 target check runs on all eight ranks and compares:

- FP32 scale maximum absolute error;
- FP8 bin mismatch rate;
- dequantized normalized difference;
- dequantized mean and maximum absolute error.

For the target workload, the eight-rank maxima were:

```text
scale max abs          0.000e+00
FP8 mismatch rate      3.116e-07
dequantized diff        5.357e-09
dequantized mean abs    5.354e-07
dequantized max abs     9.699e+00
```

The isolated maximum comes from a very small number of FP8 bin-boundary
differences; the normalized and mean errors remain small.

## H200 performance

Environment: 8 x H200, 132 SMs per GPU. Workload per rank:

```text
tokens/rank=6976, total tokens=55808
local experts=64, local top-k=2
M(rank 0)=115584, H=2048, I=1280
```

The benchmark uses ready rank flags so scale-layout transformation is outside the
measured kernel sequences. It performs two steady-state passes in opposite order
and merges their distributions. With 5 warmups and 20 iterations per pass:

```text
gather L1 only              1052.528 us rank median
standalone SwiGLU            203.008 us rank median
baseline two-kernel path    1309.712 us rank median
N256 fused one-kernel path  1278.896 us rank median
speedup                        1.024x
```

In the immediately following run, the selectable N128 cluster path measured
1475.920 us against a 1312.128 us baseline (`0.889x`). The single-CTA macro-tile
therefore removes about 197 us from the fused kernel by eliminating DSM exchange
and duplicate gather-A loads. It turns the previous regression into a modest
2.4% end-to-end improvement over the two-kernel sequence.

An M64/N128 experiment with 264 persistent CTAs and a reduced register budget was
spill-free but measured about 1788 us, so it is not retained. An initial N256
prescale/WGMMA/postscale variant used only two fragments and had just 8 bytes of
spill, but accumulated excessive rounding error at the target K=2048
(`7.87e-5` dequantized diff and 5.25% FP8 mismatch). The retained partial-fragment
implementation restores the same target accuracy as the N128 cluster path.

## Full MoE pipeline benchmark

`tests/bench_sm90_gather_l1_swiglu_e2e.py` extends the comparison to the full
five-stage logical path:

```text
baseline: NCCL dispatch -> gather L1 -> SwiGLU -> L2 + remote scatter -> local combine
fused:    NCCL dispatch -> fused gather L1 + SwiGLU -> L2 + remote scatter -> local combine
```

The dispatch uses the repository's direct C++ `ncclAllGather` wrapper. By
default it gathers both the FP8 activation data and its per-token FP32 scales as
two collectives on the same communication stream. Routing, gather-layout
construction, weight preprocessing, and tensor allocation remain outside the
timed region. This isolates execution of an already-routed MoE layer rather than
gate or setup work.

L2 supports both existing direct-accumulator scatter epilogues. The BF16 mode
uses the N32 weight permutation, writes BF16 values to the peer CUDA-IPC buffer,
and calls `combine_reduce_slots`. The FP8 mode uses the N64 permutation, writes
E4M3 values plus one FP32 scale per 32 output columns, and calls
`combine_reduce_slots_fp8`. The timed path synchronizes all outbound peer stores
before local combine; this synchronization is required for correctness and is
included in both paths. Top-k weights are first applied by the local combine
kernel, not by either L2 epilogue.

For the target 8-H200 workload, the logical M values were
`[115584, 115200, 115584, 115328, 115328, 115072, 115456, 115328]`. The following
numbers use the maximum across per-rank medians as the critical latency.
Components are diagnostic and should not be summed: isolated stages have
different cache state and do not include the same cross-rank arrival skew.

```text
stage                                      BF16 critical    FP8 critical
NCCL dispatch                                  359.87 us       357.38 us
baseline gather L1 + SwiGLU                   1300.93 us      1305.65 us
fused gather L1 + SwiGLU                      1311.06 us      1303.89 us
L2 + remote scatter + peer completion         1511.32 us      1224.26 us
local combine                                  167.71 us       192.36 us
baseline L1/SwiGLU + L2/scatter chain         2795.88 us      2530.11 us
fused L1/SwiGLU + L2/scatter chain            2662.29 us      2459.09 us
baseline full pipeline                        3287.97 us      3040.22 us
fused full pipeline                           3136.39 us      2897.83 us
fusion speedup                                   1.048x          1.049x
critical time saved                            151.59 us       142.38 us
```

The full-only repeat measured `3317.35 us -> 3166.89 us` for BF16 (`1.048x`)
and `3023.08 us -> 2888.33 us` for FP8 (`1.047x`). FP8 makes the L2/scatter
stage about 287 us faster than BF16 in the component run. Its dequantizing local
combine is about 25 us slower, but the complete fused path is still about 239 us
faster. The reduced remote payload is the likely primary contributor.

For both scatter precisions, the integrated compute chain saves more time than
the isolated L1/SwiGLU microbenchmark. This shows that removing the large BF16
intermediate affects the immediately following L2 execution and/or cross-rank
arrival behavior. Cache residency and memory-system state are plausible
contributors, but attributing the difference precisely requires a kernel
timeline and hardware-counter profile; the benchmark only establishes the
integrated effect.

The final FP32 local-combine output was compared after running both complete
paths from the same dispatched input, routing, weights, and top-k scores. The
eight-rank aggregate metrics were as follows. Each column compares fused against
the unfused baseline using the same scatter precision; it is not a BF16-versus-
FP8 output comparison.

```text
metric                         BF16             FP8
normalized difference       3.280e-09       2.502e-08
mean absolute error         3.026e-04       4.071e-04
maximum absolute error      2.540e+00       1.074e+01
exact mismatch rate         2.341e-03       5.240e-03
reference absolute mean     9.362e+01       9.361e+01
```

The exact mismatch rates include small L1 FP8 bin-boundary changes propagated
through L2 and local combine. The normalized and mean errors remain small
relative to the output magnitude.

## Commands

Build the extension:

```bash
DG_USE_LOCAL_VERSION=0 MAX_JOBS=16 python3 setup.py build_ext --inplace
```

Run focused correctness tests:

```bash
pytest -q tests/test_sm90_gather_l1_swiglu.py -s
```

Run the original gather-L1 regression:

```bash
python3 gemm_test.py --check --modes noflags flags
```

Run the final eight-GPU correctness and performance comparison:

```bash
python3 tests/bench_sm90_gather_l1_swiglu.py \
  --num-local-ranks 8 \
  --tokens-per-rank 6976 \
  --hidden 2048 \
  --intermediate 1280 \
  --top-k 16 \
  --global-num-experts 512 \
  --experts-per-rank-token 2 \
  --warmups 5 \
  --iters 20 \
  --check
```

Run the full dispatch-through-combine comparison:

```bash
python3 tests/bench_sm90_gather_l1_swiglu_e2e.py \
  --num-local-ranks 8 \
  --tokens-per-rank 6976 \
  --hidden 2048 \
  --intermediate 1280 \
  --top-k 16 \
  --global-num-experts 512 \
  --experts-per-rank-token 2 \
  --scatter-dtype both \
  --warmups 5 \
  --iters 20 \
  --check
```

`--scatter-dtype` accepts `bf16`, `fp8`, or `both` and defaults to `both`. Use
`--no-dispatch-scales` to reproduce the older all-gather benchmark's data-only
dispatch convention, and `--no-components` to skip diagnostic stage
measurements while retaining the balanced full-path comparison.

Run the retained N128 cluster comparison by prefixing the same command with:

```bash
DG_SM90_SWIGLU_BLOCK_N=128
```

Force a clean JIT compile and inspect ptxas resources by setting a new cache
directory and `DG_JIT_PTXAS_VERBOSE=1` on the benchmark command.
