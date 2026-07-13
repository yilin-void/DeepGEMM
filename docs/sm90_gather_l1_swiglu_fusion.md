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
both 64-channel halves. Math threads then quantize into a 16 KB FP8 shared tile.
The tile uses an eight-row XOR swizzle, after which all CTA threads perform
coalesced 16-byte shared loads and global stores. The wider B tile allows four
A/B pipeline stages. For the target specialization ptxas reports:

```text
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 168 registers, used 10 barriers
```

An earlier M128/N128 N2-cluster experiment used six pipeline stages and exchanged
512 bytes of row amax through DSM transaction barriers. Its implementation is no
longer retained in the production source; the measurements remain documented
below as the direct predecessor of the macro-tile design.

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
gather L1 only              1048.832 us rank median
standalone SwiGLU            202.832 us rank median
baseline two-kernel path    1336.464 us rank median
N256 fused one-kernel path  1175.584 us rank median
saved                        160.880 us
speedup                        1.137x
```

Three independent executions placed the optimized fused rank median in the
narrow range `1173.040-1175.584 us`; the table uses the final complete run.
The baseline two-kernel rank median varied more, from `1305.088 us` to
`1338.128 us`.

In the historical N128 cluster run, the fused path measured 1475.920 us against
a 1312.128 us baseline (`0.889x`). The initial single-CTA macro-tile reduced the
fused time to 1278.896 us by eliminating DSM exchange and duplicate gather-A
loads. Packed BF16 conversion and the conflict-free shared-output swizzle reduce
it further to 1175.584 us, turning the original regression into a 13.7%
improvement over the current two-kernel sequence.

An M64/N128 experiment with 264 persistent CTAs and a reduced register budget was
spill-free but measured about 1788 us, so it is not retained. An initial N256
prescale/WGMMA/postscale variant used only two fragments and had just 8 bytes of
spill, but accumulated excessive rounding error at the target K=2048
(`7.87e-5` dequantized diff and 5.25% FP8 mismatch). The retained partial-fragment
implementation restores the same target accuracy as the N128 cluster path.

## Single-H200 epilogue optimization

The SwiGLU epilogue was profiled separately on one H200 with the same per-GPU
amount of L1 work as the eight-GPU target. The shape was `M=115456`, `H=2048`,
`I=1280`, and 64 local experts. Each timing combines two passes in opposite
order, with 10 warmups and 50 iterations per pass.

```text
variant                                      fused median       vs. original
scalar BF16 round-trip, row-major shared      1297.616 us          1.000x
packed BF16 round-trip, row-major shared      1272.992 us          1.019x
scalar BF16 round-trip, XOR shared            1279.216 us          1.014x
packed BF16 round-trip, XOR shared            1231.968 us          1.053x
```

The packed version rounds two FP32 accumulator values to BF16 with
`__float22bfloat162_rn(make_float2(...))`, then converts the packed result back
to two FP32 registers for the existing scalar `exp`, reciprocal, multiply, and
amax sequence. This preserves the standalone BF16 L1 boundary exactly. Static
SASS changes the 128 scalar BF16 conversions into 64
`F2FP.BF16.F32.PACK_AB` instructions; the 64 `MUFU.EX2` and 72 `MUFU.RCP`
instructions are unchanged. CUDA has no corresponding packed FP32 exponential,
so `float2` only helps the BF16 conversion here.

The quantized fragment layout is not contiguous in global-memory order. The
kernel therefore keeps the existing shared-memory staging step: math threads
write packed FP8 pairs, synchronize, and all 256 math threads cooperatively issue
`LDS.128` plus `STG.E.128`. The new physical shared vector index is
`logical_vector XOR (row & 7)`; the global vector index remains logical. NCU
reported the following counters for one profiled launch:

```text
metric                                  original       packed + XOR
kernel duration                          1.44 ms             1.28 ms
shared-store bank conflicts           16,163,840               927
shared-load instructions               2,633,840         2,633,840
shared-store instructions              2,309,516         2,309,516
global-store instructions                432,960           432,960
TMA-store instructions                          0                 0
```

NCU replay duration is diagnostic rather than the benchmark latency. Its main
result is that the XOR layout removes essentially all shared-store bank
conflicts without changing global traffic. The final cubin uses 168 registers,
zero stack bytes, and zero local-memory bytes, so neither optimization adds a
spill.

A one-buffer TMA-store variant was also tested. It reduced executed global-store
instructions from 432,960 to 144,320 and shared-load instructions from 2,633,840
to 2,345,200, but required a completion wait before the shared buffer could be
reused. Interleaved comparisons ranged from a 0.8% regression to a 1.9% gain and
did not reproduce consistently. The retained implementation therefore uses the
cooperative 128-bit store path rather than TMA.

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

Baseline has five measured stages:

```text
stage                                 BF16 median/critical     FP8 median/critical
1. NCCL dispatch                         379.83 /  384.20 us     378.24 /  381.00 us
2. gather L1                            1082.23 / 1105.77 us    1080.83 / 1125.06 us
3. SwiGLU + FP8 quant                    257.64 /  305.11 us     253.10 /  312.72 us
4. L2 + remote scatter + peer sync      1554.90 / 1557.08 us    1285.60 / 1288.86 us
5. local combine                         166.20 /  181.70 us     190.09 /  206.45 us
isolated five-stage sum                 3444.51 / 3523.94 us    3205.00 / 3304.40 us
measured full path                      3296.10 / 3299.26 us    3053.68 / 3060.68 us
```

The fused path has four measured stages:

```text
stage                                 BF16 median/critical     FP8 median/critical
1. NCCL dispatch                         379.83 /  384.20 us     378.24 /  381.00 us
2. fused gather L1 + SwiGLU             1136.87 / 1204.86 us    1115.88 / 1197.79 us
3. L2 + remote scatter + peer sync      1554.90 / 1557.08 us    1285.60 / 1288.86 us
4. local combine                         166.20 /  181.70 us     190.09 /  206.45 us
isolated four-stage sum                 3248.24 / 3303.92 us    2971.98 / 3056.22 us
measured full path                      3084.89 / 3085.75 us    2844.07 / 2850.03 us
```

The authoritative full-path comparisons are therefore:

```text
scatter dtype       baseline       fused       saved      speedup
BF16               3299.26 us    3085.75 us   213.51 us    1.069x
FP8                3060.68 us    2850.03 us   210.64 us    1.074x
```

A full-only repeat measured `3289.03 us -> 3081.09 us` for BF16 (`1.067x`)
and `3067.16 us -> 2868.93 us` for FP8 (`1.069x`). This confirms that the
improvement does not depend on the cache state introduced by component timing.
FP8 makes the isolated L2/scatter stage about 268 us faster than BF16. Its
dequantizing local combine is about 25 us slower, but the complete fused path is
still about 236 us faster. The reduced remote payload is the likely primary
contributor.

Joint-sequence measurements provide the bridge between isolated stages and the
full path:

```text
sequence                                  BF16 critical    FP8 critical
baseline gather L1 + SwiGLU                  1371.58 us       1368.12 us
baseline L1/SwiGLU + L2/scatter              2835.31 us       2581.28 us
fused L1/SwiGLU + L2/scatter                 2626.72 us       2393.12 us
```

The jointly measured middle-stage delta is about 167 us for BF16 and 170 us for
FP8. Adding L2/scatter increases the delta to about 209 us and 188 us,
respectively. Removing the large BF16 intermediate therefore also affects the
immediately following L2 execution and/or cross-rank arrival behavior. Cache
residency and memory-system state are plausible contributors, but attributing
the difference precisely requires a kernel timeline and hardware-counter
profile; the benchmark only establishes the integrated effect. The isolated
stage sum is intentionally reported but is not the end-to-end latency: each
isolated measurement introduces its own stream completion, cache state, and
rank synchronization.

The stage breakdown leads to four practical conclusions:

- Removing the standalone 253-258 us median SwiGLU kernel does not translate
  into the same saving, because the fused L1 stage still performs SwiGLU, amax
  reduction, FP8 quantization, and output stores. The jointly measured
  middle-stage saving is 167-170 us.
- BF16 L2/scatter is the largest isolated stage at about 1.56 ms, roughly 44%
  of the baseline isolated-stage sum. With FP8 scatter it drops to about 1.29
  ms and is comparable to the fused L1/SwiGLU stage.
- Dispatch is about 381-384 us critical and local combine about 182 us for BF16
  or 206 us for FP8. These stages are unchanged by this fusion and limit the
  layer-level speedup.
- The final 211-214 us saving is close to the jointly measured L1-through-L2
  saving of 188-209 us. The full-only repeat retains a 198-208 us saving, so the
  layer-level improvement is reproducible without component measurements.

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

Force a clean JIT compile and inspect ptxas resources by setting a new cache
directory and `DG_JIT_PTXAS_VERBOSE=1` on the benchmark command.
