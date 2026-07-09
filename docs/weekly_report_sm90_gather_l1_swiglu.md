# 周报：SM90 Gather L1 + SwiGLU 融合

## 目标

将 Hopper MoE 的 `gather L1 + SwiGLU` 从两个 kernel 融合为一个；完整路径由
5 次逻辑阶段降为 4 次：

```text
baseline: dispatch -> gather L1 -> SwiGLU -> L2 + scatter -> local combine
fused:    dispatch -> fused gather L1 + SwiGLU -> L2 + scatter -> local combine
```

## 最终实现

- 单 CTA `M128/N256/BK128` macro-tile；复用同一份 gather A 和 N256 B tile。
- 实际 MMA 保留两组原生 `m64n128k32`，避免原生 N256 WGMMA 带来的寄存器溢出。
- gate/up FP8 weight 按 8 通道交错，K128 weight scale 保持原布局。
- 保留 baseline 的 BF16 L1 边界，再做 SwiGLU、per-K128 amax 和 E4M3 quant。
- 4-stage pipeline，16 KB FP8 shared staging；ptxas 为 168 registers、0 spill。
- L2 scatter 支持 BF16（N32 permutation）和 FP8（N64 permutation + per-32 FP32 scale）。

## 方案取舍

| 方案 | 结果 | 结论 |
|---|---:|---|
| M128/N128、2 CTA cluster + DSM amax | `1475.920 us` vs baseline `1312.128 us`，`0.889x` | DSM、重复 gather A 开销过高 |
| M64/N128、264 persistent CTA | 约 `1788 us`，0 spill | tile 过小，调度/访存开销上升 |
| N256 prescale/WGMMA/postscale | 8-byte spill；dequant diff `7.87e-5`；FP8 mismatch `5.25%` | 精度不可接受 |
| 最终 M128/N256 macro-tile | `1278.896 us` vs baseline `1309.712 us`，`1.024x` | 保留；精度与资源均满足要求 |

## 测试口径

```text
GPU                  8 x H200，132 SM/GPU
tokens/rank          6976（total 55808）
H / I                2048 / 1280
global/local experts 512 / 64
global/local top-k   16 / 2
M/rank               [115584, 115200, 115584, 115328,
                      115328, 115072, 115456, 115328]
dispatch             C++ NCCL all-gather，FP8 data + FP32 scale
payload/rank         14.733 MB，NCCL CTAs=64
timing               5 warmups，20 iterations；正反顺序各一轮
```

每个 rank 先对迭代取 median；下表统一报告这些 rank median 的最大值
（`critical`），因为 collective 和跨 rank 同步的完整路径由最慢 rank 决定。
独立阶段各自包含 stream completion/同步，其求和仅用于拆解，不能替代真实
整链计时。

## 测试结果

### 1. Isolated L1 + SwiGLU

| 测试 | Gather L1 | SwiGLU | Baseline 2 kernels | Fused 1 kernel | 加速比 |
|---|---:|---:|---:|---:|---:|
| 最终测试 | `1052.528 us` | `203.008 us` | `1309.712 us` | `1278.896 us` | `1.024x` |

中间 FP8 输出正确性（8-rank max）：

```text
scale max abs          0.000e+00
FP8 mismatch rate      3.116e-07
dequantized diff       5.357e-09
dequantized mean abs   5.354e-07
dequantized max abs    9.699e+00
```

### 2. Full MoE：Baseline 5 阶段

| 阶段 | BF16 critical | FP8 critical |
|---|---:|---:|
| 1. NCCL dispatch | `357.92 us` | `358.54 us` |
| 2. Gather L1 | `1091.79 us` | `1094.59 us` |
| 3. SwiGLU + FP8 quant | `274.31 us` | `274.45 us` |
| 4. L2 + remote scatter + peer sync | `1506.95 us` | `1230.36 us` |
| 5. Local combine | `166.61 us` | `191.98 us` |
| 独立阶段之和 | `3395.74 us` | `3143.48 us` |
| **真实完整路径** | **`3290.38 us`** | **`3049.12 us`** |

### 3. Full MoE：Fused 4 阶段

| 阶段 | BF16 critical | FP8 critical |
|---|---:|---:|
| 1. NCCL dispatch | `357.92 us` | `358.54 us` |
| 2. Fused gather L1 + SwiGLU | `1261.78 us` | `1258.03 us` |
| 3. L2 + remote scatter + peer sync | `1506.95 us` | `1230.36 us` |
| 4. Local combine | `166.61 us` | `191.98 us` |
| 独立阶段之和 | `3291.42 us` | `3029.43 us` |
| **真实完整路径** | **`3140.55 us`** | **`2905.00 us`** |

完整路径结论：

| Scatter | Baseline | Fused | 节省 | 加速比 |
|---|---:|---:|---:|---:|
| BF16 | `3290.38 us` | `3140.55 us` | `149.83 us` | `1.048x` |
| FP8 | `3049.12 us` | `2905.00 us` | `144.12 us` | `1.050x` |

联合序列（critical）：

| 序列 | BF16 | FP8 |
|---|---:|---:|
| Baseline gather L1 + SwiGLU | `1320.14 us` | `1330.08 us` |
| Baseline L1/SwiGLU + L2/scatter | `2810.35 us` | `2555.61 us` |
| Fused L1/SwiGLU + L2/scatter | `2667.56 us` | `2454.54 us` |

### 4. 完整输出正确性

以下均为 fused 对同 scatter 精度 baseline，并非 BF16 与 FP8 互比：

| 指标 | BF16 | FP8 |
|---|---:|---:|
| Normalized difference | `3.280e-09` | `2.502e-08` |
| Mean absolute error | `3.026e-04` | `4.071e-04` |
| Maximum absolute error | `2.540e+00` | `1.074e+01` |
| Exact mismatch rate | `2.341e-03` | `5.240e-03` |
| Reference absolute mean | `9.362e+01` | `9.361e+01` |

补充验证：

```text
focused pytest             4 passed
gather L1 BF16 ref diff    0.00068398 < 0.001
noflags / flags            与 plain bitwise equal
padding rows               全 0
ptxas                      168 registers，10 barriers，0 stack/0 spill
```

## 结论

- 融合有效但收益有限：isolated 提升 `1.024x`；完整层在 BF16/FP8 scatter
  下分别提升 `1.048x` 和 `1.050x`。
- 独立 SwiGLU 为 `249-256 us`，但融合 kernel 自身仍承担激活、amax、quant
  和 store，联合中间阶段仅净省 `58-72 us`。
- 加入 L2/scatter 后净省扩大到 BF16 `143 us`、FP8 `101 us`，说明移除 BF16
  中间张量还改善了后续缓存/内存状态或跨 rank 到达行为。
- BF16 最大瓶颈是 L2/scatter（约 `1.51 ms`）；FP8 将其降至约 `1.23 ms`，
  虽然 local combine 慢约 `25 us`，完整 fused 路径仍比 BF16 快约 `236 us`。
- dispatch（约 `356-359 us`）和 combine（约 `167/192 us`）未被本次融合覆盖，
  是继续提升 layer-level speedup 的固定开销。

## 复现命令

```bash
pytest -q tests/test_sm90_gather_l1_swiglu.py -s
python3 gemm_test.py --check --modes noflags flags

python3 tests/bench_sm90_gather_l1_swiglu.py \
  --num-local-ranks 8 --tokens-per-rank 6976 \
  --hidden 2048 --intermediate 1280 --top-k 16 \
  --global-num-experts 512 --experts-per-rank-token 2 \
  --warmups 5 --iters 20 --check

python3 tests/bench_sm90_gather_l1_swiglu_e2e.py \
  --num-local-ranks 8 --tokens-per-rank 6976 \
  --hidden 2048 --intermediate 1280 --top-k 16 \
  --global-num-experts 512 --experts-per-rank-token 2 \
  --scatter-dtype both --warmups 5 --iters 20 --check
```
