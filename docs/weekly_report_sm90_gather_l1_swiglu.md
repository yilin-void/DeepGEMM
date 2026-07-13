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
- BF16 round-trip 使用 packed `bfloat162` 转换；FP8 shared tile 使用按行 XOR
  swizzle，将 shared-store bank conflict 从约 1616 万降至 927。
- 4-stage pipeline，16 KB FP8 shared staging；ptxas 为 168 registers、0 spill。
- L2 scatter 支持 BF16（N32 permutation）和 FP8（N64 permutation + per-32 FP32 scale）。

## 方案取舍

| 方案 | 结果 | 结论 |
|---|---:|---|
| M128/N128、2 CTA cluster + DSM amax | `1475.920 us` vs baseline `1312.128 us`，`0.889x` | DSM、重复 gather A 开销过高 |
| M64/N128、264 persistent CTA | 约 `1788 us`，0 spill | tile 过小，调度/访存开销上升 |
| N256 prescale/WGMMA/postscale | 8-byte spill；dequant diff `7.87e-5`；FP8 mismatch `5.25%` | 精度不可接受 |
| 最终 M128/N256 macro-tile + epilogue 优化 | `1175.584 us` vs baseline `1336.464 us`，`1.137x` | 保留；精度与资源均满足要求 |

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

每个 rank 先对迭代取 median。Isolated L1 表沿用 benchmark 输出，报告各
rank median 的中位数；Full MoE 表报告各 rank median 的最大值（`critical`），
因为 collective 和跨 rank 同步的完整路径由最慢 rank 决定。独立阶段各自
包含 stream completion/同步，其求和仅用于拆解，不能替代真实整链计时。

## 测试结果

### 1. Isolated L1 + SwiGLU

| 测试 | Gather L1 | SwiGLU | Baseline 2 kernels | Fused 1 kernel | 加速比 |
|---|---:|---:|---:|---:|---:|
| 最终测试 | `1048.832 us` | `202.832 us` | `1336.464 us` | `1175.584 us` | `1.137x` |

三次独立执行的 fused rank median 为 `1173.040 / 1173.136 / 1175.584 us`；
上表保留最后一轮完整数据。

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
| 1. NCCL dispatch | `384.20 us` | `381.00 us` |
| 2. Gather L1 | `1105.77 us` | `1125.06 us` |
| 3. SwiGLU + FP8 quant | `305.11 us` | `312.72 us` |
| 4. L2 + remote scatter + peer sync | `1557.08 us` | `1288.86 us` |
| 5. Local combine | `181.70 us` | `206.45 us` |
| 独立阶段之和 | `3523.94 us` | `3304.40 us` |
| **真实完整路径** | **`3299.26 us`** | **`3060.68 us`** |

### 3. Full MoE：Fused 4 阶段

| 阶段 | BF16 critical | FP8 critical |
|---|---:|---:|
| 1. NCCL dispatch | `384.20 us` | `381.00 us` |
| 2. Fused gather L1 + SwiGLU | `1204.86 us` | `1197.79 us` |
| 3. L2 + remote scatter + peer sync | `1557.08 us` | `1288.86 us` |
| 4. Local combine | `181.70 us` | `206.45 us` |
| 独立阶段之和 | `3303.92 us` | `3056.22 us` |
| **真实完整路径** | **`3085.75 us`** | **`2850.03 us`** |

完整路径结论：

| Scatter | Baseline | Fused | 节省 | 加速比 |
|---|---:|---:|---:|---:|
| BF16 | `3299.26 us` | `3085.75 us` | `213.51 us` | `1.069x` |
| FP8 | `3060.68 us` | `2850.03 us` | `210.64 us` | `1.074x` |

联合序列（critical）：

| 序列 | BF16 | FP8 |
|---|---:|---:|
| Baseline gather L1 + SwiGLU | `1371.58 us` | `1368.12 us` |
| Baseline L1/SwiGLU + L2/scatter | `2835.31 us` | `2581.28 us` |
| Fused L1/SwiGLU + L2/scatter | `2626.72 us` | `2393.12 us` |

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

- packed BF16 转换和 shared XOR swizzle 将 isolated 融合收益从原来的
  `1.024x` 提高到 `1.137x`；fused kernel 三次执行稳定在 `1173-1176 us`。
- 完整层在 BF16/FP8 scatter 下分别提升 `1.069x` 和 `1.074x`，关键路径净省
  `213.51 us` 和 `210.64 us`。full-only 复测仍分别节省 `207.94 us` 和
  `198.23 us`。
- 独立 SwiGLU median 为 `253-258 us`，融合 kernel 自身仍承担激活、amax、
  quant 和 store；联合中间阶段实际净省 `167-170 us`。
- 加入 L2/scatter 后联合序列净省扩大到 BF16 `209 us`、FP8 `188 us`，说明
  移除 BF16 中间张量还影响了后续缓存/内存状态或跨 rank 到达行为。
- BF16 最大瓶颈是 L2/scatter（约 `1.56 ms`）；FP8 将其降至约 `1.29 ms`。
  dispatch（约 `381-384 us` critical）和 combine（约 `182/206 us`）仍是
  本次融合没有覆盖的固定开销。

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
