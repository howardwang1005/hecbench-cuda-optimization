# Profiling-Guided CUDA Optimization Report

This document summarizes the profiling evidence, optimization decisions,
implementation changes, failed experiments, correctness checks, and measured
performance for five HeCBench CUDA applications.

The optimized implementations are isolated under `optimized/`. The original
programs under `src/` remain unchanged and are used as the baseline.

## 1. Experimental Setup

### Hardware and software

- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA: 12.3
- CUDA architecture: `sm_70`
- Scheduler partition: `gtest`
- Profiling tool: `nvprof`
- Profiling job: `952187`

### Source data

- Profiling analysis: `profiling/results/952187/analysis.md`
- Raw profiling output: `profiling/results/952187/*.nvprof.txt`
- Main baseline/optimized comparison: `optimized/benchmark-952201.out`
- NW/BFS five-run comparison: `optimized/bfs-nw-v2-952209.out`

### Measurement methodology

The reported performance metric is each program's own kernel-performance
output, not Slurm wall time. Baseline and optimized programs were executed
sequentially on the same V100 with identical input and repeat arguments.
Lower execution time is better, and speedup is calculated as:

```text
speedup = baseline execution time / optimized execution time
```

The profiling run and final comparison use different repeat counts in some
applications. Profiling repeat counts were selected to expose stable kernel and
API activity. Final speedups only compare baseline and optimized executions
from the same comparison job.

| Application | Profiling command arguments | Main comparison arguments |
|---|---|---|
| all-pairs-distance | `10000` | `10000` |
| NW | `16384 10 100` | `16384 10 10` |
| Snake | `100 <dataset> 30000 1000` | `100 <dataset> 30000 100` |
| BFS | `graph1MW_6.txt` | `graph1MW_6.txt` |
| Floyd-Warshall | `1024 100 16` | `1024 10 16` |

All final optimized implementations passed their original CPU-reference or
output-comparison correctness checks.

## 2. Result Summary

| Application / metric | Baseline | Optimized | Speedup | Interpretation |
|---|---:|---:|---:|---|
| all-pairs k1 register/atomic | 193.740 us | 189.848 us | 1.02x | Noise; k1 algorithm unchanged |
| all-pairs k2 shared reduction | 159.199 us | 87.916 us | **1.81x** | Confirmed improvement |
| all-pairs k3 CUB reduction | 157.376 us | 86.527 us | **1.82x** | Confirmed improvement |
| NW total kernel time, five-run median | 21.923 ms | 20.589 ms | **1.06x** | Small but stable improvement |
| Snake aggregate across thresholds 0-25 | 6956.320 us | 3537.590 us | **1.97x** | Confirmed aggregate improvement |
| BFS total kernel time, five-run median | 963.668 us | 962.199 us | 1.002x | Measurement noise |
| Floyd-Warshall average kernel time | 9.815 ms | 1.806 ms | **5.43x** | Confirmed improvement |

All-pairs, Snake, and Floyd-Warshall values are from one paired comparison job.
NW and BFS values are medians from five paired runs. Additional repeated jobs
should be collected before reporting confidence intervals or statistical
significance.

## 3. Profiling and Source Overview

| Application | Main profiling/source observation | Optimization decision |
|---|---|---|
| all-pairs-distance | k1/k2/k3 repeatedly compute the full symmetric matrix | Compute only unique pairs for k2/k3 |
| NW | 409,400 short kernel launches and repeated tile barriers | Reduce synchronization cost and host submission overhead |
| Snake | `sneaky_snake` accounts for 99.99% of GPU activity | Reduce threshold-dependent work inside the kernel |
| BFS | Transfers dominate GPU activity; kernels are already short | Test frontier compaction and reduce redundant flag writes |
| Floyd-Warshall | 102,400 launches of a 7.906 us kernel | Replace pass-by-pass algorithm with blocked Floyd-Warshall |

The first `nvprof` pass reports GPU activity timing and CUDA API timing. It does
not directly prove whether a kernel is memory-bound, compute-bound, or
divergence-bound. Statements about internal kernel behavior below combine
profiling evidence with source-code inspection.

## 4. Application Analysis

### 4.1 All-Pairs Distance

#### Baseline behavior

The program computes a `224 x 224` distance matrix with three GPU kernels:

- `k1`: register accumulation followed by global atomic additions.
- `k2`: shared-memory tree reduction.
- `k3`: CUB block reduction.

Each block computes one pair of instances. The baseline launches
`224 * 224 = 50,176` blocks for each kernel, even though distance is symmetric:

```text
distance(x, y) = distance(y, x)
```

#### Profiling evidence

| Kernel | Strategy | Calls | Average time | GPU activity |
|---|---|---:|---:|---:|
| k1 | Registers + global atomic | 10,000 | 191.96 us | 37.03% |
| k2 | Shared-memory reduction | 10,000 | 162.26 us | 31.30% |
| k3 | CUB reduction | 10,000 | 160.22 us | 30.90% |

#### Implemented optimization

The optimized k2 and k3 kernels:

1. Map a one-dimensional block index to one upper-triangular pair.
2. Compute only pairs where `x <= y`.
3. Write the result to both `(x, y)` and `(y, x)`.

This reduces the number of blocks from:

```text
224 * 224 = 50,176
```

to:

```text
224 * 225 / 2 = 25,200
```

#### Rejected experiment

The triangular method was also tested on k1. It required atomic additions to
both symmetric output positions, and the extra atomic contention caused a large
regression. The final optimized version therefore leaves k1 unchanged.

#### Result

| Kernel | Baseline | Optimized | Speedup |
|---|---:|---:|---:|
| k1 | 193.740 us | 189.848 us | 1.02x, noise |
| k2 | 159.199 us | 87.916 us | **1.81x** |
| k3 | 157.376 us | 86.527 us | **1.82x** |

The k2/k3 results closely follow the approximately 50% reduction in pair
computations. k1 should not be reported as optimized.

### 4.2 Needleman-Wunsch

#### Baseline behavior

NW uses a tiled wavefront dynamic-programming algorithm. A tile depends on the
tile above it, the tile to its left, and the upper-left tile. The host launches
`kernel1` and `kernel2` for successive anti-diagonals.

Each block contains only 16 threads, but the tile calculation repeatedly uses
full-block barriers.

#### Profiling evidence

| Kernel | Calls | Average time | GPU activity |
|---|---:|---:|---:|
| kernel1 | 204,800 | 10.339 us | 41.75% |
| kernel2 | 204,600 | 10.361 us | 41.80% |

Additional evidence:

- Total kernel launches: `409,400`
- `cudaLaunchKernel` API time: `4.446 s`
- Launch API share: `82.28%` of reported CUDA API time

The wavefront dependency prevents arbitrary fusion because later
anti-diagonals require earlier results.

#### Implemented optimization

The final version combines two changes:

1. Capture one complete wavefront iteration in a CUDA Graph and replay it.
2. Replace repeated `__syncthreads()` calls with
   `__syncwarp(0x0000ffff)`.

Warp-level synchronization is valid here because each block contains 16
threads, all within one warp. It preserves the required shared-memory ordering
while reducing barrier overhead.

#### Rejected experiments

- Increasing tile size from 16 to 32 reduced the number of launches but
  increased per-tile work and synchronization cost, causing a regression.
- CUDA Graph replay alone produced no improvement because the GPU still
  executes the same dependent kernels.

#### Result

Five paired runs:

| Run | Baseline | Optimized | Speedup |
|---:|---:|---:|---:|
| 1 | 21.991 ms | 20.590 ms | 1.07x |
| 2 | 21.922 ms | 20.573 ms | 1.07x |
| 3 | 21.911 ms | 20.595 ms | 1.06x |
| 4 | 21.923 ms | 20.583 ms | 1.07x |
| 5 | 22.023 ms | 20.589 ms | 1.07x |
| Median | 21.923 ms | 20.589 ms | **1.06x** |

The improvement is stable but limited. A larger improvement likely requires a
persistent/cooperative kernel or a redesigned wavefront algorithm that reduces
the number of dependent kernel launches.

### 4.3 Snake

#### Baseline behavior

Snake runs the same `sneaky_snake` kernel for error thresholds 0 through 25.
Higher thresholds execute more upper/lower diagonal comparisons and more
corner-case mask construction.

#### Profiling evidence

| Kernel | Calls | Total time | Average | GPU activity |
|---|---:|---:|---:|---:|
| sneaky_snake | 26,000 | 6.971 s | 268.10 us | 99.99% |

The profiled average combines 26 substantially different workloads. Source
inspection identified loops that construct masks one bit at a time, with loop
length increasing with the error threshold.

#### Implemented optimization

- Replace repeated `set_bit` loops with constant-time bit-mask construction.
- Add a reusable `bit_range_mask` helper for corner cases.
- Remove unused variables and redundant assignments.

This reduces instruction count inside threshold-dependent paths without
changing the matching algorithm.

#### Complete threshold results

The aggregate value is the sum of the 26 reported average kernel times. It
represents the cost of executing one kernel at every threshold with equal
weight; it is not a separate metric printed by Snake.

| Threshold | Baseline (us) | Optimized (us) | Speedup |
|---:|---:|---:|---:|
| 0 | 6.89 | 7.12 | 0.97x |
| 1 | 8.87 | 9.05 | 0.98x |
| 2 | 13.22 | 13.07 | 1.01x |
| 3 | 21.39 | 18.79 | 1.14x |
| 4 | 35.65 | 29.33 | 1.22x |
| 5 | 44.48 | 35.32 | 1.26x |
| 6 | 59.37 | 43.94 | 1.35x |
| 7 | 72.72 | 53.97 | 1.35x |
| 8 | 89.51 | 65.93 | 1.36x |
| 9 | 106.08 | 74.75 | 1.42x |
| 10 | 122.26 | 82.41 | 1.48x |
| 11 | 145.82 | 91.41 | 1.60x |
| 12 | 170.90 | 101.57 | 1.68x |
| 13 | 196.23 | 114.72 | 1.71x |
| 14 | 234.56 | 128.07 | 1.83x |
| 15 | 266.30 | 139.17 | 1.91x |
| 16 | 304.98 | 165.46 | 1.84x |
| 17 | 348.22 | 176.47 | 1.97x |
| 18 | 394.49 | 189.18 | 2.09x |
| 19 | 435.60 | 215.05 | 2.03x |
| 20 | 485.12 | 230.48 | 2.10x |
| 21 | 540.83 | 247.41 | 2.19x |
| 22 | 575.53 | 254.92 | 2.26x |
| 23 | 686.70 | 291.70 | 2.35x |
| 24 | 767.43 | 370.20 | 2.07x |
| 25 | 823.17 | 388.10 | 2.12x |
| Aggregate | 6956.32 | 3537.59 | **1.97x** |

Low thresholds perform little mask-construction work, so helper overhead can
slightly outweigh the optimization. The benefit becomes substantial as the
threshold and original loop lengths increase.

### 4.4 BFS

#### Baseline behavior

The baseline uses a bitmap frontier:

1. `Kernel` scans all nodes and expands active vertices.
2. `Kernel2` scans all nodes, activates the next frontier, and writes a global
   continuation flag.

#### Profiling evidence

| Operation | Calls | Total time | GPU activity |
|---|---:|---:|---:|
| Host-to-device copies | 18 | 7.158 ms | 84.14% |
| Device-to-host copies | 13 | 590.75 us | 6.94% |
| Kernel | 12 | 522.72 us | 6.14% |
| Kernel2 | 12 | 235.39 us | 2.77% |

The program's printed kernel-time metric excludes initialization transfers.
Therefore, transfer optimization would improve end-to-end time but would not
substantially change the reported kernel metric.

#### Implemented experiment retained in final code

`Kernel2` originally allows every active thread to write the same continuation
flag. The optimized version uses a warp ballot and permits only one lane per
active warp to write the flag.

#### Rejected experiments

- A fused two-frontier implementation removed `Kernel2`, but atomic visited
  updates caused a major regression.
- A compact frontier queue processed only active vertices, but wide-frontier
  atomic contention increased kernel time to approximately `2.9 ms`.
- Increasing block size from 256 to 512 did not produce a stable improvement.

#### Result

| Metric | Baseline median | Optimized median | Ratio |
|---|---:|---:|---:|
| Total kernel execution time | 963.668 us | 962.199 us | 1.002x |

The 0.15% difference is within measurement noise and must not be reported as a
confirmed speedup. On this graph, the bitmap traversal is more suitable than
the tested queue traversal.

Potential future end-to-end directions include keeping graph data resident on
the GPU across repeated traversals, pinned host memory, and direction-optimizing
top-down/bottom-up BFS.

### 4.5 Floyd-Warshall

#### Baseline behavior

The baseline launches one full-matrix kernel for every intermediate vertex.
For a 1024-node graph, this requires 1024 sequential kernel launches per
iteration.

#### Profiling evidence

| Kernel | Calls | Average time | GPU activity |
|---|---:|---:|---:|
| floydWarshallPass | 102,400 | 7.906 us | 92.19% |

`cudaLaunchKernel` consumed `367.67 ms` across the profiling run. The high
number of short sequential launches is the dominant structural bottleneck.

#### Implemented optimization

The optimized implementation uses blocked Floyd-Warshall with `16 x 16` tiles
and three phases for each pivot tile:

1. Update the pivot tile.
2. Update all tiles in the pivot row and pivot column.
3. Update all remaining tiles using the completed row and column tiles.

The algorithm reuses tile data through shared memory and reduces launches per
iteration from:

```text
1024
```

to:

```text
3 * (1024 / 16) = 192
```

#### Result

| Baseline | Optimized | Speedup |
|---:|---:|---:|
| 9.815 ms | 1.806 ms | **5.43x** |

This is the largest measured improvement because it addresses both launch
count and repeated global-memory access.

## 5. Rejected Optimization Summary

Failed experiments are useful evidence: they show that reducing an apparent
operation count does not guarantee lower GPU execution time.

| Application | Experiment | Result | Reason |
|---|---|---|---|
| all-pairs k1 | Triangular pairs with mirrored atomics | Large regression | Two output atomics increased contention |
| NW | Tile size 32 | Regression | Higher tile work and synchronization cost |
| NW | CUDA Graph only | No improvement | Dependent kernels still execute separately |
| BFS | Fused frontier with atomic visited state | Regression | Irregular atomic contention |
| BFS | Compact frontier queue | About 2.9 ms, slower | Wide-frontier queue atomics |
| BFS | Block size 512 | No stable improvement | Baseline kernels already short |

## 6. Correctness Validation

Every retained implementation passed its application's original validation:

| Application | Validation method | Final status |
|---|---|---|
| all-pairs-distance | Compare each GPU distance matrix with CPU result | PASS |
| NW | Compare complete DP matrix with CPU NW reference | PASS |
| Snake | Compare accepted/rejected results at every threshold | PASS |
| BFS | Compare GPU node costs with CPU BFS result | Passed |
| Floyd-Warshall | Compare GPU distance matrix with CPU result | PASS |

Performance changes should only be interpreted after correctness passes. This
was especially important for blocked Floyd-Warshall and the attempted BFS
algorithm changes.

## 7. Implementation Map

| Application | Final optimized source | Main code changes |
|---|---|---|
| all-pairs-distance | `optimized/all-pairs-distance-cuda/main.cu` | Triangular pair mapping and mirrored k2/k3 output |
| NW | `optimized/nw-cuda/nw.cu` | CUDA Graph replay and warp-level synchronization |
| Snake | `optimized/snake-cuda/kernel.h` | Constant-time corner-case mask construction |
| BFS | `optimized/bfs-cuda/bfs.cu` | Warp-aggregated continuation flag |
| Floyd-Warshall | `optimized/floydwarshall-cuda/main.cu` | Three-phase blocked Floyd-Warshall |

Benchmark and validation scripts:

- `optimized/validate_optimized.sbatch`
- `optimized/benchmark_baseline_vs_optimized.sbatch`
- `optimized/benchmark_bfs_nw_v2.sbatch`

## 8. Conclusions for the Final Report

Confirmed profiling-guided improvements:

- Floyd-Warshall: **5.43x** through blocked computation, shared-memory reuse,
  and fewer launches.
- Snake: **1.97x aggregate** across thresholds by replacing threshold-dependent
  bit-setting loops.
- All-pairs-distance k2/k3: approximately **1.81x** by exploiting matrix
  symmetry.
- NW: **1.06x median** by reducing synchronization overhead within one-warp
  blocks.

No confirmed kernel-time improvement was achieved for BFS. Profiling and failed
experiments indicate that the tested graph favors the existing bitmap
traversal, while end-to-end execution is dominated by data transfers.

The report should not claim that every possible optimization has been
exhausted. It can accurately state that the major bottlenecks identified by the
first profiling pass were analyzed, targeted with implementations, and
evaluated through correctness-checked A/B experiments.

## 9. Reproduction

Build one optimized application:

```bash
cd /home/u3958285/HeCBench/optimized/<application>-cuda
module load cuda/12.3
make ARCH=sm_70
```

Run the main comparison:

```bash
cd /home/u3958285/HeCBench
sbatch optimized/benchmark_baseline_vs_optimized.sbatch
```

Run the five-pair NW/BFS comparison:

```bash
cd /home/u3958285/HeCBench
sbatch optimized/benchmark_bfs_nw_v2.sbatch
```
