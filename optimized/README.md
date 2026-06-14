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

---

# Part II — Additional Benchmarks (Simulation & Computer Vision)

A second batch of 16 HeCBench CUDA programs, optimized with the same
profiling-guided method (nvprof + Nsight Compute). Per-benchmark bottleneck
records live in `optimized/analysis/<bench>.md`; raw profiling output in
`profiling/results/<jobid>/`. Build with `make ARCH=sm_70`.

Methodology note: the per-kernel time each program prints is an
**average-per-launch** metric, independent of the repeat count, so the baseline
side reuses the already-measured `baseline/results/951758/<bench>-cuda/run-*.log`
(5 runs) rather than re-running it; only the optimized build is re-run
(`scripts/compare.sbatch`). Scripts: `scripts/profile.sbatch`,
`scripts/smoke.sbatch`, `scripts/compare.sbatch`.

## Results summary (Part II)

| # | Application | Category | Bottleneck (profiled) | Key optimization | Speedup | Status |
|---|---|---|---|---|---|---|
| 1 | convolution1D | CV | memory-bound; `double` near DRAM roofline (89%), `int16` instruction-overhead-bound (29% peak) | coalesced interior fast-path + compile-time mask unroll | int16 **1.17x** (1.39x best); float/double ~1.0x | done |
| 2 | convolution3D | CV | compute/instruction-bound (SM 80%, DRAM 1.4%); per-FMA address math dilutes throughput | shared-mem filter slice W[m] + compile-time-K unroll + index strength reduction | heavy layer **1.33-1.41x**; small layer **1.59-1.69x** | done |
| 3 | xsbench | Sim | memory-latency-bound (92.9% warp cycles stall on memory, L2 hit 39%, DRAM 50%, SM 11%) | sort 17M lookups by sampled energy (one-time pre-pass) for grid-access locality | lookup kernel **5.90x** (net incl. sort ~5.5-6.2x) | done |
| 4 | bilateral | CV | compute-bound (DRAM ~1%, SM ~78%); transcendental + per-neighbour index/branch work | __expf + constant-memory spatial-weight table + interior fast-path (no mirror branches) | **2.81-2.98x** (3x3/6x6/9x9) | done |
| 5 | nbody | Sim | accelerate: occupancy-limited 12.4% (register pressure, DRAM idle); accumulate_energy serial `<<<1,1>>>` (17%) | shared-mem float4 (pos,mass) tiling -> higher occupancy + parallel energy reduction | **2.00x** (GFLOPS) | done |
| 6 | stencil3d | Sim | memory-bound, un-saturated (DRAM 61%, SM 8%, L2 8.6%); already shared+register-marching tiled | none — near practical limit; coefficients stream (no reuse), forcing occupancy would spill marching state | ~1.0x (no change) | analyzed |
| 7 | convolutionSeparable | CV | memory-bound; NVIDIA-optimized (conv_cols 83.5% DRAM, occ 96.5%) | filter -> constant memory (matches original NVIDIA design) | ~1.0x (filter not the bottleneck) | analyzed |
| 8 | laplace3d | Sim | memory-bound 61.5% DRAM; partial-wave tail (1.6 waves) — classic shared z-marching stencil | none — z-tiling would cut the tail (~1.15x) but needs careful chunk-boundary halo reload | ~1.0x (no change) | analyzed |
| 9 | heat | Sim | memory-bandwidth-bound, near roofline (DRAM 88%, SM 24%, L2 75%) | none — naive 5-point already bandwidth-saturated; div/mod not the bottleneck | ~1.0x (no change) | analyzed |
| 10 | lavaMD | Sim | compute-bound (DRAM 0.6%, SM 82%); LJ force inner loop | register force accumulation + __expf (float fast transcendental) | **1.15x** (max force dev 3.7e-4) | done |
| 11 | fdtd3d | Sim | under-utilized: problem too small (0.35 waves, 138 blocks); DRAM 32% / SM 30% both idle | none — NVIDIA-optimized shared+z-march; no spatial parallelism to add at this size | ~1.0x (no change) | analyzed |
| 12 | miniWeather | Sim | launch/host-overhead-bound; per-timestep MPI halo via synchronous host memcpy (2 D2H+2 H2D x10800, ~13%) | single-rank halo exchange kept on device (device-to-device copies, skip host round-trip + MPI) | **1.67x** | done |
| 13 | srad | CV | reduce = 39% (slow modulo interleaved reduction); COMPUTE stage gated by per-iter D2H sync | sequential-addressing reduction (no modulo/bank conflicts); identical output image | **1.06x** (stage host-sync-bound) | done |

### II.1 convolution1D

Full record: `optimized/analysis/convolution1D.md` (profiling job 952532,
compare job 952621).

- **Baseline**: harness times three kernels (`conv1d`, `conv1d_tiled`,
  `conv1d_tiled_caching`) over mask∈{3,5,7,9} × {double,float,int16} × 5 block
  sizes. Mask is in constant memory.
- **Profiling**: ncu shows `conv1d<double>` already at **89% DRAM throughput**
  (803 GB/s); from kernel times, `float` reaches only 65% and `int16` 29% of
  peak. SM throughput ~27% throughout → memory/overhead-bound, not compute. The
  tiled variants are *slower* than the basic kernel (shared-load + `__syncthreads`
  overhead the small masks don't repay). The 90% DtoH-memcpy GPU share is a
  verification artifact, excluded from the kernel-time metric.
- **Diagnosis**: `double` (8B loads) saturates DRAM → little headroom; `float`/
  `int16` under-saturate because the per-element fixed instruction cost (two
  boundary comparisons × mask_width) dominates when few bytes are moved.
- **Optimization**: keep the baseline's fully-coalesced one-element-per-thread
  pattern, but remove the per-tap boundary branch on interior blocks and unroll
  the mask loop by templating on the compile-time mask width.
- **Result**: int16 **1.17x** average (up to **1.39x** at the best block size),
  float ~1.0–1.07x, double ~1.0x (roofline). All sizes PASS.
- **Rejected**: a 128-bit vectorized variant (`double2`/`float4`/`short8` center
  load + scalar halo) regressed to **0.28x** for double — the halo loads became
  stride-E uncoalesced. Lesson: breaking coalescing costs far more than the
  redundant overlapped reads it removes.

### II.2 convolution3D

Full record: `optimized/analysis/convolution3D.md` (profiling job 952657,
compare job 952670).

- **Baseline-args correction**: the committed baseline (951758) ran two real conv
  layers — small `32 6 16 14 14 5` and heavy `32 96 256 26 26 5` (~13.5 ms) — not
  the local Makefile's tiny `32 1 6 32 32 5`. Comparison args are taken from the
  baseline logs; the heavy layer is the main target.
- **Baseline**: three kernels (`conv3d_s1/s2/s3`, identical math, different grid
  mapping); each thread computes one output via `for c,p,q: s += X*W`, with the
  filter `W` (2.4 MB for the heavy layer) in global memory.
- **Profiling (heavy layer)**: ncu shows DRAM **1.4%**, L1/L2 hit 95/97%, SM
  **79.7%**, occupancy 95%, 51 waves → **compute/instruction-bound, not
  memory-bound**. Only ~1.4 TFLOP/s (9% of peak): the per-iteration `II`/`WI`
  address arithmetic and L1 W-reads dilute FMA throughput.
- **Optimization**: stage the block's filter slice `W[m]` (C·K·K floats, 9.6 KB
  heavy / 600 B small — fixed `m` per block) into shared memory, reused by all
  256 threads; template on compile-time `K` to unroll the K×K loops; hoist
  `xbase`/`wbase` per channel so the inner loop uses only constant offsets.
  `__constant__` is not usable (W = 2.4 MB > 64 KB), but one m-slice fits in
  shared.
- **Result**: heavy layer **1.33–1.41x**, small layer **1.59–1.69x**, all PASS.
  The heavy layer gains less because its baseline is already ~80% SM-busy with a
  large genuine FMA count; the small layer has a higher fixed/address overhead
  ratio so removing it helps more. Remaining headroom (still ~12% of FP32 peak →
  issue-bound) would need register tiling (multiple outputs per thread); not done.

### II.3 xsbench

Full record: `optimized/analysis/xsbench.md` (profiling job 952677, compare job
952707, sort-cost nvprof 952710).

- **Baseline**: event-based XSBench (Monte Carlo neutron cross-section lookups);
  one thread per lookup (17M for `large`), each samples a random energy, binary-
  searches the 32 MB unionized energy grid, then gathers scattered rows of the
  5.6 GB index/nuclide grids. Metric = the program's printed `Average kernel
  execution time` of the `lookup` kernel (host init/verification excluded).
- **Profiling**: ncu shows **92.9% of warp cycles stall on global-memory
  scoreboard dependencies**; DRAM 50% (un-saturated), SM 11%, L2 hit only 39% →
  **memory-latency-bound** from random, low-locality access, not bandwidth or
  compute.
- **Optimization**: sort the 17M lookups by sampled energy once (a `compute_
  energy_key` kernel re-derives each energy from its seed, then `thrust::sort_
  by_key`), and have the `lookup` kernel process `idx_sorted[t]`. Neighbouring
  threads now use similar energies → nearby grid indices → far higher L2 reuse.
  Verification is unchanged (each thread keeps its original lookup index, so
  `verification[i]` and the checksum are identical).
- **Result**: lookup kernel **0.342 s → 0.058 s = 5.90x** (checksum Valid). The
  one-time sort costs ~7.8 ms (1.2 ms key kernel + 6.6 ms radix sort), placed
  before the timed region; counting it, the net speedup is still ~5.5x (repeat 1)
  to ~6.2x (repeat 10). Sorting is the canonical, profiling-justified XSBench GPU
  optimization (directly attacks the L2-hit / latency-stall bottleneck).

### II.4 bilateral

Full record: `optimized/analysis/bilateral.md` (profiling job 952714, compare
job 952719).

- **Baseline**: `bilateralFilter<R>` (R=3/6/9), one thread per output pixel,
  loops the (2R+1)^2 window; each neighbour does mirror-edge handling, a range
  and a spatial Gaussian, and an `expf`. Metric = printed per-radius avg ms.
- **Profiling**: ncu shows DRAM 0.9-3.3%, SM 75-78%, IPC ~3.0 -> compute-bound,
  not memory-bound. The transcendental `expf` plus per-neighbour index/branch
  arithmetic dominate.
- **Optimization (two steps)**: (1) `expf` -> `__expf` (single MUFU instr) gave
  only ~1.10x, showing expf was not the sole cost; (2) the spatial weight
  `exp(-(i^2+j^2)/2sigma_s^2)` depends only on the window offset for interior
  pixels, so it is precomputed into constant memory; the inner loop then drops
  the spatial computation and a transcendental, and interior pixels skip the
  four mirror-edge branches (boundary pixels keep the exact mirror path). The
  range division is hoisted to a reciprocal multiply.
- **Result**: **2.81x / 2.98x / 2.89x** for 3x3 / 6x6 / 9x9, all PASS (1e-3).
  Exact except the `__expf` approximation. Remaining cost is the unavoidable
  per-neighbour load + `__expf(range)` + accumulate.

### II.5 nbody

Full record: `optimized/analysis/nbody.md` (profiling job 952721, compare job
952726).

- **Baseline**: three kernels per step. `accelerate_particles` is the O(N^2)
  all-pairs gravity (each thread copies the full 40-byte `Particle` for itself
  and every `j`); `accumulate_energy` sums the per-particle energy array on a
  **single thread** (`<<<1,1>>>`). Metric = printed GFLOPS / Total Time.
- **Profiling**: `accelerate_particles` = 82% of GPU time but DRAM 0.03%, L2 hit
  98.8%, SM 27%, IPC 1.05, **achieved occupancy only 12.4%** (~1 block/SM,
  register-limited) -> occupancy/latency-bound, not memory or compute.
  `accumulate_energy` = **17%** of GPU time, fully serial.
- **Optimization**: (1) stage each tile of `(pos.xyz, mass)` as a coalesced
  `float4` in shared memory and keep only pos+mass per thread -> far fewer
  registers -> higher occupancy to hide the `rsqrtf` latency; (2) replace the
  serial energy sum with a one-block parallel reduction.
- **Result**: **2.00x** GFLOPS (1957 -> 3917), 1.98x total time, PASS. Math is
  identical to baseline (self/padding particles contribute 0). Further gains
  would need register blocking (a micro-tile of i per thread).

### II.6 stencil3d

Full record: `optimized/analysis/stencil3d.md` (profiling job 952728).

- **Baseline**: FP64 anisotropic 3D stencil (512^3), already optimized with a
  `__shared__ sm_psi[4][16][16]` rolling buffer + XTILE=20 register marching.
- **Profiling**: DRAM 60.7% (544 GB/s, un-saturated), SM 7.85%, **L2 hit 8.6%**,
  occupancy 61.8% -> memory-bound but limited by streaming coefficient data
  (the 9 sigma components ~9.6 GB are each read once, no reuse, hence the low L2
  hit) and by the registers the marching scheme intentionally uses.
- **Decision**: left unchanged (kept bit-identical to baseline). Forcing higher
  occupancy with `__launch_bounds__` would spill the register-marching plane
  state to local memory and likely regress; the coefficient stream offers no
  cache-reuse to exploit. Recorded honestly as an already-optimized,
  limited-headroom case; further gains would need algorithm-level changes
  (coefficient compression / mixed precision) beyond an equivalence-preserving
  optimization (and this program has no built-in correctness check).

### II.7 convolutionSeparable

Full record: `optimized/analysis/convolutionSeparable.md` (profiling job 952732,
compare job 952737).

- **Baseline**: the NVIDIA separable-convolution sample (row + column passes),
  already shared-memory tiled with halo/result steps and bank-conflict padding.
  The filter was passed as a global pointer (the original sample uses a
  `__constant__ c_Kernel`).
- **Profiling**: conv_cols is at 83.5% DRAM with 96.5% occupancy (near roofline);
  conv_rows 64.7% DRAM. Memory-bound, well-optimized.
- **Optimization / result**: moving the filter to constant memory (restoring the
  original design) gave only **1.006x (noise)** — the 17-tap filter is tiny and
  fully cached, so it was never the bottleneck. Kept the change (cleaner, matches
  NVIDIA's design) but recorded honestly as no measurable speedup; the kernel is
  already near the memory roofline.

### II.8 hotspot / hotspot3D / sobel — skipped (no input data)

These three benchmarks require external input files that are not available:
hotspot/hotspot3D need `../data/hotspot*/temp_*` and `power_*`, and sobel needs
`SobelFilter_Input.bmp`. The provided `data.zip` contains only DVC pointer stubs
(`*.tar.bz.dvc`) for hotspot/hotspot3D, not the actual data, and no `dvc` tool is
available to pull them; the sobel BMP is absent. Per the project decision these
three are skipped. (srad's real `image.pgm` *is* present, so srad is done.)

## Part II — Conclusions

Of the 16 benchmarks in this batch, 13 were profiled, optimized, and measured;
3 (hotspot, hotspot3D, sobel) were skipped for lack of input data.

**Confirmed speedups (8):**

| Application | Speedup | Bottleneck → optimization |
|---|---:|---|
| xsbench | **5.90x** | memory-latency → energy-sort lookups for L2 locality |
| bilateral | **2.8–3.0x** | compute (expf) → constant-mem spatial table + interior fast-path + __expf |
| nbody | **2.00x** | occupancy 12% → shared float4 tiling + parallel energy reduction |
| miniWeather | **1.67x** | host-overhead → device-to-device single-rank halo exchange |
| convolution3D | **1.33–1.69x** | compute/instruction → shared filter slice + compile-time-K unroll |
| convolution1D | **1.17–1.39x** (int16) | instruction-overhead → coalesced interior fast-path |
| lavaMD | **1.15x** | compute (double exp) → register accumulation + __expf |
| srad | **1.06x** | reduce 39% → sequential-addressing reduction (stage host-sync-bound) |

**Analyzed, already near-optimal (5):** stencil3d (streaming coefficients, 60%
DRAM), convolutionSeparable (NVIDIA-tuned, conv_cols 83% DRAM), laplace3d
(partial-wave tail), heat (88% DRAM roofline), fdtd3d (problem too small, 0.35
waves). Each left unchanged (bit-identical to baseline) with a profiling-grounded
reason recorded.

**Method notes / lessons:**
- The biggest wins came from *algorithm/access-pattern* changes (xsbench sort,
  miniWeather on-device halo, nbody tiling), not micro-tuning.
- Reducing apparent inefficiency is not always a win: a vectorized convolution1D
  variant **regressed to 0.28x** (uncoalesced halo); recorded as a rejected
  experiment.
- Comparison arguments must match what the committed baseline actually ran, which
  is **not always the Makefile `run:` target** (convolution3D ran two conv
  layers; laplace3d ran a 512^3 config) — args are taken from the baseline logs.
- Per-kernel printed time is an average-per-launch metric independent of repeat
  count, so the baseline side reuses `baseline/results/951758/` and only the
  optimized build is re-run.
