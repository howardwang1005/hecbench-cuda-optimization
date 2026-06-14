# gaussian-cuda — Change Record (for report)

**Benchmark:** HeCBench `gaussian-cuda` (dense Gaussian elimination, forward substitution on GPU).
**Hardware:** NVIDIA Tesla V100-SXM2-32GB (sm_70), CUDA 12.8.
**Run arguments:** `-q -t -s 4096` (4096 × 4096 matrix generated internally).
**Baseline source:** `math_data_compress/hecbench-src/src/gaussian-cuda/` (ORNL/HeCBench).
**Optimized source:** `optimized/gaussian-cuda/`.
**Artifacts in this folder:** `changes.patch` (full unified diff), `fusion_correctness_test.cpp` (CPU verification harness).

---

## 1. Bottleneck identified (profiling evidence)

Profiled with Nsight Compute (`ncu`), run `20260614-125000`. The forward-substitution
loop launches **two** kernels per elimination step `t`, for `t = 0 … size-2`:

```
for (t = 0; t < size-1; t++) { fan1<<<...>>>(); fan2<<<...>>>(); }   // 2*(size-1) launches
```

For `size=4096` that is **8190 kernel launches**. Per-kernel metrics:

| Kernel | Grid×Block | Duration | Compute% | Memory% | Occupancy% | Reading |
|---|---|---:|---:|---:|---:|---|
| `fan1` | 16 × 256 | 3.65 us | **0.3** | 7.7 | **11** | pure launch overhead, does almost nothing |
| `fan2` | (256,256) × (16,16) | 196 us | 21 | **76** | 87 | the real work, healthy (memory-bound) |

Program's own timing (`run.log`):

```
Total kernel execution time  499336.522 us   (sum of kernel durations)
Device offloading time       949838.917 us   (wall time around the launch loop)
```

**Root cause.** `fan1` is a near-empty kernel launched ~4095 times: it only computes
the column multiplier `m[row][t] = a[row][t] / a[t][t]`, which `fan2` consumes in the
same step. Each `fan1` runs on a grid of 16 blocks at 11% occupancy and 0.3% compute —
it is essentially **launch latency**. The ~450 ms gap between "device offloading"
(950 ms) and "total kernel" (499 ms) is dominated by host-side submission overhead from
the 8190 launches. This is the same launch-overhead shape as the Floyd-Warshall case
(which reached 5.43× after the analogous fix).

---

## 2. Optimization applied

**Fuse `fan1` into `fan2`** — compute the multiplier inline, delete the separate `fan1`
kernel and the entire device `m` array.

| Aspect | Baseline | Optimized |
|---|---|---|
| Kernels per step | `fan1` + `fan2` | one fused `fan2_fused` |
| Total launches (size=4096) | **8190** | **4095** (halved) |
| Multiplier `m[row][t]` | written by fan1 to global `m`, read by fan2 | computed in a register inside fan2 |
| Device `m` array | malloc + H2D + D2H of size² floats | **removed** (and its memcpys) |

### Race-free fusion (the subtle part)

Naively reading `a[row][t]` inside fan2 to form the multiplier is a **data race**: the
column-0 thread writes `a[row][t]` (the y=0 store) while other threads of the same row
still need to read it → wrong multiplier. CPU emulation of that ordering diverges from
the reference by 3.6e+01 (a real bug).

Fix: **no thread writes the pivot column `a[row][t]`** — the column loop effectively
starts at `globalIdy = 1`, and the `globalIdy == 0` thread only performs the `b` update.
Every thread then reads the *original, untouched* `a[row][t]`, so all multipliers are
consistent — no race. The eliminated lower-triangle entry `a[row][t]` is mathematically
~0 and is never read again (`BackSub` reads only the upper triangle), so skipping the
write is exact.

```cuda
// fan2_fused: one thread per (row, column)
const int row  = globalIdx + 1 + t;
const float mult = a[size*row + t] / a[size*t + t];   // original pivot column, untouched
if (globalIdy == 0) b[row] -= mult * b[t];            // pivot col: skip a-write, do b
else a[size*row + (globalIdy+t)] -= mult * a[size*t + (globalIdy+t)];
```

---

## 3. Files changed

Full diff: **`changes.patch`**.

| File | Change |
|---|---|
| `gaussianElim.cu` | replace `fan1`+`fan2` with single `fan2_fused`; `ForwardSub` drops the fan1 launch, the `d_m` malloc/memcpys, and launches only the fused kernel |
| (others) | unchanged — `gaussianElim.h`, `utils.cu/.h`, `Makefile`, `CMakeLists.txt` |

`BLOCK_SIZE_0` is retained (still used by the startup banner). Host `m` is left
untouched and unused by verification (the host computes its own reference and compares
only `finalVec`).

---

## 4. Correctness verification

CPU harness `fusion_correctness_test.cpp` reproduces the fused kernel's exact arithmetic
(skip-pivot-column variant), runs forward sub + back sub, and compares `finalVec` to the
HeCBench reference:

```
finalVec max diff (fused vs ref) = 0.000e+00  OK   (size=128)
```

It also demonstrates the *unsafe* inline variant fails (diff 3.6e+01), documenting why
the skip-pivot-column rule is required. On device, the program's own check
(`fabsf(finalVec[i] - host[i]) > 1e-3`) should still print `PASS`.

Reproduce: `g++ -O2 -std=c++17 fusion_correctness_test.cpp -o t && ./t`

---

## 5. Expected effect (confirm with a paired run on V100)

- **Kernel launches halved** (8190 → 4095); the ~450 ms of host-submission overhead
  (the device-offload vs total-kernel gap) should shrink substantially.
- One memory-bound kernel per step instead of one near-empty + one memory-bound kernel.
- Device `m` array and its H2D/D2H copies removed.

Measure the headline speedup by running baseline and optimized back-to-back on the same
V100 with `-q -t -s 4096` and comparing "Total kernel execution time" and "Device
offloading time"; require `PASS`.

### Follow-ups if launch overhead still dominates
- **CUDA Graphs**: capture the size-1 iterations once and replay, removing per-launch
  CPU submission cost entirely (the grid shape changes with `t`, so use a graph with
  updatable kernel-node params, or a persistent-kernel/megakernel that loops over `t`
  internally with a grid-wide barrier via cooperative groups).
- Skip trailing tiny iterations on the host or shrink the fan2 grid to the live submatrix
  to cut wasted blocks late in the elimination.
