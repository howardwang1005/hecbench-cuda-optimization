# jaccard-cuda — Change Record (for report)

**Benchmark:** HeCBench `jaccard-cuda` (Jaccard similarity weights on a sparse CSR graph).
**Hardware:** NVIDIA Tesla V100-SXM2-32GB (sm_70), CUDA 12.8.
**Run arguments:** `1024 512 1000` (rows × cols × iterations; random `rand()%10` → ~90% dense).
**Baseline source:** `math_data_compress/hecbench-src/src/jaccard-cuda/` (ORNL/HeCBench).
**Optimized source:** `optimized/jaccard-cuda/`.
**Artifacts in this folder:** `changes.patch` (full unified diff), `intersection_correctness_test.cpp` (CPU verification harness).

---

## 1. Bottleneck identified (profiling evidence)

`jaccard_weight` launches five kernels per iteration (`fill×2`, `jaccard_row_sum`,
`jaccard_is_opt`, `jaccard_jw`). Nsight Compute (`ncu`, run `20260614-125000`) shows one
kernel dominates everything:

| Kernel | Duration | Compute% | Memory% | DRAM% | Occupancy% |
|---|---:|---:|---:|---:|---:|
| `jaccard_row_sum` | 36 us | 11 | 8 | — | 10 |
| **`jaccard_is_opt`** | **7.75 ms** | **49** | 34 | 0.1 | **20** |
| `jaccard_jw` | 11 us | 13 | 65 | — | 79 |

`jaccard_is_opt` is ~99 % of GPU time. DRAM is 0.1 % (the CSR fits in L2), so it is
**compute/latency-bound**, not bandwidth-bound.

**Root cause.** The kernel is launched with block `(x=8, y=4, z=8)` = 256 threads, but
its intersection work runs entirely under `if (threadIdx.x == 0)` — a **serial
two-pointer merge** of two sorted CSR rows. So **7 of every 8 lanes (the whole x
dimension) sit idle**, and each active lane walks `O(Ni+Nj)` elements serially. With the
~90 %-dense input each row has ~460 nnz, so every one of the ~471 K nnz drives a ~920-step
serial scan on a single lane → the 7.75 ms and the 20 % occupancy.

---

## 2. Optimization applied

**Parallelize the intersection across the idle `threadIdx.x` lanes.**

| Aspect | Baseline (`is_opt`) | Optimized |
|---|---|---|
| Active lanes per j | 1 (`threadIdx.x==0`) | all `blockDim.x` (8) |
| Intersection method | serial two-pointer merge, `O(Ni+Nj)` on one lane | each lane strides the reference row, **binary-searches** in the current row, `O((Ni/8)·log Nj)` per lane |
| Combine | single lane's running sum | `__shfl_down_sync` tree reduction over the x-group, then one `atomicAdd` |
| `weight_s[j]` write | implicit (x==0 block) | explicitly guarded to `threadIdx.x==0` (avoids 8× redundant stores) |

```cuda
// all 8 x-lanes cooperate; each handles a strided slice of the reference row
T local_sum = 0;
for (int i = ref_beg + threadIdx.x; i < ref_end; i += blockDim.x) {
  int ref_col = csrInd[i];
  // binary search ref_col in the sorted current row -> match?
  if (found) local_sum += weighted ? v[ref_col] : (T)1.0;
}
// reduce across exactly this x-group's lanes, then one atomic
unsigned grp_mask = ((1u<<blockDim.x)-1u) << (blockDim.x*threadIdx.y);
for (int off = blockDim.x>>1; off>0; off>>=1)
  local_sum += __shfl_down_sync(grp_mask, local_sum, off, blockDim.x);
if (threadIdx.x == 0 && local_sum != 0) atomicAdd(&weight_i[j], local_sum);
```

### Correctness subtlety — the shfl mask

The x-lanes for a fixed `threadIdx.y` are contiguous in the warp
(`warp lane = threadIdx.x + blockDim.x·threadIdx.y`). But different `y` groups iterate
the inner `j` loop a different number of times, so they are **not** all active at the same
shfl call. Naming the full warp (`0xFFFFFFFF`) would be undefined. The fix builds a mask
covering **exactly this x-group's lanes**:
`grp_mask = ((1<<blockDim.x)-1) << (blockDim.x·threadIdx.y)`.

### Honest note on work vs parallelism
Binary search does more total comparisons than the two-pointer merge
(`Ni·log Nj` vs `Ni+Nj`), but that work is now split across 8 lanes and the win comes
from filling the idle lanes / raising occupancy, not from doing less work. The actual
speedup must be measured on the V100 (see §5).

---

## 3. Files changed

Full diff: **`changes.patch`**. Only `jaccard_is_opt` and its `weight_s` write changed;
the other kernels, the host driver, and launch configuration are untouched.

| File | Change |
|---|---|
| `main.cu` | rewrite `jaccard_is_opt` intersection from serial two-pointer (1 lane) to lane-parallel binary search + shfl reduction (8 lanes); guard `weight_s[j]` to lane 0 |
| (others) | unchanged — `Makefile`, `CMakeLists.txt` |

---

## 4. Correctness verification

CPU harness `intersection_correctness_test.cpp` models the exact 8-lane decomposition
(strided reference-row split, binary search, shfl-down tree reduction) and compares the
accumulated `weight_i` against the baseline serial two-pointer merge on a CSR built the
same way as `main.cu` (`rand()%10`, weighted `v[col]=(col+1)/e`):

```
n=200 e=26973  max|new - ref| = 0.000e+00  OK
```

Exact match. The program's own verification is only compiled under `-DDEBUG` (checks six
known `weight_j` values to 1e-5); the default run prints timing and relies on the
algorithm being identical, which the CPU test confirms.

Reproduce: `g++ -O2 -std=c++17 intersection_correctness_test.cpp -o t && ./t`

---

## 5. Expected effect (confirm with a paired run on V100)

- The dominant kernel uses **8× the lanes** (1 → 8 active per j); occupancy should rise
  from 20 % and the 7.75 ms should drop.
- Because the kernel is ~99 % of GPU time, its speedup ≈ the program's speedup.

Measure by running baseline and optimized back-to-back with `1024 512 1000` and comparing
"Average execution time of kernels" (both the weighted and unweighted passes). For a hard
correctness gate, build both with `-DDEBUG` and confirm `PASS`.

### Follow-ups if still compute/occupancy-limited
- Tune the block shape (e.g. larger `blockDim.x`, or one warp per j) to map the
  reduction onto a full warp and raise active-warp count.
- Balance work per nnz rather than per row (long rows dominate); consider a
  segmented / load-balanced assignment of (row,j) pairs to warps.
- For very dense rows, a shared-memory hashed membership test can beat repeated binary
  searches.
