# scan-cuda — Change Record (for report)

**Benchmark:** HeCBench `scan-cuda` (block-wide work-efficient Blelloch scan; Harris bank-conflict-aware variant).
**Hardware:** NVIDIA Tesla V100-SXM2-32GB (sm_70), CUDA 12.8.
**Run arguments:** `268435456 100` (256 M elements, 100 repeats). Sweeps block element-counts N = 128…2048 and element types char/short/int/long.
**Baseline source:** `math_data_compress/hecbench-src/src/scan-cuda/` (ORNL/HeCBench).
**Optimized source:** `optimized/scan-cuda/`.
**Artifacts in this folder:** `changes.patch` (full unified diff).

---

## 1. Bottleneck identified (profiling evidence)

The kernels `scan` / `scan_bcao` each scan N elements per block with N/2 threads, and
**grid-stride** over `num_blocks` logical blocks. Nsight Compute (`ncu`, run
`20260614-125000`) captured `scan<char,128>` (grid 1280 × block 64):

| Metric | Value | Reading |
|---|---:|---|
| Duration | 3.65 ms | — |
| Compute (SM) Throughput | 78 % | well-utilized per warp |
| Memory Throughput | 87 % | memory pipe busy |
| **Theoretical Occupancy** | **100 %** | no register/smem cap |
| **Achieved Occupancy** | **49.5 %** | half empty |
| **Waves Per SM** | **0.50** | only half the block slots are filled |
| Block Limit SM | 32 blocks/SM | 32·80 = 2560 slots available |

**Root cause.** The kernel is *not* idle per-warp (78 %/87 %), but the launch only
creates `grids = 16 · SM = 16·80 = 1280` blocks, while the device can hold
`32 blocks/SM · 80 SM = 2560`. So **Waves Per SM = 1280/2560 = 0.50** and achieved
occupancy is ~50 % even though theoretical is 100 %. There simply aren't enough resident
warps to hide the (inefficient, byte-granular) memory latency. The grid-stride loop means
each physical block already processes ~1638 logical blocks, so adding blocks is free of
algorithmic cost.

---

## 2. Optimization applied

**Size the grid to actually fill the device** — per kernel, per type T, per N.

| Aspect | Baseline | Optimized |
|---|---|---|
| Grid size | fixed `16 · SM` (1280) | `maxActiveBlocksPerSM · SM`, computed via `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| Occupancy (N=128) | ~50 % (0.5 waves) | targets full block-slot fill (≥ 1 wave) |
| Algorithm | unchanged | **unchanged** (only launch config) |

```cuda
auto fill_grid = [&](auto kernel) -> int {
  int maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, kernel, N/2, 0);
  int64_t g = (int64_t)maxBlocksPerSM * sm;
  if (g > num_blocks) g = num_blocks;     // never launch more blocks than work
  return (int)g;
};
dim3 grids      (fill_grid(scan<T, N>));       // scan and scan_bcao may differ
dim3 grids_bcao (fill_grid(scan_bcao<T, N>));  // (bcao uses 2x shared memory)
```

`scan_bcao` allocates `temp[2*N]` (twice the shared memory of `scan`), so its
`maxActiveBlocksPerSM` can differ — the grid is computed separately for each. The cap
`g ≤ num_blocks` avoids launching idle blocks when there is little work.

---

## 3. Files changed

Full diff: **`changes.patch`**.

| File | Change |
|---|---|
| `main.cu` | replace fixed `16*SM` grid with occupancy-filled grid per kernel; launch `scan` and `scan_bcao` with their own grid sizes |
| (others) | unchanged — `Makefile`, `CMakeLists.txt` |

No kernel/algorithm change — only the launch configuration.

---

## 4. Correctness

This change does **not** touch kernel math; it only changes how many physical blocks are
launched. The grid-stride loop `for (bid = blockIdx.x; bid < nblocks; bid += gridDim.x)`
produces identical output for any grid size, so correctness is preserved by construction.
The program's built-in `verify()` (compares against a CPU exclusive scan for every block,
type, and N) must still print `PASS` for all combinations — that is the correctness gate.

(No separate CPU harness is included because there is no algorithmic change to validate.)

---

## 5. Expected effect & honest scope (confirm with a paired run on V100)

- **Lever:** raise achieved occupancy from ~50 % (0.5 waves) toward full block-slot
  occupancy by launching enough blocks; more resident warps to hide memory latency.
- **Honest caveat:** the kernel already reports 87 % memory / 78 % compute throughput. If
  it is closer to a real bandwidth/throughput ceiling than the 164 GB/s figure suggests,
  doubling the wave count may yield a modest rather than large speedup. Of the five
  optimized kernels this is the most uncertain — it must be measured.

Measure with `268435456 100` and compare the per-(N, type) "Average execution time of
scan"; require `PASS` from the non-timing verification pass.

### Follow-ups if occupancy alone is not enough
- Replace the shared-memory Blelloch sweep (log N barriers) with a **`__shfl` warp-scan
  + small cross-warp combine**, cutting `__syncthreads()` count and shared-memory traffic.
- **Vectorized / multi-element-per-thread** global loads (esp. for char/short) to issue
  fewer, wider memory transactions — the byte-granular accesses are why "86 % memory"
  corresponds to only 164 GB/s of useful bandwidth.
- A single-pass **decoupled-lookback** scan (CUB `DeviceScan`) for the full array if the
  per-block-segment semantics allow it.
