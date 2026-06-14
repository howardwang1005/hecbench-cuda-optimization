# bscan-cuda — Change Record (for report)

**Benchmark:** HeCBench `bscan-cuda` (block-wide binary prefix sum; Harris & Garland, GPU Computing Gems).
**Hardware:** NVIDIA Tesla V100-SXM2-32GB (sm_70), CUDA 12.8.
**Run arguments:** `1000` (repeat count). The program sweeps block sizes N = 32, 64, 128, 256, 512, 1024 internally, each over `grid = 12·7·8·9·10 = 60480` blocks.
**Baseline source:** `math_data_compress/hecbench-src/src/bscan-cuda/` (ORNL/HeCBench).
**Optimized source:** `optimized/bscan-cuda/`.
**Artifacts in this folder:** `changes.patch` (full unified diff), `single_warp_scan_test.cpp` (CPU verification harness).

---

## 1. Bottleneck identified (profiling evidence)

Each block does an independent block-wide binary prefix sum of N elements via
`block_binary_prefix_sums` (per-warp ballot/popc scan, then a shared-memory combine of
warp partial sums). Nsight Compute (`ncu`, run `20260614-125000`) captured the **N=32**
instance (`binary_scan` at grid 60480 × block 32):

| Metric | Value | Reading |
|---|---:|---|
| Duration | 199 us | — |
| Block Size | 32 (1 warp) | — |
| Registers/Thread | 16 | not the limiter |
| **Block Limit Shared Mem** | **32 blocks/SM** | the occupancy limiter |
| Block Limit Warps | 64 | what we want to reach |
| Theoretical Occupancy | 50 % | capped by shared mem |
| Achieved Occupancy | 45.8 % | — |

**Root cause.** For N = 32 the block is a single warp, yet `block_binary_prefix_sums`
still runs the full multi-warp machinery: it declares `__shared__ int sdata[64]`, writes
warp partial sums, runs a second `warp_scan` over them, and executes **two
`__syncthreads()`**. The static shared-memory allocation makes the occupancy limiter
**Block Limit Shared Mem = 32 blocks/SM** (→ 50 % occupancy), and the two block barriers
are pure overhead — for one warp there is nothing to combine across warps.

---

## 2. Optimization applied

**Specialize the single-warp case (N ≤ 32) at compile time.** Template
`block_binary_prefix_sums` / `binary_scan` on the block size N and, for N ≤ 32, return
the per-warp scan result directly.

| Aspect | Baseline (N=32) | Optimized (N=32) |
|---|---|---|
| Shared memory | `__shared__ int sdata[64]` declared & used | **none** (not even declared, via `if constexpr`) |
| `__syncthreads()` | 2 | **0** |
| Cross-warp combine | warp_scan over partial sums | skipped (single warp = whole block) |
| Occupancy limiter | Block Limit Shared Mem = 32 → 50 % | Block Limit Warps = 64 → up to 100 % |

```cuda
template <int N>
__device__ __inline__ int block_binary_prefix_sums(int x) {
  bool predicate = valid(x);
  int warpPrefix = binary_warp_scan(predicate);     // ballot + popc(lanemask_lt)
  if constexpr (N <= 32) {
    return warpPrefix;                               // one warp: this IS the block scan
  } else {
    __shared__ int sdata[64];                        // multi-warp path unchanged
    ... two __syncthreads(), warp_scan over partials ...
    return warpPrefix + sdata[warpIdx];
  }
}
```

`if constexpr` guarantees the N=32 instantiation declares **zero static shared memory**,
which is what lifts the Block Limit Shared Mem cap. The N = 64…1024 instances are
unchanged (they genuinely need the cross-warp combine).

### Why returning `warpPrefix` is exact
For a single warp, the baseline computes `warpPrefix + sdata[warpIdx]` where, after the
section-C `warp_scan`, `sdata[0]` (warp 0's exclusive partial-sum prefix) is `0`. So the
baseline already returns `warpPrefix + 0`. The specialization returns the same value
without the shared-memory round trip or barriers.

---

## 3. Files changed

Full diff: **`changes.patch`**.

| File | Change |
|---|---|
| `main.cu` | template `block_binary_prefix_sums<N>` and `binary_scan<N>` on block size; add `if constexpr (N<=32)` fast path; launch `binary_scan<N>` |
| (others) | unchanged — `Makefile`, `CMakeLists.txt` |

---

## 4. Correctness verification

CPU harness `single_warp_scan_test.cpp` confirms the single-warp identity the fast path
relies on: the exclusive binary prefix sum equals `popc(ballot(p) & lanemask_lt(lane))`
for every lane:

```
W=32 trial=0..4  OK
```

The program's own per-run check (exclusive-scan reference in `bscan<N>`) must still print
`verify = PASS` for all six block sizes; the unchanged multi-warp path keeps N>32 correct,
and the N=32 fast path is value-identical to the baseline (see §2).

Reproduce: `g++ -O2 -std=c++17 single_warp_scan_test.cpp -o t && ./t`

---

## 5. Expected effect & honest scope (confirm with a paired run on V100)

- **Scope:** this change helps only the **N = 32** case — occupancy 50 % → up to 100 %,
  two `__syncthreads()` removed, no shared-memory traffic. N = 64…1024 are unchanged.
- N=32 is already the cheapest case (174 us); the larger blocks (256/512/1024 at ~676 us)
  dominate total runtime and are not addressed here.

Measure with `1000` and compare the per-block-size "Average execution time" / "Billion
elements per second"; require `verify = PASS` on all sizes.

### Follow-ups for the multi-warp cases (where the time actually is)
- The N≥256 blocks spend two barriers plus a serial `warp_scan` over up to 32 partial
  sums; replacing section C with a `__shfl`-based warp scan (no shared memory) would cut
  the combine cost and shared-memory pressure for the large blocks too.
- Consider processing multiple elements per thread (vectorized loads) so fewer, larger
  blocks cover the same 60480·N elements with better memory efficiency.
