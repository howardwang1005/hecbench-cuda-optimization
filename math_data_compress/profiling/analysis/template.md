# Bottleneck Analysis — 9 Target Benchmarks (V100, sm_70)

How to read this file: each benchmark has a **prior hypothesis** (from reading the
source) and the **profiler metrics that confirm or refute it**. After you run
`profile-all.sh` and send results back, the "Measured" column gets filled and the
optimization direction is locked.

## How to classify a kernel (the one rule that matters)

From `ncu` **SpeedOfLight (SOL)** section, look at two numbers:

| SOL Compute % | SOL Memory % | Verdict | Optimize by |
|---|---|---|---|
| low | **high (>70%)** | **memory-bound** | wider loads (float4/int4), coalescing, fewer global atomics, shared-mem privatization |
| **high (>70%)** | low | **compute-bound** | reduce work, better ILP, faster math, warp primitives |
| low | low | **latency / occupancy-bound** | raise occupancy, more in-flight work, cut dependency chains, fewer launches |

Plus from `nsys gpukernsum`: **which kernel owns the time** (don't optimize a kernel
that's 3% of runtime), and `gpumemtimesum`: **is it actually memcpy-dominated** (then
the kernel isn't the target at all — overlap/pinned memory is).

---

## Reduction / scan family (expect memory-bound + atomic/sync contention)

### atomicReduction-cuda  — `./main` (no args)
- **Source:** `kernels.h` — strided sum loop + one `atomicAdd(out, sum)` per thread.
  v2/v4/v8/v16 already unroll the load width. Output is GB/s (higher better).
- **Prior:** memory-bound; baseline already hits ~850 GB/s (V100 peak ≈ 900). Limited
  headroom on bandwidth, but the per-thread `atomicAdd` to a single global counter is
  a contention point at high block counts.
- **Confirm with:** SOL Memory % (expect high), DRAM throughput vs peak, and whether
  `atomicAdd` serialization shows in MemoryWorkloadAnalysis (L2 atomic traffic).
- **Likely win:** block-level reduction (shared mem or `__reduce_add_sync`/CUB) so each
  *block* does one atomicAdd instead of each thread. Beating ~850 GB/s is hard — be honest.

### scan-cuda  — `268435456 100`
- **Source:** Blelloch work-efficient scan (`scan_bcao` with bank-conflict avoidance,
  and plain `scan`). `__shared__ temp[2*N]`, two sweep loops with `__syncthreads`.
- **Prior:** memory-bound at the device level (streams 256M elements), but the kernel
  itself is sync-heavy (log N barriers). `scan_bcao` already avoids bank conflicts.
- **Confirm with:** SOL Memory %, achieved occupancy, `smsp__sass_average_branch...`/
  barrier stalls (Warp State). Compare bcao vs plain to see conflict cost.
- **Likely win:** larger elements per thread (vectorized global I/O), `__shfl`-based
  warp scan to cut shared-mem barriers, or decoupled-lookback single-pass scan (CUB).

### bscan-cuda  — `1000`
- **Source:** binary scan; `__shared__ int sdata[64]`, warp-level work.
- **Prior:** tiny working set; likely **latency/occupancy-bound**, not bandwidth.
- **Confirm with:** SOL (expect both low), occupancy, launch count from nsys.
- **Likely win:** `__ballot_sync` + `__popc` for the binary prefix (one instruction
  per warp), removing the shared-mem reduction entirely.

### histogram-cuda  — `--i=100`
- **Source:** `histogram_smem_atomics.h` (per-block shared histogram + merge) vs
  `histogram_gmem_atomics.h` (direct global atomics). run.log prints GB/s + %peak.
- **Prior:** memory-bound + atomic contention. smem version already ~2x gmem. The
  per-block privatization is the known good pattern; headroom is in load width and
  bin-merge efficiency.
- **Confirm with:** SOL Memory %, L2/shared atomic throughput, replay overhead from
  atomic conflicts (depends on input entropy — note the `entropy-reduction` arg).
- **Likely win:** vectorized `uchar4`/`float4` loads (already partly there), more
  sub-histograms per block to spread atomic conflicts, warp-aggregated atomics.

### filter-cuda  — `100000000 256 100` (stream compaction)
- **Source:** `main.cu` — block-local `atomicAdd(&l_n,1)` to count survivors, then one
  `atomicAdd(nres, l_n)` per block; a warp-aggregated variant uses cooperative groups
  `g.size()` + `atomicAdd(ctr, ...)`.
- **Prior:** memory-bound streaming read; the shared-then-global atomic is the
  compaction cost. Warp-aggregation already reduces global atomic pressure.
- **Confirm with:** SOL Memory %, DRAM read throughput (should be near peak if optimal),
  atomic traffic. Compare the two kernels in nsys.
- **Likely win:** vectorized loads, fully warp-aggregated counting (`__ballot`+`__popc`),
  fewer global atomics. Bandwidth ceiling caps the realistic speedup.

---

## Math / solver family (algorithm-dependent — profile before assuming)

### jacobi-cuda  — `./main` (no args)
- **Source:** `jacobi_step` with `__shared__ float f_old_tile[18][18]` (16x16 + halo),
  then a block reduction of the error via `__shfl` + `atomicAdd(error,...)`.
- **Prior:** stencil → memory-bound on the tile loads; the per-step `atomicAdd(error)`
  global reduction and the host-side convergence loop (many short launches) add overhead.
- **Confirm with:** nsys launch count + gpukernsum (is error-reduction a separate
  hot kernel?), SOL Memory %, occupancy with that 18x18 tile.
- **Likely win:** fuse error reduction, fewer global atomics, bigger tiles / halo reuse,
  or reduce launch count if the convergence loop relaunches per iteration.

### gaussian-cuda  — `-q -t -s 4096`
- **Source:** two kernels (`Fan1`/`Fan2` style around lines 173/185) inside a host loop
  `for (t=0; t<size-1; t++)` — i.e. **(size-1) launches**, each doing a shrinking amount
  of work (forward elimination triangle).
- **Prior:** **launch-overhead + load-imbalance bound** (classic Gaussian elim pattern).
  Late iterations have almost no work but still pay launch latency. This is the same
  shape as the Floyd-Warshall 5.43x win you already got.
- **Confirm with:** nsys gpukernsum (huge launch count, small avg kernel time), API time
  vs GPU time, occupancy collapsing in late iterations.
- **Likely win (highest upside):** reduce launch count, fuse the two kernels, or block
  the elimination. Strong candidate for a big speedup.

### thomas-cuda  — `1024 16384 64 100`
- **Source:** `cuThomasBatch` — one thread per tridiagonal system, sequential forward
  then backward sweep (`for i = first+stride; ...` then reverse). Batched solver.
- **Prior:** **memory-bound with a serial dependency chain per thread**; the sweep is
  inherently sequential, so it's about memory layout. With BATCHCOUNT stride, access is
  interleaved across systems — coalescing depends on layout.
- **Confirm with:** SOL Memory %, global load/store efficiency (coalescing), occupancy.
- **Likely win:** ensure coalesced layout (system-interleaved is good), more systems
  in flight for latency hiding, or PCR/CR hybrid if dependency stalls dominate.

### jaccard-cuda  — `1024 512 1000`
- **Source:** CSR sparse kernels (`jaccard_row_*`), warp-cooperative with `__shfl_up_sync`
  prefix, nested CSR loops, `atomicAdd(&weight_i[j], ...)`. Plus fill/scale kernels.
- **Prior:** **sparse, irregular → likely memory-bound + load-imbalanced** across rows;
  atomics on weight accumulation; `__shfl` reduction already present.
- **Confirm with:** nsys gpukernsum (which of the row kernels dominates), SOL Memory %,
  warp execution efficiency (divergence from variable row lengths), atomic traffic.
- **Likely win:** balance work across rows (per-nnz instead of per-row mapping), reduce
  atomics, better coalescing on CSR value loads. Irregularity makes gains data-dependent.

---

## Expected priority (pre-profiling guess — REORDER after real data)

Ranked by *upside × confidence × low-risk*:

1. **gaussian-cuda** — launch-overhead pattern, same shape as your 5.43x FW win. Highest upside.
2. **jacobi-cuda** — launch count + global atomic error reduction; clear levers.
3. **bscan-cuda** — small, ballot/popc rewrite is clean and likely a real win.
4. **scan-cuda** — warp-scan / vectorized I/O; moderate, well-trodden.
5. **jaccard-cuda** — real upside via load balancing, but data-dependent and riskier.
6. **histogram / filter / atomicReduction** — already near bandwidth ceiling; honest
   gains are small. Good for *confirming* you match/beat baseline, not for headline speedups.

This is a hypothesis. The nsys launch-count + ncu SOL numbers decide the real order.
