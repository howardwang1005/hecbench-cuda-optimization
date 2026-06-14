# thomas-cuda — PCR optimization

## Profiling evidence (V100, run 20260614-125000)

Baseline kernel `cuThomasBatch` (args `1024 16384 64 100`):

| metric | value | meaning |
|---|---:|---|
| Duration | 2.45 ms | hot kernel |
| Achieved Occupancy | **9.95 %** | GPU nearly empty |
| Waves Per SM | **0.18** | less than one wave of work launched |
| Compute (SM) | 4 % | not compute-bound |
| Memory | 55 % (490 GB/s) | latency-exposed, not bandwidth-bound |
| Theoretical Occupancy | 56 % | capped by 50 reg/thread |

Root cause: baseline maps **one thread per system** and runs a serial forward+
backward Thomas sweep (M=1024 iterations, long dependency chain). With N=16384
systems that is only 16448 threads (grid 257 × block 64) — far below the
~163 840 resident threads a V100 wants. There aren't enough warps to hide the
per-element global-memory latency. The data layout is already interleaved and
coalesced, so coalescing is **not** the problem; parallelism is.

## Optimization: Parallel Cyclic Reduction (PCR), one block per system

- Grid = N blocks (one per system) → fills the device.
- Block = M threads (one equation per thread); all M equations worked in parallel.
- PCR solves each system in **log2(M) parallel steps** (10 for M=1024) instead of
  **M serial steps** (1024), removing the dependency chain and raising occupancy.
- Coefficients held in shared memory, double-buffered (`8·M` doubles/block); for
  M=1024 that is 64 KB, opted-in via `cudaFuncAttributeMaxDynamicSharedMemorySize`.
- Global I/O unchanged (interleaved layout) so loads/stores stay coalesced.

### Numerical safety
The synthetic systems are strictly diagonally dominant (|D|∈[5,10] > |L|+|U|,
L,U∈[-2,2]), so PCR is stable. Verified on CPU that this exact PCR matches serial
Thomas to ≤ 2.2e-16 for M ∈ {1,2,3,7,8,15,16,1000,1024} (`/tmp/pcr_test.cpp`).

## Expected effect
Occupancy 9.95 % → ~50 % (smem-limited to 1 block/SM at 64 KB), dependency chain
1024 → 10 steps. Memory was at 490/≈900 GB/s, so there is real bandwidth headroom
to capture. Confirm the actual speedup with a paired baseline-vs-optimized run on
the same V100 (`1024 16384 64 100`); keep `Maximum error` ~1e-12 or smaller.

## Files changed vs baseline
- `cuThomasBatch.cu` — added `cuThomasBatchPCR` (the new hot kernel); original
  serial kernel kept as a compatibility fallback.
- `cuThomasBatch.h` — declare `cuThomasBatchPCR`.
- `main.cu` — launch PCR (grid=N, block=M, dynamic smem) with an opt-in for the
  large smem carveout and a guard for `M > maxThreadsPerBlock`.

## Possible follow-ups if occupancy is still the limit
- Cut shared memory (drop double-buffering of l/u, or store fewer arrays) to fit
  2 blocks/SM → 100 % occupancy.
- Hybrid PCR+Thomas: a few PCR steps to split into independent sub-systems, then a
  short serial Thomas per sub-system (less total work than full PCR's M·log M).
