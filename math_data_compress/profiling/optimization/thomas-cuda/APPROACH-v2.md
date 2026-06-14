# thomas-cuda — Optimization v2 (register-carry CSE)

## v1 (FAILED): Parallel Cyclic Reduction (1 block per system)
PCR spread one system across 1024 threads with ~20 block barriers + 48KB smem.
The baseline's 1-thread-per-system version has zero barriers and the GPU is busy
(each thread does M=1024 serial steps), so PCR's sync overhead made it ~0.55x
(2x slower) on the same compiler. Reverted to baseline.

## v2 (current): minimal, same-algorithm micro-optimization
Profiling-honest reassessment: the baseline is already coalesced (interleaved
layout) and work-saturated; there is no occupancy lever (work = 16384 systems is
fixed). The only thing baseline leaves on the table is redundant work inside the
serial sweep:
  - it recomputes the denominator D[i]-L[i]*U[i-stride] TWICE per iteration;
  - it re-loads U[i-stride] and RHS[i-stride] from global memory each iteration.

v2 keeps the exact algorithm but carries U/RHS of the previous element in
registers and computes the denominator once. Fewer FP ops and fewer global loads
in the hot loop. CPU-verified bit-exact vs the baseline for M in {2..1024}.

Expected: small win at best (nvcc may already CSE some of it); the point is it
CANNOT regress like PCR did. Confirm vs baseline on the same compiler (12.3).
