/**
 *  @file cuThomasBatch.cu  (optimized)
 *
 *  Optimized batched tridiagonal solver for HeCBench `thomas-cuda`.
 *
 *  --------------------------------------------------------------------------
 *  Why the baseline is slow (profiled on V100, sm_70):
 *    - Baseline maps ONE thread to ONE system and runs a serial forward +
 *      backward Thomas sweep (M=1024 iterations, a long dependency chain).
 *    - With N=16384 systems that is only 16448 threads (grid 257 x block 64).
 *      ncu: Waves Per SM = 0.18, Achieved Occupancy = 9.95%. The GPU is nearly
 *      empty and the per-element global-memory latency cannot be hidden.
 *    - Memory layout is already interleaved/coalesced, so coalescing is NOT the
 *      problem; the problem is a lack of parallelism / resident warps.
 *
 *  Optimization: Parallel Cyclic Reduction (PCR), one block per system.
 *    - Grid = BATCHCOUNT blocks  -> 16384 blocks fill the device.
 *    - Block = M threads, one element per thread, all M elements of a system
 *      worked in parallel in shared memory.
 *    - PCR solves the system in log2(M) parallel steps instead of M serial
 *      steps, removing the long dependency chain AND raising occupancy.
 *    - The synthetic systems are strictly diagonally dominant
 *      (|D| in [5,10] > |L|+|U| in [-2,2]), so PCR is numerically stable here.
 *
 *  Drop-in: same kernel name / signature semantics, launched with one block
 *  per system (see main.cu). Data layout unchanged: element i of system s is at
 *  array[i*BATCHCOUNT + s] (interleaved), so smem loads/stores stay coalesced.
 *  --------------------------------------------------------------------------
 */
#include "cuThomasBatch.h"

// One block solves one tridiagonal system of M equations via PCR.
//   blockDim.x == M, gridDim.x == BATCHCOUNT.
// Shared memory: 4 doubles per equation, double-buffered = 8*M doubles.
__global__ void cuThomasBatchPCR(const double *__restrict__ L,
                                 const double *__restrict__ D,
                                 const double *__restrict__ U,
                                       double *__restrict__ RHS,
                                 const int M,
                                 const int BATCHCOUNT)
{
  extern __shared__ double smem[];
  // current buffers
  double *sl = smem;            // [M]
  double *sd = sl + M;          // [M]
  double *su = sd + M;          // [M]
  double *sr = su + M;          // [M]
  // ping-pong buffers for the updated coefficients
  double *sl2 = sr + M;         // [M]
  double *sd2 = sl2 + M;        // [M]
  double *su2 = sd2 + M;        // [M]
  double *sr2 = su2 + M;        // [M]

  const int sys = blockIdx.x;
  if (sys >= BATCHCOUNT) return;
  const int i = threadIdx.x;            // equation index within this system
  if (i >= M) return;

  // Interleaved layout: element i of system `sys` lives at i*BATCHCOUNT + sys.
  // Consecutive systems (sys, sys+1) are adjacent -> coalesced across blocks.
  const long gi = (long)i * BATCHCOUNT + sys;
  sl[i] = L[gi];
  sd[i] = D[gi];
  su[i] = U[gi];
  sr[i] = RHS[gi];
  __syncthreads();

  // PCR: double the reach each step until it spans the whole system.
  for (int delta = 1; delta < M; delta <<= 1) {
    double dl = sl[i], dd = sd[i], du = su[i], dr = sr[i];

    double k1 = 0.0, k2 = 0.0;
    if (i - delta >= 0)   k1 = dl / sd[i - delta];
    if (i + delta <  M)   k2 = du / sd[i + delta];

    double nd = dd, nr = dr, nl = 0.0, nu = 0.0;
    if (i - delta >= 0) {
      nd -= su[i - delta] * k1;
      nr -= sr[i - delta] * k1;
      nl  = -sl[i - delta] * k1;
    }
    if (i + delta < M) {
      nd -= sl[i + delta] * k2;
      nr -= sr[i + delta] * k2;
      nu  = -su[i + delta] * k2;
    }

    __syncthreads();           // all reads of the old buffer are done
    sl2[i] = nl; sd2[i] = nd; su2[i] = nu; sr2[i] = nr;
    __syncthreads();           // updated buffer visible to all

    // swap current <- updated
    double *t;
    t = sl; sl = sl2; sl2 = t;
    t = sd; sd = sd2; sd2 = t;
    t = su; su = su2; su2 = t;
    t = sr; sr = sr2; sr2 = t;
  }

  // After PCR the system is diagonal: sd[i]*x[i] = sr[i].
  RHS[gi] = sr[i] / sd[i];
}

// Host launcher kept in THIS translation unit so that taking the address of the
// __global__ cuThomasBatchPCR (for cudaFuncSetAttribute) happens where the kernel
// is defined. Taking a __global__'s address from another .cu under whole-program
// compilation produces an "undefined reference" at link time; launching it via
// <<<>>> from the same TU avoids needing -rdc=true.
void launchThomasPCR(const double *L, const double *D, double *U, double *RHS,
                     int M, int BATCHCOUNT, size_t smem_bytes)
{
  // Take the kernel address through an explicitly-typed function pointer so the
  // compiler resolves the single definition (avoids the "more than one instance
  // of overloaded function" error when passing the bare __global__ name).
  void (*kptr)(const double *, const double *, double *, double *, int, int) =
      &cuThomasBatchPCR;
  cudaFuncSetAttribute((const void *)kptr,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)smem_bytes);
  cuThomasBatchPCR<<<BATCHCOUNT, M, smem_bytes>>>(L, D, U, RHS, M, BATCHCOUNT);
}

// Compatibility wrapper keeping the original name/signature so callers that
// still launch "cuThomasBatch" work. main.cu launches cuThomasBatchPCR directly
// with one block per system and the shared-memory size; this wrapper is only a
// fallback and is not used on the hot path.
__global__ void cuThomasBatch(const double *L, const double *D,
                              double *U, double *RHS,
                              const int M, const int BATCHCOUNT)
{
  // Not used by the optimized main.cu; kept for ABI/source compatibility.
  // Falls back to the original serial-per-system algorithm.
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < BATCHCOUNT) {
    int first = tid;
    int last  = BATCHCOUNT * (M - 1) + tid;
    U[first]   /= D[first];
    RHS[first] /= D[first];
    for (int i = first + BATCHCOUNT; i < last; i += BATCHCOUNT) {
      U[i]   /= D[i] - L[i] * U[i - BATCHCOUNT];
      RHS[i]  = (RHS[i] - L[i] * RHS[i - BATCHCOUNT]) /
                (D[i] - L[i] * U[i - BATCHCOUNT]);
    }
    RHS[last] = (RHS[last] - L[last] * RHS[last - BATCHCOUNT]) /
                (D[last] - L[last] * U[last - BATCHCOUNT]);
    for (int i = last - BATCHCOUNT; i >= first; i -= BATCHCOUNT) {
      RHS[i] -= U[i] * RHS[i + BATCHCOUNT];
    }
  }
}
