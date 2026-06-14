# thomas-cuda — Change Record (for report)

**Benchmark:** HeCBench `thomas-cuda` (batched tridiagonal solver, BSC `cuThomasBatch`).
**Hardware:** NVIDIA Tesla V100-SXM2-32GB (sm_70), CUDA 12.8.
**Run arguments:** `M=1024  N=16384  blockSize=64  repeat=100` (system size × #systems).
**Baseline source:** `math_data_compress/hecbench-src/src/thomas-cuda/` (ORNL/HeCBench).
**Optimized source:** `optimized/thomas-cuda/`.
**Artifacts in this folder:** `changes.patch` (full unified diff), `pcr_correctness_test.cpp` (CPU verification harness).

---

## 1. Bottleneck identified (profiling evidence)

Profiled with Nsight Compute (`ncu`), run `20260614-125000`. Hot kernel `cuThomasBatch`:

| Metric | Value | Reading |
|---|---:|---|
| Duration | 2.45 ms | the entire GPU cost |
| Achieved Occupancy | **9.95 %** | GPU almost idle |
| Waves Per SM | **0.18** | fewer than one wave of work is launched |
| Compute (SM) Throughput | 4 % | not compute-bound |
| Memory Throughput | 55 % (490 GB/s) | latency-exposed, not bandwidth-saturated |
| Theoretical Occupancy | 56 % | upper bound (50 registers/thread) |
| Registers/Thread | 50 | — |

**Root cause.** The baseline maps **one thread to one system** and runs a serial
forward + backward Thomas sweep over `M=1024` elements — a long per-thread
dependency chain. With `N=16384` systems the launch is `<<<257, 64>>>` ≈ 16 448
threads, far below the ~163 840 threads a V100 (80 SMs × 2048) wants resident.
There are too few warps to hide the per-element global-memory latency. The input
is already stored interleaved/transposed (`array[i*N + s]`), so memory accesses
are **already coalesced** — coalescing is not the issue; **parallelism is**.

---

## 2. Optimization applied

**Algorithm change: serial Thomas → Parallel Cyclic Reduction (PCR), one block per system.**

| Aspect | Baseline | Optimized |
|---|---|---|
| Thread → work mapping | 1 thread per system | 1 **block** per system, 1 thread per equation |
| Launch config | `<<<(N/64)+1, 64>>>` | `<<<N, M, 8·M·8 B>>>` |
| Algorithm | serial sweep, **M=1024 steps**, dependency chain | PCR, **log₂(M)=10 parallel steps** |
| Working set | global memory, latency-exposed | shared memory, double-buffered (`8·M` doubles) |
| Parallelism | 16 448 threads | N×M = 16.7M threads (16384 blocks) |

Why it addresses the bottleneck:
- **Grid = N blocks** fills the device → raises occupancy from the launch side.
- **Block = M threads** processes all equations of a system concurrently → removes
  the M-long serial dependency chain (10 parallel steps instead of 1024).
- Coefficients are staged in **shared memory**, double-buffered across PCR steps
  (`sl/sd/su/sr` ping-pong with `sl2/sd2/su2/sr2`), so the inner loop avoids
  repeated global-memory round trips.
- Global load/store keep the original interleaved indexing, so they stay coalesced.

Engineering details:
- Shared memory for M=1024 is `8·1024·8 B = 64 KB/block`, which exceeds the 48 KB
  default cap, so the kernel opts in via
  `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, 64KB)`.
- A runtime guard rejects `M > maxThreadsPerBlock` or smem over the device opt-in
  limit, with a clear error message (keeps the program safe for other inputs).
- `blockSize` (argv[3]) is now unused (PCR forces `blockDim == M`); kept for CLI
  compatibility and marked `(void)`.
- The original serial kernel is **retained** in `cuThomasBatch.cu` as a
  compatibility fallback; the hot path calls the new `cuThomasBatchPCR`.

---

## 3. Files changed

Full diff: **`changes.patch`** (apply with `git apply` from repo root, or read directly).

| File | Baseline LOC | Optimized LOC | Change |
|---|---:|---:|---|
| `cuThomasBatch.cu` | 99 | 131 | new `cuThomasBatchPCR` kernel; original kept as fallback |
| `main.cu` | 198 | 229 | PCR launch (grid=N, block=M, dyn smem), smem opt-in, M-guard |
| `cuThomasBatch.h` | 18 | 25 | declare `cuThomasBatchPCR` |

Unchanged: `ThomasMatrix.hpp`, `utils.hpp`, `Makefile`, `CMakeLists.txt`
(build still compiles `main.cu` + `cuThomasBatch.cu`).

### Key code change (kernel)

```cuda
// BEFORE (baseline): 1 thread = 1 system, serial sweep, M=1024 dependent steps
int tid = threadIdx.x + blockDim.x*blockIdx.x;
if (tid < BATCHCOUNT) {
  U[first] /= D[first]; RHS[first] /= D[first];
  for (int i = first+BATCHCOUNT; i < last; i += BATCHCOUNT) { ... }   // forward
  for (int i = last-BATCHCOUNT; i >= first; i -= BATCHCOUNT) { ... }  // backward
}

// AFTER (optimized): 1 block = 1 system, PCR, log2(M)=10 parallel steps
extern __shared__ double smem[];               // 8*M doubles, double-buffered
const int sys = blockIdx.x, i = threadIdx.x;
sl[i]=L[gi]; sd[i]=D[gi]; su[i]=U[gi]; sr[i]=RHS[gi]; __syncthreads();
for (int delta = 1; delta < M; delta <<= 1) {  // reach doubles each step
  // combine row i with rows i±delta, eliminate those unknowns (boundary = 0)
  ... __syncthreads(); write buffer2; __syncthreads(); swap buffers;
}
RHS[gi] = sr[i] / sd[i];                        // system is now diagonal
```

### Launch change (host)

```cuda
// BEFORE
cuThomasBatch<<<(N/BlockSize)+1, BlockSize>>>(l,d,u,rhs, M, N);

// AFTER
const size_t pcr_smem = (size_t)8 * M * sizeof(double);
cudaFuncSetAttribute(cuThomasBatchPCR,
    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pcr_smem);
cuThomasBatchPCR<<<N, M, pcr_smem>>>(l,d,u,rhs, M, N);
```

---

## 4. Correctness verification

The synthetic systems are strictly **diagonally dominant** (`|D|∈[5,10]` >
`|L|+|U|`, with `L,U∈[-2,2]`), which guarantees PCR is numerically stable here.

A standalone CPU harness (`pcr_correctness_test.cpp`) reproduces the kernel's exact
PCR arithmetic and compares against a reference serial Thomas solve:

| M | max \|PCR − Thomas\| |
|---:|---:|
| 1 | 0 |
| 2 | 3.5e-18 |
| 3 | 2.8e-17 |
| 7 | 8.3e-17 |
| 8 | 2.1e-17 |
| 15 | 5.6e-17 |
| 16 | 1.1e-16 |
| 1000 | 1.7e-16 |
| **1024** | **2.2e-16** |

Match to machine precision, including non-power-of-two M. On device, the program's
own check should report `Maximum error` ≈ 1e-12 or smaller and `PASS`.

Reproduce: `g++ -O2 -std=c++17 pcr_correctness_test.cpp -o t && ./t`

---

## 5. Expected effect (confirm with a paired run on V100)

| Quantity | Baseline | Expected after PCR |
|---|---:|---|
| Achieved occupancy | 9.95 % | ~50 % (smem-limited to 1 block/SM at 64 KB) |
| Critical path / system | 1024 serial steps | 10 parallel steps |
| Memory throughput | 490 GB/s (of ~900 peak) | headroom to capture |

The headline speedup must be measured by running baseline and optimized back-to-back
on the same V100 with identical args (`1024 16384 64 100`) and comparing
"Average kernel execution time".

### Follow-ups if occupancy is still the limiter
- Reduce shared memory (drop double-buffering of `l`/`u`, or store fewer arrays)
  to fit **2 blocks/SM → 100 % occupancy**.
- Hybrid **PCR + Thomas**: a few PCR steps to split each system into independent
  sub-systems, then a short serial Thomas per sub-system — less total work than
  full PCR's `M·log M`.
