# Profiling Workflow — bottleneck analysis → optimization

This drives the "profile first, then optimize the 9 targets" plan. The 9 benchmarks: `histogram, scan, filter, atomicReduction, bscan` (reduction/scan) and `jacobi, gaussian, thomas, jaccard` (math/solver).

HeCBench source is sparse-checked-out under `math_data_compress/hecbench-src/src/` (ORNL/HeCBench). All paths below are relative to the repo root.

## Quick Start

On the GPU box (V100, sm_70), from the repo root:

```bash
# 0. load CUDA if the toolchain isn't on PATH
module load cuda            # skip if nvcc/nsys/ncu already available

# 1. probe the toolchain — tells us nsys+ncu (best) vs nvprof (fallback)
bash math_data_compress/profiling/scripts/env-probe.sh | tee math_data_compress/profiling/analysis/env.txt

# 2. build + run + profile all 9 benchmarks
ARCH=sm_70 bash math_data_compress/profiling/scripts/profile-all.sh

# 3. tarball the results to send back
tar czf profiling-out.tgz math_data_compress/profiling/results/<timestamp>
```

Then send back `env.txt` and `profiling-out.tgz`. Useful overrides: `ONLY="gaussian-cuda jacobi-cuda"` to profile a subset, `SRC=...` / `OUT=...` to relocate input/output dirs.

## Step 0 — probe the GPU box (you don't have a GPU locally)

On the V100 (load CUDA module first if needed, e.g. `module load cuda`):

```bash
bash math_data_compress/profiling/scripts/env-probe.sh | tee math_data_compress/profiling/analysis/env.txt
```

Send `env.txt` back. It tells us whether to use **nsys+ncu** (best) or **nvprof** (fallback).

## Step 1 — profile all 9

```bash
ARCH=sm_70 bash math_data_compress/profiling/scripts/profile-all.sh
tar czf profiling-out.tgz math_data_compress/profiling/results/<timestamp>
```

Per benchmark it captures: `run.log` (authoritative timing), `nsys.txt` (which kernel owns the time + memcpy share), `ncu.txt` (SpeedOfLight / occupancy / memory throughput). Send the tarball back.

## Step 2 — classify bottlenecks

`math_data_compress/profiling/analysis/template.md` has, per benchmark, the prior hypothesis (from source) and which metric confirms it. Once real numbers come back we fill it in and the optimization direction is locked. The one rule: ncu SOL Compute% vs Memory% → memory-bound / compute-bound / latency-bound.

## Step 3 — prioritize, then optimize one at a time

Reorder the priority list in `template.md` using real launch-count + SOL data, then optimize highest-upside-first. Each optimized program goes under `optimized/<name>/` (same convention as the existing `optimized/` dir), and we compare baseline-vs-optimized on the same V100 with identical args. Pre-profiling guess: **gaussian** and **jacobi** have the most headroom (launch-overhead pattern, like the 5.43x Floyd-Warshall win); the bandwidth-bound ones (histogram/filter/atomicReduction) have little honest headroom.

## Files
- `scripts/env-probe.sh` — toolchain detection
- `scripts/profile-all.sh` — build + run + profile all 9
- `analysis/template.md` — per-benchmark bottleneck hypotheses + metrics to check
- `analysis/env.txt`, `results/<ts>/` — produced on the V100, sent back
