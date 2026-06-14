#!/usr/bin/env bash
# env-probe.sh — Detect the CUDA profiling toolchain available on the V100 box.
# Run this FIRST on the GPU machine, then send the output back.
#
#   bash math_data_compress/profiling/scripts/env-probe.sh | tee math_data_compress/profiling/analysis/env.txt
#
set -u

line() { printf '%s\n' "------------------------------------------------------------"; }

echo "# CUDA profiling environment probe"
echo "# host: $(hostname)    date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
line

echo "## GPU (nvidia-smi)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi: NOT FOUND"
fi
line

echo "## Compiler"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | grep -E "release|Cuda"
  echo "nvcc path: $(command -v nvcc)"
else
  echo "nvcc: NOT FOUND  (need: module load cuda  ? )"
fi
line

echo "## Profilers"
for t in nsys ncu nvprof; do
  if command -v "$t" >/dev/null 2>&1; then
    ver="$("$t" --version 2>&1 | head -1)"
    printf "%-7s AVAILABLE  (%s)  %s\n" "$t" "$(command -v "$t")" "$ver"
  else
    printf "%-7s MISSING\n" "$t"
  fi
done
line

echo "## ncu metric-set check (needs a GPU; may require sudo/perfmon perms)"
if command -v ncu >/dev/null 2>&1; then
  # Just confirm ncu can talk to the driver; harmless null kernel run skipped.
  ncu --list-sets 2>&1 | head -8 || echo "ncu present but cannot query sets (permission?)"
else
  echo "ncu missing — will fall back to nsys + nvprof"
fi
line

echo "## Recommendation"
if command -v ncu >/dev/null 2>&1 && command -v nsys >/dev/null 2>&1; then
  echo "USE: nsys (timeline / where-is-time) + ncu (per-kernel roofline). Best case."
elif command -v nvprof >/dev/null 2>&1; then
  echo "USE: nvprof (deprecated on Volta but works). Coarser metrics."
else
  echo "NO PROFILER FOUND. Check 'module avail cuda' / 'module load nsight'."
fi
