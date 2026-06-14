#!/bin/bash
#SBATCH -A ACD115083
#SBATCH -p gp1d
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node 1
#SBATCH -J hec-compare
#SBATCH -t 90
#SBATCH -o logs/compare-%j.out
#
# Run the OPTIMIZED build 5 times for one benchmark; each program prints its own
# per-kernel time (an average-per-launch metric, independent of repeat count).
# The BASELINE is NOT re-run: it is already measured in
#   baseline/results/951758/<bench>-cuda/run-{1..5}.log
# (same V100, same per-kernel metric) -- reuse those as the baseline side.
# Usage:  sbatch scripts/compare.sbatch <bench>
#
# Logs land in profiling/results/<jobid>/<bench>.optimized.run<N>.log
# Speedup = median(baseline kernel time) / median(optimized kernel time).

set -u
BENCH="${1:?usage: sbatch scripts/compare.sbatch <bench>}"
module load cuda

ROOT=/home/r14922146/hecbench-cuda-optimization
SRC=/home/r14922146/src/${BENCH}-cuda
OPT=$ROOT/optimized/${BENCH}-cuda
OUT=$ROOT/profiling/results/${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"

# ---- per-bench comparison arguments ------------------------------------------
# Args MUST match what the committed baseline (951758) actually ran -- which is
# NOT always the local Makefile `run:` target (e.g. convolution3D ran two conv
# layers, not the Makefile's tiny config). Derive args from baseline run logs.
# A benchmark may need MULTIPLE arg sets (CMP_ARG_SETS) run sequentially into one
# log to mirror the baseline's structure. Per-kernel time is repeat-independent,
# so a smaller repeat is fine as long as both sides use the same problem sizes.
case "$BENCH" in
  convolution1D)  CMP_ARG_SETS=("134217728 50") ;;
  convolution3D)  CMP_ARG_SETS=("32 6 16 14 14 5 100" "32 96 256 26 26 5 100") ;;
  bilateral)      CMP_ARG_SETS=("2960 1440 0.5 0.5 200") ;;
  nbody)          CMP_ARG_SETS=("16000 10") ;;
  stencil3d)      CMP_ARG_SETS=("512 100") ;;
  convolutionSeparable)  CMP_ARG_SETS=("8192 8192 200") ;;
  laplace3d)      CMP_ARG_SETS=("128 128 128 100 1" "512 512 512 100 0") ;;
  heat)           CMP_ARG_SETS=("4096 1000") ;;
  fdtd3d)         CMP_ARG_SETS=("--dimx=192 --dimy=184 --timesteps=90") ;;
  miniWeather)    CMP_ARG_SETS=("") ;;
  xsbench)        CMP_ARG_SETS=("-s large -m event -r 10") ;;
  *)  echo "ERROR: no compare args defined for '$BENCH' yet"; exit 1 ;;
esac

echo "== build optimized =="
make -C "$OPT" clean >/dev/null 2>&1
make -C "$OPT" ARCH=sm_70 -j 2>&1 | tail -4 || { echo "BUILD FAILED"; exit 1; }

echo "== run optimized x5 (arg sets: ${#CMP_ARG_SETS[@]}) =="
for r in 1 2 3 4 5; do
  : > "$OUT/${BENCH}.optimized.run${r}.log"
  for args in "${CMP_ARG_SETS[@]}"; do
    "$OPT/main" $args >> "$OUT/${BENCH}.optimized.run${r}.log" 2>&1
  done
  echo "  optimized run $r done"
done

echo "DONE -> $OUT"
echo "baseline reference: baseline/results/951758/${BENCH}-cuda/run-{1..5}.log"
ls -l "$OUT"/${BENCH}.optimized.run*.log
