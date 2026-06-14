#!/bin/bash
#SBATCH -A ACD115083
#SBATCH -p gp1d
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node 1
#SBATCH -J hec-profile
#SBATCH -t 90
#SBATCH -o logs/profile-%j.out
#
# Profile one HeCBench benchmark with nvprof + Nsight Compute (ncu).
# Usage:  sbatch scripts/profile.sbatch <bench>
#   e.g.  sbatch scripts/profile.sbatch convolution1D
#
# Produces, under profiling/results/<jobid>/:
#   <bench>.<baseline|optimized>.gpu-summary.txt   (nvprof per-kernel time share)
#   <bench>.<baseline|optimized>.api-summary.txt   (nvprof CUDA API time share)
#   <bench>.<baseline|optimized>.ncu.txt           (ncu per-kernel metrics)

set -u
BENCH="${1:?usage: sbatch scripts/profile.sbatch <bench>}"
module load cuda

ROOT=/home/r14922146/hecbench-cuda-optimization
SRC=/home/r14922146/src/${BENCH}-cuda
OPT=$ROOT/optimized/${BENCH}-cuda
OUT=$ROOT/profiling/results/${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"

# ---- per-bench profiling arguments -------------------------------------------
# PROF_ARGS : reduced repeat, FULL problem size (keep memory footprint > L2 so the
#             memory behaviour stays representative).
# NCU_OPTS  : full ncu option string (metric set + how many kernels to replay).
#             ncu replays each captured kernel many times, so cap with
#             --launch-count and use lighter --section sets for slow kernels.
case "$BENCH" in
  convolution1D)
      PROF_ARGS="134217728 2"
      NCU_OPTS="--set full --kernel-name regex:conv1d --launch-count 18" ;;
  convolution3D)
      PROF_ARGS="32 96 256 26 26 5 2"   # baseline's heavy conv layer (the real target)
      NCU_OPTS="--set full --kernel-name regex:conv3d --launch-count 6" ;;
  bilateral)
      PROF_ARGS="2960 1440 0.5 0.5 2"
      NCU_OPTS="--set full --kernel-name regex:bilateralFilter --launch-count 3" ;;
  nbody)
      PROF_ARGS="16000 3"
      NCU_OPTS="--set full --kernel-name regex:accelerate_particles --launch-count 2" ;;
  stencil3d)
      PROF_ARGS="512 10"
      NCU_OPTS="--set full --kernel-name regex:stencil3d --launch-count 2" ;;
  convolutionSeparable)
      PROF_ARGS="8192 8192 10"
      NCU_OPTS="--set full --kernel-name regex:conv_ --launch-count 2" ;;
  laplace3d)
      PROF_ARGS="512 512 512 10 0"
      NCU_OPTS="--set full --kernel-name regex:laplace3d --launch-count 2" ;;
  heat)
      PROF_ARGS="4096 100"
      NCU_OPTS="--set full --kernel-name regex:solve --launch-count 2" ;;
  lavaMD)
      PROF_ARGS="-boxes1d 30"
      NCU_OPTS="--set full --kernel-name regex:md --launch-count 1" ;;
  fdtd3d)
      PROF_ARGS="--dimx=192 --dimy=184 --timesteps=10"
      NCU_OPTS="--set full --kernel-name regex:finite_difference --launch-count 2" ;;
  miniWeather)
      PROF_ARGS=""
      NCU_OPTS="--launch-count 12 --section SpeedOfLight --section Occupancy" ;;
  srad)
      PROF_ARGS="1000 0.5 502 458"
      NCU_OPTS="--launch-count 8 --section SpeedOfLight --section Occupancy" ;;
  xsbench)
      PROF_ARGS="-s large -m event -r 1"
      NCU_OPTS="--launch-count 1 --section SpeedOfLight --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis" ;;
  *)  echo "ERROR: no profiling args defined for '$BENCH' yet"; exit 1 ;;
esac

build() {
  echo "== build: $1 =="
  make -C "$1" clean >/dev/null 2>&1
  make -C "$1" ARCH=sm_70 -j 2>&1 | tail -4
}

run_nvprof() {
  local tag=$1 bin=$2
  echo "== nvprof gpu-summary [$tag] =="
  nvprof --print-gpu-summary "$bin" $PROF_ARGS > "$OUT/${BENCH}.${tag}.gpu-summary.txt" 2>&1
  echo "== nvprof api-summary [$tag] =="
  nvprof --print-api-summary "$bin" $PROF_ARGS > "$OUT/${BENCH}.${tag}.api-summary.txt" 2>&1
}

run_ncu() {
  local tag=$1 bin=$2
  echo "== ncu [$tag] =="
  ncu $NCU_OPTS "$bin" $PROF_ARGS > "$OUT/${BENCH}.${tag}.ncu.txt" 2>&1
}

build "$SRC"
[ -e "$SRC/main" ] && { run_nvprof baseline "$SRC/main"; run_ncu baseline "$SRC/main"; }

if [ -d "$OPT" ]; then
  build "$OPT"
  [ -e "$OPT/main" ] && { run_nvprof optimized "$OPT/main"; run_ncu optimized "$OPT/main"; }
fi

echo "DONE -> $OUT"
ls -l "$OUT"
