#!/bin/bash
#SBATCH -A ACD115083
#SBATCH -p gp1d
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node 1
#SBATCH -J hec-smoke
#SBATCH -t 15
#SBATCH -o logs/smoke-%j.out
#
# Quick build + correctness smoke test of the OPTIMIZED build (small args).
# Usage:  sbatch scripts/smoke.sbatch <bench>

set -u
BENCH="${1:?usage: sbatch scripts/smoke.sbatch <bench>}"
module load cuda

ROOT=/home/r14922146/hecbench-cuda-optimization
OPT=$ROOT/optimized/${BENCH}-cuda

case "$BENCH" in
  convolution1D)  SMOKE_ARG_SETS=("1048576 3") ;;
  convolution3D)  SMOKE_ARG_SETS=("32 6 16 14 14 5 5" "32 96 256 26 26 5 5") ;;
  bilateral)      SMOKE_ARG_SETS=("2960 1440 0.5 0.5 5") ;;
  nbody)          SMOKE_ARG_SETS=("16000 3") ;;
  stencil3d)      SMOKE_ARG_SETS=("512 5") ;;
  convolutionSeparable)  SMOKE_ARG_SETS=("8192 8192 5") ;;
  laplace3d)      SMOKE_ARG_SETS=("128 128 128 100 1") ;;
  heat)           SMOKE_ARG_SETS=("4096 100") ;;
  fdtd3d)         SMOKE_ARG_SETS=("--dimx=192 --dimy=184 --timesteps=10") ;;
  miniWeather)    SMOKE_ARG_SETS=("") ;;
  xsbench)        SMOKE_ARG_SETS=("-s small -m event -r 2") ;;
  *)  echo "ERROR: no smoke args for '$BENCH'"; exit 1 ;;
esac

echo "== build optimized =="
make -C "$OPT" clean >/dev/null 2>&1
make -C "$OPT" ARCH=sm_70 -j 2>&1 | tail -6 || { echo "BUILD FAILED"; exit 1; }

for args in "${SMOKE_ARG_SETS[@]}"; do
  echo "== run optimized: ./main $args =="
  "$OPT/main" $args
done
echo "== smoke done =="
