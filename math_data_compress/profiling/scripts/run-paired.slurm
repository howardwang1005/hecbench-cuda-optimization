#!/bin/bash
#SBATCH --job-name=paired-bench
#SBATCH --account=ACD115083        # TWCC project (平行與分散式程式設計); required for billing
#SBATCH --partition=gp1d           # same partition as the official baseline (job 951758)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1               # one V100, like the official run
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00            # headroom; scan is the long pole (~256M elems)
#SBATCH --output=paired-%j.out
#SBATCH --error=paired-%j.err
#
# Submit the paired benchmark on a DEDICATED GPU compute node, matching the
# official HeCBench baseline conditions. Do NOT run benchmarks on the login
# node (un-ln01): its GPUs are shared/throttled, so timings drift (e.g. thomas
# measured 1.17 ms on the login node vs the official 2.35 ms on a gp1d node).
#
# Usage (from repo root, on the login node):
#   sbatch math_data_compress/profiling/scripts/run-paired.slurm
#   squeue -u $USER          # watch it
#   cat paired-<jobid>.out   # results when done
#
# Adjust partition/account below if your project requires them
# (check with: sinfo / your TWCC project docs).
set -u

echo "host=$(hostname)  job=$SLURM_JOB_ID  partition=$SLURM_JOB_PARTITION"
module load cuda            # CUDA 12.x; official baseline used 12.3
export CUDA_VISIBLE_DEVICES=0   # single card, like the official run

nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,driver_version --format=csv,noheader

# 5 repeats + medians, matching the official 5-run methodology.
ARCH=sm_70 REPEAT=5 bash math_data_compress/profiling/scripts/paired-benchmark.sh
