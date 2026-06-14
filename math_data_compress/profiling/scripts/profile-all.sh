#!/usr/bin/env bash
# profile-all.sh — Build + run + profile the 9 target HeCBench CUDA benchmarks.
#
# Usage (on the V100 box, from repo root, with HeCBench src checked out):
#
#   ARCH=sm_70 bash math_data_compress/profiling/scripts/profile-all.sh
#
# Env vars:
#   SRC   path to HeCBench src dir          (default: math_data_compress/hecbench-src/src)
#   ARCH  GPU arch                          (default: sm_70  for V100)
#   OUT   output dir                        (default: math_data_compress/profiling/results/<timestamp>)
#   ONLY  space-separated subset of names   (default: all 9)
#
# Output per benchmark (under $OUT/<name>/):
#   run.log         program's own stdout (the authoritative timing numbers)
#   nsys.txt        nsys stats: gpukernsum (kernel time %) + gpumemtimesum (memcpy)
#   ncu.txt         ncu SpeedOfLight + Occupancy + memory throughput per kernel
#   nvprof.txt      fallback if nsys/ncu missing
#
# Send the whole $OUT dir back. analysis.md is generated from it.
set -u

SRC="${SRC:-math_data_compress/hecbench-src/src}"
ARCH="${ARCH:-sm_70}"
OUT="${OUT:-math_data_compress/profiling/results/$(date +%Y%m%d-%H%M%S)}"

# name | run-args  (args chosen to match HeCBench Makefile `run` targets)
read -r -d '' BENCHES <<'EOF'
histogram-cuda        --i=100
scan-cuda             268435456 100
filter-cuda           100000000 256 100
atomicReduction-cuda
bscan-cuda            1000
jacobi-cuda
gaussian-cuda         -q -t -s 4096
thomas-cuda           1024 16384 64 100
jaccard-cuda          1024 512 1000
EOF

ONLY="${ONLY:-}"
HAVE_NSYS=0; command -v nsys  >/dev/null 2>&1 && HAVE_NSYS=1
HAVE_NCU=0;  command -v ncu   >/dev/null 2>&1 && HAVE_NCU=1
HAVE_NVPROF=0; command -v nvprof >/dev/null 2>&1 && HAVE_NVPROF=1

echo "SRC=$SRC ARCH=$ARCH OUT=$OUT"
echo "profilers: nsys=$HAVE_NSYS ncu=$HAVE_NCU nvprof=$HAVE_NVPROF"
mkdir -p "$OUT"

# ncu: limit to the first N kernel invocations so it doesn't run forever on
# benchmarks that launch thousands of kernels. Grab the heaviest sections only.
NCU_SETS="--set basic --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis"
NCU_LIMIT="--launch-count 3 --launch-skip 2"   # skip warmup, sample 3 launches

profile_one() {
  local name="$1"; shift
  local args="$*"
  local bdir="$SRC/$name"
  local odir="$OUT/$name"
  mkdir -p "$odir"
  echo "=================================================="
  echo ">>> $name   args: [$args]"

  if [ ! -d "$bdir" ]; then echo "  SKIP: $bdir missing" | tee "$odir/SKIP"; return; fi

  echo "  building..."
  ( cd "$bdir" && make clean >/dev/null 2>&1; make ARCH="$ARCH" -j 2>&1 ) > "$odir/build.log" 2>&1
  if [ ! -x "$bdir/main" ]; then
    echo "  BUILD FAILED — see $odir/build.log" | tee "$odir/BUILD_FAILED"
    tail -15 "$odir/build.log"
    return
  fi

  echo "  warmup + timing run..."
  ( cd "$bdir" && ./main $args ) > "$odir/run.log" 2>&1
  echo "  --- run.log tail ---"; tail -6 "$odir/run.log" | sed 's/^/    /'

  if [ "$HAVE_NSYS" = 1 ]; then
    echo "  nsys (timeline)..."
    ( cd "$bdir" && nsys profile --force-overwrite true -o "$PWD/.nsys_$name" \
        --stats=false ./main $args ) >/dev/null 2>>"$odir/nsys.err"
    ( cd "$bdir" && nsys stats --report gpukernsum --report gpumemtimesum \
        "$PWD/.nsys_$name.nsys-rep" ) > "$odir/nsys.txt" 2>>"$odir/nsys.err"
    ( cd "$bdir" && rm -f ".nsys_$name.nsys-rep" ".nsys_$name.sqlite" )
    echo "  --- top kernels ---"; grep -A8 "gpukernsum" "$odir/nsys.txt" 2>/dev/null | head -12 | sed 's/^/    /'
  fi

  if [ "$HAVE_NCU" = 1 ]; then
    echo "  ncu (roofline / SOL)..."
    ( cd "$bdir" && ncu $NCU_LIMIT $NCU_SETS ./main $args ) > "$odir/ncu.txt" 2>>"$odir/ncu.err"
  fi

  if [ "$HAVE_NSYS" = 0 ] && [ "$HAVE_NVPROF" = 1 ]; then
    echo "  nvprof (fallback)..."
    ( cd "$bdir" && nvprof --print-gpu-summary ./main $args ) > "$odir/nvprof.txt" 2>&1
    ( cd "$bdir" && nvprof --metrics gld_efficiency,gst_efficiency,achieved_occupancy,dram_utilization \
        ./main $args ) >> "$odir/nvprof.txt" 2>&1
  fi
  echo "  done: $odir"
}

while IFS= read -r row; do
  [ -z "${row// }" ] && continue
  name="${row%% *}"
  args="${row#"$name"}"; args="${args#"${args%%[![:space:]]*}"}"  # trim leading ws
  if [ -n "$ONLY" ] && ! grep -qw "$name" <<<"$ONLY"; then continue; fi
  profile_one "$name" $args
done <<< "$BENCHES"

echo "=================================================="
echo "ALL DONE. Output in: $OUT"
echo "Tarball it back:  tar czf profiling-out.tgz $OUT"
