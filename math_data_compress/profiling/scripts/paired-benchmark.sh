#!/usr/bin/env bash
# paired-benchmark.sh — Baseline vs optimized comparison for the 5 optimized
# HeCBench CUDA benchmarks, on the V100.
#
# For each benchmark it builds the baseline and the optimized version, runs both
# with IDENTICAL arguments (warmup + N repeats), saves full logs, extracts the
# primary timing metric, and prints a speedup table.
#
# Usage (repo root, on the GPU box; load CUDA first if needed):
#   ARCH=sm_70 bash math_data_compress/profiling/scripts/paired-benchmark.sh
#
# Env vars:
#   ARCH     GPU arch                      (default: sm_70)
#   BASE     baseline src root             (default: math_data_compress/hecbench-src/src)
#   OPT      optimized src root            (default: optimized)
#   OUT      output dir                    (default: math_data_compress/profiling/paired/<ts>)
#   REPEAT   timing runs per program       (default: 3; medians reported)
#   ONLY     subset, e.g. "thomas gaussian"(default: all 5)
set -u

ARCH="${ARCH:-sm_70}"
BASE="${BASE:-math_data_compress/hecbench-src/src}"
OPT="${OPT:-optimized}"
OUT="${OUT:-math_data_compress/profiling/paired/$(date +%Y%m%d-%H%M%S)}"
REPEAT="${REPEAT:-3}"
ONLY="${ONLY:-thomas gaussian jaccard bscan scan}"

# name | dir-suffix | run args   (args match the profiling run exactly)
read -r -d '' BENCHES <<'EOF'
thomas    thomas-cuda    1024 16384 64 100
gaussian  gaussian-cuda  -q -t -s 4096
jaccard   jaccard-cuda   1024 512 1000
bscan     bscan-cuda     1000
scan      scan-cuda      268435456 100
EOF

mkdir -p "$OUT"
echo "ARCH=$ARCH  REPEAT=$REPEAT  OUT=$OUT"
echo "baseline=$BASE  optimized=$OPT"
SUMMARY="$OUT/summary.csv"
echo "benchmark,variant,metric,unit,value" > "$SUMMARY"

build() { # dir
  ( cd "$1" && make clean >/dev/null 2>&1; make ARCH="$ARCH" -j ) >/dev/null 2>"$2"
  [ -x "$1/main" ]
}

# median of stdin numbers
median() { sort -n | awk '{a[NR]=$1} END{ if(NR==0){print "NA"} else if(NR%2){print a[(NR+1)/2]} else {print (a[NR/2]+a[NR/2+1])/2} }'; }

# --- per-benchmark primary metric extractors (echo "value unit role") ---
# role: lower = lower-is-better, higher = higher-is-better
extract_thomas()  { grep "Average kernel execution time" "$1" | grep -oE "[0-9.]+" | head -1; }   # ms, lower
extract_gaussian(){ grep "Total kernel execution time"   "$1" | grep -oE "[0-9.]+" | head -1; }   # us, lower
# jaccard prints two lines (weighted, unweighted); sum them as the primary metric
extract_jaccard() { grep "Average execution time of kernels" "$1" | grep -oE "[0-9.eE+-]+" \
                    | awk '{s+=$1} END{printf "%.8f", s}'; }                                        # s, lower
# bscan: sum of per-block-size avg times (total work), lower better
extract_bscan()   { grep "^Average execution time:" "$1" | grep -oE "[0-9.]+" \
                    | awk '{s+=$1} END{printf "%.3f", s}'; }                                        # us, lower
# scan: sum of all "w/ bank conflicts" timing lines (primary kernel), lower better
extract_scan()    { grep "scan (w/  bank conflicts)" "$1" | grep -oE "[0-9.]+ \(us\)" \
                    | grep -oE "[0-9.]+" | awk '{s+=$1} END{printf "%.3f", s}'; }                   # us, lower

# correctness gate per benchmark (echo PASS/FAIL/UNKNOWN)
check_thomas()  { local e; e=$(grep "Maximum error" "$1" | grep -oE "[0-9.eE+-]+" | head -1); awk -v e="$e" 'BEGIN{print (e=="")?"UNKNOWN":((e+0<1e-6)?"PASS":"FAIL")}'; }
check_gaussian(){ grep -q "^PASS" "$1" && echo PASS || echo FAIL; }
check_jaccard() { echo "UNKNOWN(needs -DDEBUG)"; }   # default build has no verify
check_bscan()   { grep -q "verify = FAIL" "$1" && echo FAIL || (grep -q "verify = PASS" "$1" && echo PASS || echo UNKNOWN); }
check_scan()    { grep -q "FAIL" "$1" && echo FAIL || (grep -q "PASS" "$1" && echo PASS || echo UNKNOWN); }

unit_of()  { case "$1" in thomas) echo ms;; gaussian|bscan|scan) echo us;; jaccard) echo s;; esac; }

# Official HeCBench baseline medians (baseline/results/951758/summary), in the
# SAME unit/aggregation each extractor produces, so optimized can be compared to
# the published target regardless of how the locally-rebuilt baseline behaves.
#   thomas: average_kernel_execution_time (ms)
#   gaussian: total_kernel_execution_time (us)
#   jaccard: weighted+unweighted pipeline sum (s)
#   bscan: sum of 6 block-size execution_time (us)
#   scan: sum of 20 with-conflicts timings (us)
official_of() {
  case "$1" in
    thomas)   echo 2.352232 ;;
    gaussian) echo 490962 ;;
    jaccard)  echo 0.01229218 ;;
    bscan)    echo 2884.6 ;;
    scan)     echo 107187.2 ;;
    *)        echo NA ;;
  esac
}

run_variant() { # name dir placeholder logpath args...
  local name="$1" dir="$2" log="$4"; shift 4; local args="$*"
  # warmup
  ( cd "$dir" && ./main $args ) >/dev/null 2>&1
  for r in $(seq 1 "$REPEAT"); do
    ( cd "$dir" && ./main $args ) > "${log}.run${r}" 2>&1
  done
  cp "${log}.run1" "${log}"   # keep run1 as the representative full log
}

printf "\n%-10s %12s %12s %12s %9s %9s  %-7s\n" \
  "benchmark" "official" "local_base" "optimized" "vs_offic" "vs_local" "opt_ok"
printf -- "----------------------------------------------------------------------------------------\n"

while IFS= read -r row; do
  [ -z "${row// }" ] && continue
  set -- $row
  name="$1"; sfx="$2"; shift 2; args="$*"
  grep -qw "$name" <<<"$ONLY" || continue

  bdir="$BASE/$sfx"; odir="$OPT/$sfx"
  bodir="$OUT/$name"; mkdir -p "$bodir"

  if ! build "$bdir" "$bodir/baseline.build.err"; then
    echo "$name: BASELINE BUILD FAILED (see $bodir/baseline.build.err)"; continue; fi
  if ! build "$odir" "$bodir/optimized.build.err"; then
    echo "$name: OPTIMIZED BUILD FAILED (see $bodir/optimized.build.err)"; continue; fi

  run_variant "$name" "$bdir" x "$bodir/baseline.log" $args
  run_variant "$name" "$odir" x "$bodir/optimized.log" $args

  # median metric over the REPEAT runs
  bval=$(for r in $(seq 1 "$REPEAT"); do "extract_$name" "$bodir/baseline.log.run${r}"; echo; done | median)
  oval=$(for r in $(seq 1 "$REPEAT"); do "extract_$name" "$bodir/optimized.log.run${r}"; echo; done | median)
  unit=$(unit_of "$name")
  bok=$("check_$name" "$bodir/baseline.log"); ook=$("check_$name" "$bodir/optimized.log")

  off=$(official_of "$name")
  # speedup vs official published baseline (the real target) and vs local rebuild
  sp_off=$(awk -v b="$off"  -v o="$oval" 'BEGIN{ if(o>0 && b!="NA" && o!="NA") printf "%.2fx", b/o; else print "NA" }')
  sp_loc=$(awk -v b="$bval" -v o="$oval" 'BEGIN{ if(o>0 && b!="NA" && o!="NA") printf "%.2fx", b/o; else print "NA" }')
  printf "%-10s %12s %12s %12s %9s %9s  %-7s\n" "$name" "$off" "$bval" "$oval" "$sp_off" "$sp_loc" "$ook"
  echo "$name,official_baseline,kernel_time,$unit,$off"  >> "$SUMMARY"
  echo "$name,local_baseline,kernel_time,$unit,$bval"    >> "$SUMMARY"
  echo "$name,optimized,kernel_time,$unit,$oval"         >> "$SUMMARY"
  echo "$name,speedup_vs_official,ratio,x,$sp_off"       >> "$SUMMARY"
  echo "$name,speedup_vs_local,ratio,x,$sp_loc"          >> "$SUMMARY"
done <<< "$BENCHES"

printf -- "----------------------------------------------------------------------------------------\n"
echo "Lower time = better. vs_offic = official_baseline/optimized (THE TARGET);"
echo "vs_local = locally-rebuilt baseline/optimized (same-env control, can drift)."
echo "Official medians from baseline/results/951758. Pin CUDA_VISIBLE_DEVICES=0 for stable local numbers."
echo "Full logs + summary.csv in: $OUT"
echo
echo "NOTE: jaccard's built-in check only runs under -DDEBUG; to gate correctness"
echo "rebuild both with EXTRA_CFLAGS=-DDEBUG and confirm PASS, then benchmark without it."
