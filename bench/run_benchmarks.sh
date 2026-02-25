#!/bin/bash
#
# run_benchmarks.sh - Run trx-cpp benchmarks separately to minimize memory usage
#
# Usage:
#   ./bench/run_benchmarks.sh [options]
#
# Options:
#   --realdata          Run real data benchmarks (bench_trx_realdata, default)
#   --reference PATH    Path to reference TRX file (for realdata)
#   --out-dir DIR       Output directory for JSON results (default: bench)
#   --profile MODE      Benchmark profile: core (default) or full
#   --allow-synth-mp    Allow synthetic multiprocessing (experimental)
#   --verbose           Enable verbose progress logging
#   --help              Show this help message
#
# Environment variables (optional):
#   TRX_BENCH_BUFFER_MULTIPLIER    Buffer size multiplier for slow storage (default: 1)
#   TRX_BENCH_MAX_STREAMLINES      Maximum streamline count to test (profile default)
#   TRX_BENCH_PROCESSES            Number of processes (synthetic only, default: 1)
#

set -e  # Exit on error

# Default values
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$BENCH_DIR")"
OUT_DIR="$BENCH_DIR"
RUN_SYNTHETIC=false
RUN_REALDATA=true
REFERENCE_TRX="$PROJECT_ROOT/test-data/10milHCP_dps-sift2.trx"
VERBOSE_FLAG=""
BUILD_DIR="$PROJECT_ROOT/build-release"
PROFILE="core"
ALLOW_SYNTH_MP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --synthetic)
      RUN_SYNTHETIC=true
      shift
      ;;
    --realdata)
      RUN_REALDATA=true
      shift
      ;;
    --both)
      RUN_SYNTHETIC=true
      RUN_REALDATA=true
      shift
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --allow-synth-mp)
      ALLOW_SYNTH_MP=true
      shift
      ;;
    --reference)
      REFERENCE_TRX="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE_FLAG="--verbose"
      shift
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --help)
      head -n 20 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

if [[ "$PROFILE" != "core" && "$PROFILE" != "full" ]]; then
  echo "Error: --profile must be 'core' or 'full' (got '$PROFILE')"
  exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"

echo "========================================"
echo "TRX-CPP Benchmark Runner"
echo "========================================"
echo "Output directory: $OUT_DIR"
echo "Build directory: $BUILD_DIR"
echo "Run synthetic: $RUN_SYNTHETIC"
echo "Run realdata: $RUN_REALDATA"
echo "Profile: $PROFILE"
echo "Synthetic multiprocessing: $([[ "$ALLOW_SYNTH_MP" == "true" ]] && echo "experimental" || echo "disabled")"
if [[ "$RUN_REALDATA" == "true" ]]; then
  echo "Reference TRX: $REFERENCE_TRX"
fi
echo "Verbose: ${VERBOSE_FLAG:-disabled}"
echo ""

# Function to run a single benchmark
run_benchmark() {
  local executable=$1
  local filter=$2
  local output_file=$3
  local extra_env=$4
  local extra_flags=$5
  
  echo "----------------------------------------"
  echo "Running: $(basename "$executable") --benchmark_filter=$filter"
  echo "Output: $(basename "$output_file")"
  echo "----------------------------------------"
  
  if [[ -n "$extra_env" ]]; then
    eval "$extra_env" "$executable" $VERBOSE_FLAG $extra_flags \
      --benchmark_filter="$filter" \
      --benchmark_out="$output_file" \
      --benchmark_out_format=json
  else
    "$executable" $VERBOSE_FLAG $extra_flags \
      --benchmark_filter="$filter" \
      --benchmark_out="$output_file" \
      --benchmark_out_format=json
  fi
  
  echo "✓ Completed: $(basename "$output_file")"
  echo ""
}

# Build profile environment defaults (users can override by exporting env vars)
if [[ "$PROFILE" == "core" ]]; then
  CORE_ENV="TRX_BENCH_PROFILE=${TRX_BENCH_PROFILE:-core} TRX_BENCH_MAX_STREAMLINES=${TRX_BENCH_MAX_STREAMLINES:-10000000} TRX_BENCH_QUERY_CACHE_MAX=${TRX_BENCH_QUERY_CACHE_MAX:-5} TRX_BENCH_CORE_INCLUDE_BUNDLES=${TRX_BENCH_CORE_INCLUDE_BUNDLES:-0} TRX_BENCH_INCLUDE_CONNECTOME=${TRX_BENCH_INCLUDE_CONNECTOME:-0} TRX_BENCH_CORE_ZIP_MAX_STREAMLINES=${TRX_BENCH_CORE_ZIP_MAX_STREAMLINES:-1000000}"
else
  CORE_ENV="TRX_BENCH_PROFILE=${TRX_BENCH_PROFILE:-full} TRX_BENCH_MAX_STREAMLINES=${TRX_BENCH_MAX_STREAMLINES:-10000000} TRX_BENCH_SKIP_DPV_AT=${TRX_BENCH_SKIP_DPV_AT:-10000000} TRX_BENCH_QUERY_CACHE_MAX=${TRX_BENCH_QUERY_CACHE_MAX:-10} TRX_BENCH_INCLUDE_CONNECTOME=${TRX_BENCH_INCLUDE_CONNECTOME:-1}"
fi

SYNTH_ENV="$CORE_ENV"
if [[ "$ALLOW_SYNTH_MP" != "true" ]]; then
  SYNTH_ENV="$SYNTH_ENV TRX_BENCH_PROCESSES=${TRX_BENCH_PROCESSES:-1}"
fi

# Synthetic benchmarks
if [[ "$RUN_SYNTHETIC" == "true" ]]; then
  SYNTHETIC_BIN="$BUILD_DIR/bench/bench_trx_stream"
  
  if [[ ! -f "$SYNTHETIC_BIN" ]]; then
    echo "Error: Synthetic benchmark not found: $SYNTHETIC_BIN"
    echo "Build with: cmake --build $BUILD_DIR --target bench_trx_stream"
    exit 1
  fi
  
  echo "========================================"
  echo "SYNTHETIC DATA BENCHMARKS"
  echo "========================================"
  echo ""
  
  run_benchmark "$SYNTHETIC_BIN" "BM_TrxFileSize_Float16" \
    "$OUT_DIR/results_synthetic_filesize.json" \
    "$SYNTH_ENV"
  
  run_benchmark "$SYNTHETIC_BIN" "BM_TrxStream_TranslateWrite" \
    "$OUT_DIR/results_synthetic_translate.json" \
    "$SYNTH_ENV"
  
  run_benchmark "$SYNTHETIC_BIN" "BM_TrxQueryAabb_Slabs" \
    "$OUT_DIR/results_synthetic_query.json" \
    "$SYNTH_ENV TRX_QUERY_TIMINGS_PATH=$OUT_DIR/query_timings_synthetic.jsonl"
  
  echo "✓ All synthetic benchmarks completed"
  echo ""
fi

# Real data benchmarks
if [[ "$RUN_REALDATA" == "true" ]]; then
  REALDATA_BIN="$BUILD_DIR/bench/bench_trx_realdata"
  
  if [[ ! -f "$REALDATA_BIN" ]]; then
    echo "Error: Real-data benchmark not found: $REALDATA_BIN"
    echo "Build with: cmake --build $BUILD_DIR --target bench_trx_realdata"
    exit 1
  fi
  
  if [[ ! -f "$REFERENCE_TRX" ]]; then
    echo "Error: Reference TRX file not found: $REFERENCE_TRX"
    echo "Use --reference to specify the path"
    exit 1
  fi
  
  echo "========================================"
  echo "REAL DATA BENCHMARKS"
  echo "========================================"
  echo ""
  
  # Reference flag for all realdata benchmarks
  REALDATA_FLAGS="--reference-trx $REFERENCE_TRX"
  REALDATA_ENV="$CORE_ENV"
  
  run_benchmark "$REALDATA_BIN" "BM_TrxFileSize_Float16" \
    "$OUT_DIR/results_realdata_filesize.json" \
    "$REALDATA_ENV" \
    "$REALDATA_FLAGS"
  
  run_benchmark "$REALDATA_BIN" "BM_TrxStream_TranslateWrite" \
    "$OUT_DIR/results_realdata_translate.json" \
    "$REALDATA_ENV" \
    "$REALDATA_FLAGS"
  
  run_benchmark "$REALDATA_BIN" "BM_TrxQueryAabb_Slabs" \
    "$OUT_DIR/results_realdata_query.json" \
    "$REALDATA_ENV TRX_QUERY_TIMINGS_PATH=$OUT_DIR/query_timings.jsonl" \
    "$REALDATA_FLAGS"
  
  echo "✓ All real-data benchmarks completed"
  echo ""
fi

echo "========================================"
echo "BENCHMARK SUMMARY"
echo "========================================"
echo ""
echo "Results saved to: $OUT_DIR"
ls -lh "$OUT_DIR"/results_*.json 2>/dev/null || echo "No result files found"
echo ""
echo "To generate plots:"
echo "  Rscript bench/plot_bench.R --bench-dir $OUT_DIR --out-dir docs/_static/benchmarks"
echo ""
