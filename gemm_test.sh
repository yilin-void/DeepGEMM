#!/bin/bash
set -euo pipefail

MODE="${1:-profile}"
if [[ $# -gt 0 ]]; then
  shift
fi

usage() {
  echo "Usage: $0 [profile|trace|check|ptxas|ptxas-flags] [extra gemm_test.py args...]" >&2
  echo "Example: $0 profile --expert-srank-padding" >&2
  echo "         $0 profile --no-expert-srank-padding" >&2
}

TS=$(date +"%Y%m%d%H%M")
export DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-/tmp/dg_single_gemm_${MODE}_${TS}}"

echo "DG_JIT_CACHE_DIR: ${DG_JIT_CACHE_DIR}"

case "${MODE}" in
  profile)
    # Default to CUDA-events timing (no CUPTI). To get a Chrome trace instead,
    # pass `--timing-mode trace [--trace-path /path]` after the mode arg.
    python gemm_test.py --profile "$@" 2>&1 | tee gemm_test.log
    ;;
  trace)
    TRACE_PATH="${TRACE_PATH:-/tmp/dg_single_gemm_trace_${TS}.json}"
    python gemm_test.py --profile --timing-mode trace --trace-path "${TRACE_PATH}" "$@" 2>&1 | tee gemm_test.log
    ;;
  check)
    python gemm_test.py --check "$@" 2>&1 | tee gemm_test.log
    ;;
  ptxas)
    export DG_JIT_PTXAS_VERBOSE=1
    export DG_PRINT_CONFIGS=1
    python gemm_test.py --modes noflags "$@" 2>&1 | tee gemm_test.log
    ;;
  ptxas-flags)
    export DG_JIT_PTXAS_VERBOSE=1
    export DG_PRINT_CONFIGS=1
    python gemm_test.py --modes flags "$@" 2>&1 | tee gemm_test.log
    ;;
  *)
    usage
    exit 2
    ;;
esac
