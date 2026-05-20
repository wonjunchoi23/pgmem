#!/bin/bash
# Ministral-3-3B-Instruct-2512 x PersonaMem
# Usage:
#   bash run/run_llama_personamem.sh [--gpu-memory FLOAT] [--max-model-len INT] \
#                                    [--module NAME|all] [--benchmark-size 32k|128k]
#   CUDA_VISIBLE_DEVICES=0 bash run/run_llama_personamem.sh --benchmark-size 128k

set -u

# ── defaults ──────────────────────────────────────
GPU_MEM=0.5
MAX_MODEL_LEN=8192
MODULE=all
BENCHMARK_SIZE=32k

# ── parse args ────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-memory)     GPU_MEM="$2"; shift 2 ;;
    --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
    --module)         MODULE="$2"; shift 2 ;;
    --benchmark-size) BENCHMARK_SIZE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--gpu-memory FLOAT] [--max-model-len INT] [--module NAME|all] [--benchmark-size 32k|128k]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ── benchmark-size → session range ────────────────
case "$BENCHMARK_SIZE" in
  32k)  START_SESSION=0; END_SESSION=36 ;;
  128k) START_SESSION=0; END_SESSION=59 ;;
  *)
    echo "Unsupported --benchmark-size: $BENCHMARK_SIZE (only 32k or 128k preset; edit script for 1M)" >&2
    exit 1 ;;
esac

# ── fixed config ──────────────────────────────────
MODEL="mistralai/Ministral-3-3B-Instruct-2512"
MODEL_TAG="mistral"
BENCH="personamem"
BENCH_TAG="personamem_${BENCHMARK_SIZE}"
TP=1

ALL_MODULES=(amem gmem ldagent memorybank ubllm theanine)

# ── paths / logs ──────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/run/logs/$MODEL_TAG"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/${BENCH_TAG}_MAIN_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1
echo "Master log: $MASTER_LOG"
echo "Model=$MODEL  Bench=$BENCH($BENCHMARK_SIZE)  Range=$START_SESSION-$END_SESSION  GPU_MEM=$GPU_MEM  MAX_LEN=$MAX_MODEL_LEN  MODULE=$MODULE  CUDA=${CUDA_VISIBLE_DEVICES:-unset}"
echo ""

# ── decide targets ────────────────────────────────
if [[ "$MODULE" == "all" ]]; then
  TARGETS=("${ALL_MODULES[@]}")
else
  TARGETS=("$MODULE")
fi

# ── runner ────────────────────────────────────────
run_one() {
  local mod="$1"
  local tag="${BENCH_TAG}_${mod}"
  local log="$LOG_DIR/$tag.log"

  echo "==> [$(date +%H:%M:%S)] START $tag"
  (
    cd "$REPO_ROOT/$BENCH/$mod"
    python run_experiment.py \
      --config config_0 \
      --start-session "$START_SESSION" \
      --end-session "$END_SESSION" \
      --benchmark-size "$BENCHMARK_SIZE" \
      --model "$MODEL" \
      --tensor-parallel "$TP" \
      --gpu-memory "$GPU_MEM" \
      --max-model-len "$MAX_MODEL_LEN"
  ) > "$log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && echo "<== DONE  $tag" || echo "<== FAIL  $tag (exit=$rc) -> $log"
}

# ── run ───────────────────────────────────────────
for m in "${TARGETS[@]}"; do
  run_one "$m"
done

echo ""
echo "All done. Master log: $MASTER_LOG"
