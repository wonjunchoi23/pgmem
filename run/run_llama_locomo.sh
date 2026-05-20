#!/bin/bash
# Llama-3.2-3B-Instruct x LoComo
# Usage:
#   bash run/run_llama_locomo.sh [--gpu-memory FLOAT] [--max-model-len INT] [--module NAME|all]
#   CUDA_VISIBLE_DEVICES=0 bash run/run_llama_locomo.sh --gpu-memory 0.5

set -u

# ── defaults ──────────────────────────────────────
GPU_MEM=0.5
MAX_MODEL_LEN=8192
MODULE=all

# ── parse args ────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-memory)    GPU_MEM="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --module)        MODULE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--gpu-memory FLOAT] [--max-model-len INT] [--module NAME|all]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ── fixed config ──────────────────────────────────
MODEL="meta-llama/Llama-3.2-3B-Instruct"
MODEL_TAG="llama"
BENCH="locomo"
BENCH_TAG="locomo"
START_SAMPLE=0
END_SAMPLE=9
TP=1

ALL_MODULES=(amem gmem ldagent memorybank ubllm theanine)

# ── paths / logs ──────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/run/logs/$MODEL_TAG"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/${BENCH_TAG}_MAIN_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1
echo "Master log: $MASTER_LOG"
echo "Model=$MODEL  Bench=$BENCH  GPU_MEM=$GPU_MEM  MAX_LEN=$MAX_MODEL_LEN  MODULE=$MODULE  CUDA=${CUDA_VISIBLE_DEVICES:-unset}"
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
      --start-sample "$START_SAMPLE" \
      --end-sample "$END_SAMPLE" \
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
