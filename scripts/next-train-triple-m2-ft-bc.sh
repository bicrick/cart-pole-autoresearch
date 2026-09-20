#!/usr/bin/env bash
# Short TQC fine-tune from BC LQR UUU hold policy (quiet-basin M2 env contract).
# Warm-starts TQC actor from policies/bc-lqr-uuu.pt; critics from scratch.
# Defaults: TOTAL_STEPS=40k, INIT_NOISE=0.02, DEVICE=cuda, walls ON, progress_w=0.
# One-shot launch (no babysit/continue watchers).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TOTAL_STEPS="${TOTAL_STEPS:-40000}"
export INIT_NOISE="${INIT_NOISE:-0.02}"
export DEVICE="${DEVICE:-cuda}"
export RUN_NAME="${RUN_NAME:-m2-ft-bc-nt${INIT_NOISE}-t${TOTAL_STEPS}}"
export CHECKPOINT="${CHECKPOINT:-policies/tqc-m2-ft-bc.zip}"
export BC_CHECKPOINT="${BC_CHECKPOINT:-policies/bc-lqr-uuu.pt}"
export LEARNING_STARTS="${LEARNING_STARTS:-1000}"
export SAVE_FREQ="${SAVE_FREQ:-10000}"
export EVAL_FREQ="${EVAL_FREQ:-5000}"

echo "=== M2 FT from BC ===" >&2
echo "  BC=$BC_CHECKPOINT TOTAL_STEPS=$TOTAL_STEPS INIT_NOISE=$INIT_NOISE DEVICE=$DEVICE" >&2
echo "  run=$RUN_NAME ckpt=$CHECKPOINT" >&2

exec env TOTAL_STEPS="$TOTAL_STEPS" INIT_NOISE="$INIT_NOISE" DEVICE="$DEVICE" \
  RUN_NAME="$RUN_NAME" CHECKPOINT="$CHECKPOINT" BC_CHECKPOINT="$BC_CHECKPOINT" \
  bash "$ROOT/scripts/next-train-triple-m2-hold.sh" \
  --learning-starts "$LEARNING_STARTS" \
  --save-freq "$SAVE_FREQ" \
  --eval-freq "$EVAL_FREQ" \
  "$@"
