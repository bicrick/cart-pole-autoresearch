#!/usr/bin/env bash
# Aggressive local W1 TQC UUU hold on Apple Silicon.
# Batched physics + TQC on MPS. BC warm-start from LQR demos.
# Does not touch GCP.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

export DEVICE="${DEVICE:-mps}"
export ENV_DEVICE="${ENV_DEVICE:-mps}"
export N_ENVS="${N_ENVS:-16}"
export GRADIENT_STEPS="${GRADIENT_STEPS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export BUFFER_SIZE="${BUFFER_SIZE:-200000}"
export TOTAL_STEPS="${TOTAL_STEPS:-150000}"
export INIT_NOISE="${INIT_NOISE:-0.02}"
export INIT_NOISE_MIN="${INIT_NOISE_MIN:-0}"
export TRACK_WALLS="${TRACK_WALLS:-1}"
export PROGRESS_W="${PROGRESS_W:-0}"
export SPARSE_BONUS="${SPARSE_BONUS:-1.0}"
export LOG_STD_INIT="${LOG_STD_INIT:--4.0}"
export ENT_COEF="${ENT_COEF:-0.001}"
export LEARNING_STARTS="${LEARNING_STARTS:-0}"
export BC_REG_COEF="${BC_REG_COEF:-0}"
export GOAL="${GOAL:-UUU}"
if [[ "${GOAL}" != "UUU" ]]; then
  # Vanilla fawraw M2: no UUU LQR-BC, no actor freeze.
  export BC_CHECKPOINT="${BC_CHECKPOINT_OVERRIDE:-}"
  export ACTOR_FREEZE_STEPS="${ACTOR_FREEZE_STEPS:-0}"
else
  export ACTOR_FREEZE_STEPS="${ACTOR_FREEZE_STEPS:-20000}"
  # Set BC_CHECKPOINT_OVERRIDE (even empty) to skip the UUU hold anchor,
  # e.g. a from-bottom swing net.
  if [[ -n "${BC_CHECKPOINT_OVERRIDE+x}" ]]; then
    export BC_CHECKPOINT="${BC_CHECKPOINT_OVERRIDE}"
  else
    export BC_CHECKPOINT="${BC_CHECKPOINT:-policies/bc-lqr-uuu.pt}"
  fi
fi
export RESUME_FROM="${RESUME_FROM:-}"
export VER="${VER:-0}"
export RUN_NAME="${RUN_NAME:-m2-hold-${GOAL}-mac-w1-nt${INIT_NOISE}-n${N_ENVS}}"
export CHECKPOINT="${CHECKPOINT:-policies/tqc-m2-${GOAL}-hold-mac.zip}"
export LOGDIR="${LOGDIR:-runs}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  export TOTAL_STEPS=2500
  export N_ENVS=8
  export GRADIENT_STEPS=4
  export BUFFER_SIZE=10000
  export RUN_NAME="${RUN_NAME}-smoke"
  export CHECKPOINT="policies/tqc-m2-${GOAL}-hold-mac-smoke.zip"
fi

echo "=== Apple Silicon W1 TQC hold ===" >&2
echo "  DEVICE=$DEVICE ENV_DEVICE=$ENV_DEVICE N_ENVS=$N_ENVS GRADIENT_STEPS=$GRADIENT_STEPS" >&2
echo "  TOTAL_STEPS=$TOTAL_STEPS BATCH=$BATCH_SIZE BUFFER=$BUFFER_SIZE" >&2
echo "  BC=$BC_CHECKPOINT run=$RUN_NAME" >&2

ENV_DEVICE_ARGS=()
if [[ -n "${ENV_DEVICE}" ]]; then
  ENV_DEVICE_ARGS+=(--env-device "$ENV_DEVICE")
fi

exec bash "$ROOT/scripts/next-train-triple-m2-hold.sh" \
  "${ENV_DEVICE_ARGS[@]}" \
  "$@"
