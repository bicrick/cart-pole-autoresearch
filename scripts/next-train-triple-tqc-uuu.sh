#!/usr/bin/env bash
# TQC UUU specialist — Lim KIEE 2025 recipe (first of 8 EP policies).
# Does NOT kill live PPO a/b/c. Swap onto a free L4 when ready (prefer C).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/train${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv-tqc/bin/python" ]]; then
    PYTHON="$ROOT/.venv-tqc/bin/python"
  else
    PYTHON="python3"
  fi
fi

TOTAL_STEPS="${TOTAL_STEPS:-300000}"
FORCE_LIMIT="${FORCE_LIMIT:-40}"
INIT_MODE="${INIT_MODE:-near_target}"
INIT_NOISE="${INIT_NOISE:-0.15}"
HANG_FRAC="${HANG_FRAC:-0.05}"
WIDE_FRAC="${WIDE_FRAC:-0.25}"
PROGRESS_W="${PROGRESS_W:-1.0}"
CART_BARRIER_COEF="${CART_BARRIER_COEF:-10}"
ALPHA_TH="${ALPHA_TH:-0.5}"
W_UP="${W_UP:-5.0}"
W_DOWN="${W_DOWN:-1.0}"
LR="${LR:-3e-4}"
BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-auto}"
LOGDIR="${LOGDIR:-runs}"
RUN_NAME="${RUN_NAME:-tqc-uuu-f${FORCE_LIMIT}-nt${INIT_NOISE}-w${WIDE_FRAC}-h${HANG_FRAC}}"
CHECKPOINT="${CHECKPOINT:-policies/tqc-triple-uuu.zip}"
POLICY_ARCH="${POLICY_ARCH:-400,300}"
CRITIC_ARCH="${CRITIC_ARCH:-512,512,512}"
N_QUANTILES="${N_QUANTILES:-25}"
N_CRITICS="${N_CRITICS:-3}"
TOP_DROP="${TOP_DROP:-2}"

SMOKE_FLAG=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE_FLAG+=(--smoke)
  RUN_NAME="${RUN_NAME}-smoke"
fi

mkdir -p policies logs "$LOGDIR"

exec "$PYTHON" train/train_triple_tqc.py \
  --total-steps "$TOTAL_STEPS" \
  --force-limit "$FORCE_LIMIT" \
  --init-mode "$INIT_MODE" \
  --init-noise "$INIT_NOISE" \
  --hang-frac "$HANG_FRAC" \
  --wide-frac "$WIDE_FRAC" \
  --progress-w "$PROGRESS_W" \
  --cart-barrier-coef "$CART_BARRIER_COEF" \
  --alpha-th "$ALPHA_TH" \
  --w-up "$W_UP" \
  --w-down "$W_DOWN" \
  --lr "$LR" \
  --buffer-size "$BUFFER_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --logdir "$LOGDIR" \
  --run-name "$RUN_NAME" \
  --checkpoint "$CHECKPOINT" \
  --policy-arch "$POLICY_ARCH" \
  --critic-arch "$CRITIC_ARCH" \
  --n-quantiles "$N_QUANTILES" \
  --n-critics "$N_CRITICS" \
  --top-quantiles-to-drop "$TOP_DROP" \
  --gamma 0.99 \
  --tau 0.005 \
  "${SMOKE_FLAG[@]}" \
  "$@"
