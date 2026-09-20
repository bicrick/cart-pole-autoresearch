#!/usr/bin/env bash
# fawraw M2 UUU hold — faithful reimpl on our plant (quiet-basin + TQC + product).
# Spec: training/configs/m2_upright_tqc.yaml intent + Lim product reward.
# Walls ON (our inelastic endstops = training wheels; Lim/fawraw use rail limits).
# Do NOT start GCP VM / long train unless user green-lights GPU.
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

# --- fawraw M2 defaults (override via env) ---
TOTAL_STEPS="${TOTAL_STEPS:-150000}"
FORCE_LIMIT="${FORCE_LIMIT:-40}"
INIT_MODE="${INIT_MODE:-near_target}"
INIT_NOISE="${INIT_NOISE:-0.05}"
HANG_FRAC="${HANG_FRAC:-0.0}"
WIDE_FRAC="${WIDE_FRAC:-0.0}"
PROGRESS_W="${PROGRESS_W:-1.0}"
CART_BARRIER_COEF="${CART_BARRIER_COEF:-10}"
ALPHA_TH="${ALPHA_TH:-0.5}"
W_UP="${W_UP:-5.0}"
W_DOWN="${W_DOWN:-1.0}"
LR="${LR:-3e-4}"
BUFFER_SIZE="${BUFFER_SIZE:-200000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GAMMA="${GAMMA:-0.99}"
TAU="${TAU:-0.005}"
POLICY_ARCH="${POLICY_ARCH:-128,128}"
CRITIC_ARCH="${CRITIC_ARCH:-128,128}"
N_QUANTILES="${N_QUANTILES:-20}"
N_CRITICS="${N_CRITICS:-3}"
TOP_DROP="${TOP_DROP:-2}"
MAX_STEPS="${MAX_STEPS:-1000}"
DEVICE="${DEVICE:-auto}"
LOGDIR="${LOGDIR:-runs}"
RUN_NAME="${RUN_NAME:-m2-hold-uuu-f${FORCE_LIMIT}-nt${INIT_NOISE}-tqc}"
CHECKPOINT="${CHECKPOINT:-policies/tqc-m2-uuu-hold.zip}"
# Walls ON by default for our plant
TRACK_WALLS="${TRACK_WALLS:-1}"

SMOKE_FLAG=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE_FLAG+=(--smoke)
  RUN_NAME="${RUN_NAME}-smoke"
  DEVICE="${DEVICE:-cpu}"
  if [[ "${DEVICE}" == "auto" ]]; then
    DEVICE="cpu"
  fi
fi

mkdir -p policies logs "$LOGDIR"

WALLS_ARGS=()
if [[ "${TRACK_WALLS}" == "1" || "${TRACK_WALLS}" == "true" || "${TRACK_WALLS}" == "on" ]]; then
  WALLS_ARGS+=(--track-walls)
  if [[ "${RUN_NAME}" != *walls* && "${RUN_NAME}" != *m2-hold* ]]; then
    RUN_NAME="${RUN_NAME}-walls"
  fi
elif [[ "${TRACK_WALLS}" == "0" || "${TRACK_WALLS}" == "false" || "${TRACK_WALLS}" == "off" ]]; then
  WALLS_ARGS+=(--no-track-walls)
fi

echo "=== launch M2 UUU hold ===" >&2
echo "  TOTAL_STEPS=$TOTAL_STEPS INIT_NOISE=$INIT_NOISE walls=$TRACK_WALLS" >&2
echo "  net=[$POLICY_ARCH] buffer=$BUFFER_SIZE n_quantiles=$N_QUANTILES" >&2
echo "  PRIMARY METRICS: success_rate / at_goal (not ep_rew_mean alone)" >&2
echo "  ckpt=$CHECKPOINT run=$RUN_NAME device=$DEVICE" >&2

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
  --gamma "$GAMMA" \
  --tau "$TAU" \
  --policy-arch "$POLICY_ARCH" \
  --critic-arch "$CRITIC_ARCH" \
  --n-quantiles "$N_QUANTILES" \
  --n-critics "$N_CRITICS" \
  --top-quantiles-to-drop "$TOP_DROP" \
  --max-steps "$MAX_STEPS" \
  --device "$DEVICE" \
  --logdir "$LOGDIR" \
  --run-name "$RUN_NAME" \
  --checkpoint "$CHECKPOINT" \
  "${SMOKE_FLAG[@]}" \
  "${WALLS_ARGS[@]}" \
  "$@"
