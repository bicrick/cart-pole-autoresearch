#!/usr/bin/env bash
# fawraw M2 UUU hold — faithful reimpl on our plant (quiet-basin + TQC + product).
# Spec: fawraw m2_upright_tqc.yaml + env contract (quiet rates ±0.01, fall-kill 0.6,
#        progress_w=0). Primary success = survival (ep_len >= 0.8*max_steps).
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
INIT_NOISE_MIN="${INIT_NOISE_MIN:-0}"
HANG_FRAC="${HANG_FRAC:-0.0}"
WIDE_FRAC="${WIDE_FRAC:-0.0}"
PROGRESS_W="${PROGRESS_W:-0}"
CART_BARRIER_COEF="${CART_BARRIER_COEF:-10}"
REWARD_MODE="${REWARD_MODE:-product}"
VEL_COST_COEF="${VEL_COST_COEF:-0.02}"
CART_COST_COEF="${CART_COST_COEF:-0.2}"
TRANSITION_BONUS="${TRANSITION_BONUS:-0}"
TRANSITION_TOL="${TRANSITION_TOL:-0.3}"
TRANSITION_STEPS="${TRANSITION_STEPS:-100}"
SOFT_LANDING_RAD="${SOFT_LANDING_RAD:-0}"
VEL_NEAR_GAIN="${VEL_NEAR_GAIN:-1}"
VEL_NEAR_RAD="${VEL_NEAR_RAD:-0.3}"
EXCESS_ENERGY_COEF="${EXCESS_ENERGY_COEF:-0}"
ARRIVAL_FRAC="${ARRIVAL_FRAC:-0}"
ARRIVAL_OMEGA="${ARRIVAL_OMEGA:-0}"
LQR_BC_COEF="${LQR_BC_COEF:-0}"
LQR_BC_ANG="${LQR_BC_ANG:-0.35}"
LQR_BC_OMEGA="${LQR_BC_OMEGA:-0.40}"
LQR_SEED_EPISODES="${LQR_SEED_EPISODES:-0}"
ALPHA_TH="${ALPHA_TH:-0.5}"
W_UP="${W_UP:-5.0}"
W_DOWN="${W_DOWN:-1.0}"
SPARSE_BONUS="${SPARSE_BONUS:-1.0}"
LOG_STD_INIT="${LOG_STD_INIT:--4.0}"
ENT_COEF="${ENT_COEF:-0.001}"
LEARNING_STARTS="${LEARNING_STARTS:-0}"
ACTOR_FREEZE_STEPS="${ACTOR_FREEZE_STEPS:-20000}"
BC_REG_COEF="${BC_REG_COEF:-0}"
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
N_ENVS="${N_ENVS:-1}"
GRADIENT_STEPS="${GRADIENT_STEPS:-1}"
ENV_DEVICE="${ENV_DEVICE:-}"
DEVICE="${DEVICE:-auto}"
LOGDIR="${LOGDIR:-runs}"
RUN_NAME="${RUN_NAME:-m2-hold-uuu-f${FORCE_LIMIT}-nt${INIT_NOISE}-tqc}"
CHECKPOINT="${CHECKPOINT:-policies/tqc-m2-uuu-hold.zip}"
# Walls ON by default for our plant
TRACK_WALLS="${TRACK_WALLS:-1}"
TRACK_LIMIT="${TRACK_LIMIT:-2.4}"

BC_CHECKPOINT="${BC_CHECKPOINT:-}"
RESUME_FROM="${RESUME_FROM:-}"
VER="${VER:-0}"
HARD_IC_PATH="${HARD_IC_PATH:-}"
HARD_IC_FRAC="${HARD_IC_FRAC:-0}"
GOAL="${GOAL:-UUU}"
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

BC_ARGS=()
if [[ -n "${BC_CHECKPOINT}" ]]; then
  BC_ARGS+=(--bc-checkpoint "$BC_CHECKPOINT")
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  RESUME_ARGS+=(--resume-from "$RESUME_FROM")
fi
if [[ "${RESET_REPLAY:-0}" == "1" ]]; then
  RESUME_ARGS+=(--reset-replay-buffer)
fi

VER_ARGS=()
if [[ "${VER}" == "1" || "${VER}" == "true" || "${VER}" == "on" ]]; then
  VER_ARGS+=(--ver)
fi

HARD_IC_ARGS=()
if [[ -n "${HARD_IC_PATH}" ]]; then
  HARD_IC_ARGS+=(--hard-ic-path "$HARD_IC_PATH" --hard-ic-frac "$HARD_IC_FRAC")
fi

echo "=== launch M2 ${GOAL} hold ===" >&2
echo "  GOAL=$GOAL TOTAL_STEPS=$TOTAL_STEPS INIT_NOISE=$INIT_NOISE walls=$TRACK_WALLS track=$TRACK_LIMIT" >&2
echo "  net=[$POLICY_ARCH] buffer=$BUFFER_SIZE n_quantiles=$N_QUANTILES" >&2
echo "  PRIMARY: survival_success (ep_len>=0.8*max); also at_goal; progress_w=$PROGRESS_W reward=$REWARD_MODE" >&2
echo "  n_envs=$N_ENVS gradient_steps=$GRADIENT_STEPS ckpt=$CHECKPOINT run=$RUN_NAME device=$DEVICE env_device=${ENV_DEVICE:-$DEVICE}" >&2
if [[ -n "${BC_CHECKPOINT}" ]]; then echo "  bc_warmstart=$BC_CHECKPOINT" >&2; fi
if [[ -n "${RESUME_FROM}" ]]; then echo "  resume_from=$RESUME_FROM" >&2; fi
if [[ ${#VER_ARGS[@]} -gt 0 ]]; then echo "  ver=1 (Baek left-right replay flip)" >&2; fi
if [[ -n "${HARD_IC_PATH}" ]]; then echo "  hard_ics=$HARD_IC_PATH frac=$HARD_IC_FRAC" >&2; fi

exec "$PYTHON" train/train_triple_tqc.py \
  --goal "$GOAL" \
  --total-steps "$TOTAL_STEPS" \
  --force-limit "$FORCE_LIMIT" \
  --init-mode "$INIT_MODE" \
  --init-noise "$INIT_NOISE" \
  --init-noise-min "$INIT_NOISE_MIN" \
  --hang-frac "$HANG_FRAC" \
  --wide-frac "$WIDE_FRAC" \
  --progress-w "$PROGRESS_W" \
  --cart-barrier-coef "$CART_BARRIER_COEF" \
  --reward-mode "$REWARD_MODE" \
  --vel-cost-coef "$VEL_COST_COEF" \
  --cart-cost-coef "$CART_COST_COEF" \
  --transition-bonus "$TRANSITION_BONUS" \
  --transition-tol "$TRANSITION_TOL" \
  --transition-steps "$TRANSITION_STEPS" \
  --soft-landing-rad "$SOFT_LANDING_RAD" \
  --vel-near-gain "$VEL_NEAR_GAIN" \
  --vel-near-rad "$VEL_NEAR_RAD" \
  --excess-energy-coef "$EXCESS_ENERGY_COEF" \
  --arrival-frac "$ARRIVAL_FRAC" \
  --arrival-omega "$ARRIVAL_OMEGA" \
  --lqr-bc-coef "$LQR_BC_COEF" \
  --lqr-bc-ang "$LQR_BC_ANG" \
  --lqr-bc-omega "$LQR_BC_OMEGA" \
  --lqr-seed-episodes "$LQR_SEED_EPISODES" \
  --alpha-th "$ALPHA_TH" \
  --w-up "$W_UP" \
  --w-down "$W_DOWN" \
  --sparse-bonus "$SPARSE_BONUS" \
  --log-std-init "$LOG_STD_INIT" \
  --ent-coef "$ENT_COEF" \
  --lr "$LR" \
  --learning-starts "$LEARNING_STARTS" \
  --actor-freeze-steps "$ACTOR_FREEZE_STEPS" \
  --bc-reg-coef "$BC_REG_COEF" \
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
  --n-envs "$N_ENVS" \
  --gradient-steps "$GRADIENT_STEPS" \
  --device "$DEVICE" \
  ${ENV_DEVICE:+--env-device "$ENV_DEVICE"} \
  --logdir "$LOGDIR" \
  --run-name "$RUN_NAME" \
  --checkpoint "$CHECKPOINT" \
  --track-limit "$TRACK_LIMIT" \
  ${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"} \
  ${WALLS_ARGS[@]+"${WALLS_ARGS[@]}"} \
  ${BC_ARGS[@]+"${BC_ARGS[@]}"} \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  ${VER_ARGS[@]+"${VER_ARGS[@]}"} \
  ${HARD_IC_ARGS[@]+"${HARD_IC_ARGS[@]}"} \
  "$@"
