#!/usr/bin/env bash
# Cart-triple PPO — Lim/fawraw product-reward recipe (UUU-first → multi-eq).
# Keep PPO; product reward on world angles; low hang; cart barrier; fall grace.
# forceLimit default 40 N (was 20; underpowered for 3×0.5m + friction — see research notes).
# NOT transition-only (leave that for double xonly — currently killed for triple L4 slots).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-8192}"
UPDATES="${UPDATES:-400}"
LOGDIR="${LOGDIR:-runs}"
ROLLOUT="${ROLLOUT:-256}"
WARMUP_UPDATES="${WARMUP_UPDATES:-80}"
UU_BIAS="${UU_BIAS:-0.40}"
ANNEAL_UPDATES="${ANNEAL_UPDATES:-0}"
HER_RATIO="${HER_RATIO:-0.1}"
NEAR_GOAL_P="${NEAR_GOAL_P:-0.25}"
HANG_START_P="${HANG_START_P:-0.10}"
WARMUP_HANG_START_P="${WARMUP_HANG_START_P:-0.0}"
WRONG_EQ_P="${WRONG_EQ_P:-0.20}"
GOAL_SWITCH_P="${GOAL_SWITCH_P:-0.004}"
FOLD_PAIR_P="${FOLD_PAIR_P:-0.40}"
ENERGY_W="${ENERGY_W:-0.15}"
EPISODE_LEN="${EPISODE_LEN:-1200}"
IMPULSE_P="${IMPULSE_P:-0.01}"
LR="${LR:-3e-4}"
REWARD_MODE="${REWARD_MODE:-product}"
CART_BARRIER_COEF="${CART_BARRIER_COEF:-50}"
W_UP="${W_UP:-5.0}"
W_DOWN="${W_DOWN:-1.0}"
ALPHA_TH="${ALPHA_TH:-0.5}"
FALL_GRACE_STEPS="${FALL_GRACE_STEPS:-20}"
START_GRACE_STEPS="${START_GRACE_STEPS:-0}"
INIT_MODE="${INIT_MODE:-mixed}"
FORCE_LIMIT="${FORCE_LIMIT:-40}"
PROGRESS_W="${PROGRESS_W:-}"
FLIP_AUGMENT="${FLIP_AUGMENT:-1}"
RUN_NAME="${RUN_NAME:-ft-triple-e${NUM_ENVS}-r${ROLLOUT}-prod-uuu-f${FORCE_LIMIT}}"

FORCE_ARGS=()
if [[ -n "${FORCE_LIMIT}" ]]; then
  FORCE_ARGS+=(--force-limit "${FORCE_LIMIT}")
fi
if [[ -n "${PROGRESS_W}" ]]; then
  FORCE_ARGS+=(--progress-w "${PROGRESS_W}")
fi
# flip-augment default on in train_triple; allow FLIP_AUGMENT=0 to disable
if [[ "${FLIP_AUGMENT}" == "0" || "${FLIP_AUGMENT}" == "false" || "${FLIP_AUGMENT}" == "off" ]]; then
  FORCE_ARGS+=(--no-flip-augment)
else
  FORCE_ARGS+=(--flip-augment)
fi

exec python3 train/train_triple.py \
  --num-envs "${NUM_ENVS}" \
  --updates "${UPDATES}" \
  --rollout "${ROLLOUT}" \
  --episode-len "${EPISODE_LEN}" \
  --lr "${LR}" \
  --track-limit 2.4 \
  --oob-penalty "${OOB_PENALTY:-20}" \
  --reward-clip "${REWARD_CLIP:-8.0}" \
  --reward-mode "${REWARD_MODE}" \
  --cart-barrier-coef "${CART_BARRIER_COEF}" \
  --w-up "${W_UP}" \
  --w-down "${W_DOWN}" \
  --alpha-th "${ALPHA_TH}" \
  --alpha-u "${ALPHA_U:-0.0}" \
  --alpha-y "${ALPHA_Y:-0.0}" \
  --alpha-w "${ALPHA_W:-0.0}" \
  --fall-grace-steps "${FALL_GRACE_STEPS}" \
  --start-grace-steps "${START_GRACE_STEPS}" \
  --init-mode "${INIT_MODE}" \
  --align-w 1.5 \
  --energy-w "${ENERGY_W}" \
  --spin-w 0.0003 \
  --center-w "${CENTER_W:-0.06}" \
  --center-hold-w "${CENTER_HOLD_W:-0.30}" \
  --warmup-updates "${WARMUP_UPDATES}" \
  --warmup-goal UUU \
  --warmup-hang-start-p "${WARMUP_HANG_START_P}" \
  --uu-bias "${UU_BIAS}" \
  --anneal-updates "${ANNEAL_UPDATES}" \
  --near-goal-p "${NEAR_GOAL_P}" \
  --hang-start-p "${HANG_START_P}" \
  --wrong-eq-p "${WRONG_EQ_P}" \
  --goal-switch-p "${GOAL_SWITCH_P}" \
  --fold-pair-p "${FOLD_PAIR_P}" \
  --her-ratio "${HER_RATIO}" \
  --impulse-p "${IMPULSE_P}" \
  --logdir "${LOGDIR}" \
  --run-name "${RUN_NAME}" \
  --checkpoint "${CHECKPOINT:-policies/checkpoint-triple.pt}" \
  --out "${OUT:-policies/policy-triple.json}" \
  "${FORCE_ARGS[@]}" \
  "$@"
