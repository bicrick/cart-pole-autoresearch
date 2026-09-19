#!/usr/bin/env bash
# Edge-case FT from policies/checkpoint.pt:
# hang→goal swing-up, wrong-eq starts, mid-episode goal flips,
# stronger energy, longer episodes. Keep low HER + center hold.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-32768}"
UPDATES="${UPDATES:-400}"
LOGDIR="${LOGDIR:-runs}"
ROLLOUT="${ROLLOUT:-256}"
WARMUP_UPDATES="${WARMUP_UPDATES:-0}"
UU_BIAS="${UU_BIAS:-0.55}"
ANNEAL_UPDATES="${ANNEAL_UPDATES:-0}"
HER_RATIO="${HER_RATIO:-0.1}"
NEAR_GOAL_P="${NEAR_GOAL_P:-0.15}"
HANG_START_P="${HANG_START_P:-0.45}"
WRONG_EQ_P="${WRONG_EQ_P:-0.25}"
GOAL_SWITCH_P="${GOAL_SWITCH_P:-0.004}"
FOLD_PAIR_P="${FOLD_PAIR_P:-0.55}"
ENERGY_W="${ENERGY_W:-0.35}"
EPISODE_LEN="${EPISODE_LEN:-1200}"
IMPULSE_P="${IMPULSE_P:-0.01}"
RUN_NAME="${RUN_NAME:-ft-e${NUM_ENVS}-r${ROLLOUT}-edge-hang-xeq-gsw-ch030-her01}"

exec python3 train/train.py \
  --num-envs "${NUM_ENVS}" \
  --updates "${UPDATES}" \
  --rollout "${ROLLOUT}" \
  --episode-len "${EPISODE_LEN}" \
  --track-limit 2.4 \
  --oob-penalty "${OOB_PENALTY:-20}" \
  --reward-clip 8.0 \
  --align-w 1.5 \
  --energy-w "${ENERGY_W}" \
  --spin-w 0.0003 \
  --center-w "${CENTER_W:-0.06}" \
  --center-hold-w "${CENTER_HOLD_W:-0.30}" \
  --warmup-updates "${WARMUP_UPDATES}" \
  --warmup-goal UU \
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
  --checkpoint policies/checkpoint.pt \
  --out policies/policy.json \
  "$@"
