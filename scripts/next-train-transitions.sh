#!/usr/bin/env bash
# Transition-only FT: every reset is discrete eq A → goal B≠A (all directed pairs).
# Mid-episode flips always change the goal. No hang / near-goal / fold mix-ins.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-16384}"
UPDATES="${UPDATES:-400}"
LOGDIR="${LOGDIR:-runs}"
ROLLOUT="${ROLLOUT:-256}"
EPISODE_LEN="${EPISODE_LEN:-1200}"
GOAL_SWITCH_P="${GOAL_SWITCH_P:-0.01}"
ENERGY_W="${ENERGY_W:-0.35}"
HER_RATIO="${HER_RATIO:-0.1}"
IMPULSE_P="${IMPULSE_P:-0.01}"
RUN_NAME="${RUN_NAME:-ft-e${NUM_ENVS}-r${ROLLOUT}-xonly-gsw01}"

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
  --warmup-updates 0 \
  --uu-bias 0 \
  --near-goal-p 0 \
  --hang-start-p 0 \
  --wrong-eq-p 0 \
  --fold-pair-p 0 \
  --transition-only \
  --goal-switch-p "${GOAL_SWITCH_P}" \
  --her-ratio "${HER_RATIO}" \
  --impulse-p "${IMPULSE_P}" \
  --logdir "${LOGDIR}" \
  --run-name "${RUN_NAME}" \
  --checkpoint "${CHECKPOINT:-policies/checkpoint.pt}" \
  --out "${OUT:-policies/policy-xonly.json}" \
  "$@"
