#!/usr/bin/env bash
# Fine-tune from policies/checkpoint.pt (hard walls + center + soft UU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-32768}"
UPDATES="${UPDATES:-400}"
LOGDIR="${LOGDIR:-runs}"
ROLLOUT="${ROLLOUT:-256}"
WARMUP_UPDATES="${WARMUP_UPDATES:-20}"
UU_BIAS="${UU_BIAS:-0.55}"
ANNEAL_UPDATES="${ANNEAL_UPDATES:-0}"
RUN_NAME="${RUN_NAME:-ft-e${NUM_ENVS}-r${ROLLOUT}-hardwalls-center-uub055}"

exec python3 train/train.py \
  --num-envs "${NUM_ENVS}" \
  --updates "${UPDATES}" \
  --rollout "${ROLLOUT}" \
  --episode-len 800 \
  --track-limit 2.4 \
  --reward-clip 8.0 \
  --align-w 1.5 \
  --energy-w 0.15 \
  --spin-w 0.0003 \
  --center-w "${CENTER_W:-0.06}" \
  --center-hold-w "${CENTER_HOLD_W:-0.18}" \
  --warmup-updates "${WARMUP_UPDATES}" \
  --warmup-goal UU \
  --uu-bias "${UU_BIAS}" \
  --anneal-updates "${ANNEAL_UPDATES}" \
  --near-goal-p 0.5 \
  --her-ratio 0.3 \
  --impulse-p 0.005 \
  --logdir "${LOGDIR}" \
  --run-name "${RUN_NAME}" \
  --checkpoint policies/checkpoint.pt \
  --out policies/policy.json \
  "$@"
