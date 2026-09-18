#!/usr/bin/env bash
# Fine-tune from policies/checkpoint.pt (hard walls + center + soft UU).
# her_ratio default 0.1 per lessons (n)/(s)/(t)/(ab).
# center_hold_w default 0.30 per lessons (v)–(ab) — raise hold pressure before raising HER.
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
HER_RATIO="${HER_RATIO:-0.1}"
RUN_NAME="${RUN_NAME:-ft-e${NUM_ENVS}-r${ROLLOUT}-hardwalls-center-ch030-uub055-her01}"

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
  --center-hold-w "${CENTER_HOLD_W:-0.30}" \
  --warmup-updates "${WARMUP_UPDATES}" \
  --warmup-goal UU \
  --uu-bias "${UU_BIAS}" \
  --anneal-updates "${ANNEAL_UPDATES}" \
  --near-goal-p 0.5 \
  --her-ratio "${HER_RATIO}" \
  --impulse-p 0.005 \
  --logdir "${LOGDIR}" \
  --run-name "${RUN_NAME}" \
  --checkpoint policies/checkpoint.pt \
  --out policies/policy.json \
  "$@"
