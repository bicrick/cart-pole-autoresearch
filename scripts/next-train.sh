#!/usr/bin/env bash
# Restart command AFTER the current GCP train (updates ~370/400) finishes.
# Do not launch a second GPU job while the first is still running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-4096}"
UPDATES="${UPDATES:-400}"
LOGDIR="${LOGDIR:-runs}"

exec python3 train/train.py \
  --num-envs "${NUM_ENVS}" \
  --updates "${UPDATES}" \
  --rollout 128 \
  --episode-len 800 \
  --track-limit 4.0 \
  --reward-clip 8.0 \
  --align-w 1.5 \
  --energy-w 0.15 \
  --spin-w 0.0003 \
  --warmup-updates 80 \
  --warmup-goal UU \
  --near-goal-p 0.5 \
  --her-ratio 0.3 \
  --impulse-p 0.005 \
  --logdir "${LOGDIR}" \
  --checkpoint policies/checkpoint.pt \
  --out policies/policy.json \
  "$@"
