#!/usr/bin/env bash
# Fine-tune from policies/checkpoint.pt on walled plant + soft UU anneal.
# Paper-recipe run finished healthy (reward~416, align~0.32) but UU weak;
# hard warmup cut diluted UU — keep P(UU)>=0.55 after a short capture warmup.
# Do not launch a second GPU job while one is still running.
set -euo pipefail
# Throughput defaults: large env count + longer rollout to fill the T4.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-8192}"
UPDATES="${UPDATES:-400}"
LOGDIR="${LOGDIR:-runs}"
# Short pure-UU warmup when fine-tuning; soft bias covers the rest.
WARMUP_UPDATES="${WARMUP_UPDATES:-40}"
UU_BIAS="${UU_BIAS:-0.55}"
ANNEAL_UPDATES="${ANNEAL_UPDATES:-0}"

exec python3 train/train.py \
  --num-envs "${NUM_ENVS}" \
  --updates "${UPDATES}" \
  --rollout "${ROLLOUT:-256}" \
  --episode-len 800 \
  --track-limit 2.4 \
  --reward-clip 8.0 \
  --align-w 1.5 \
  --energy-w 0.15 \
  --spin-w 0.0003 \
  --warmup-updates "${WARMUP_UPDATES}" \
  --warmup-goal UU \
  --uu-bias "${UU_BIAS}" \
  --anneal-updates "${ANNEAL_UPDATES}" \
  --near-goal-p 0.5 \
  --her-ratio 0.3 \
  --impulse-p 0.005 \
  --logdir "${LOGDIR}" \
  --checkpoint policies/checkpoint.pt \
  --out policies/policy.json \
  "$@"
