#!/usr/bin/env bash
# One-shot UUU-ONLY specialist (same knobs as continue-triple-c).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export FORCE_LIMIT="${FORCE_LIMIT:-40}"
export REWARD_MODE="${REWARD_MODE:-product}"
export WARMUP_UPDATES="${WARMUP_UPDATES:-100000}"
export WARMUP_HANG_START_P="${WARMUP_HANG_START_P:-0.05}"
export HANG_START_P="${HANG_START_P:-0.05}"
export NEAR_GOAL_P="${NEAR_GOAL_P:-0.3}"
export WRONG_EQ_P="${WRONG_EQ_P:-0.0}"
export UU_BIAS="${UU_BIAS:-1.0}"
export ANNEAL_UPDATES="${ANNEAL_UPDATES:-0}"
export GOAL_SWITCH_P="${GOAL_SWITCH_P:-0.0}"
export FOLD_PAIR_P="${FOLD_PAIR_P:-0.0}"
export CART_BARRIER_COEF="${CART_BARRIER_COEF:-50}"
export W_UP="${W_UP:-5.0}"
export INIT_MODE="${INIT_MODE:-mixed}"
export LR="${LR:-3e-4}"
export CHECKPOINT="${CHECKPOINT:-policies/checkpoint-triple-c.pt}"
export OUT="${OUT:-policies/policy-triple-c.json}"
export RUN_NAME="${RUN_NAME:-ft-triple-c-e${NUM_ENVS:-8192}-r${ROLLOUT:-256}-uuu-only-f${FORCE_LIMIT}}"

exec bash scripts/next-train-triple.sh "$@"
