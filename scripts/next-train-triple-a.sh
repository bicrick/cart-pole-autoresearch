#!/usr/bin/env bash
# One-shot UUU swing-up specialist (same knobs as continue-triple-a).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export FORCE_LIMIT="${FORCE_LIMIT:-50}"
export REWARD_MODE="${REWARD_MODE:-product}"
export PROGRESS_W="${PROGRESS_W:-1.0}"
export FLIP_AUGMENT="${FLIP_AUGMENT:-1}"
export WARMUP_UPDATES="${WARMUP_UPDATES:-100000}"
export WARMUP_HANG_START_P="${WARMUP_HANG_START_P:-0.7}"
export HANG_START_P="${HANG_START_P:-0.7}"
export NEAR_GOAL_P="${NEAR_GOAL_P:-0.25}"
export WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0
export GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0
export CART_BARRIER_COEF="${CART_BARRIER_COEF:-10}"
export ENERGY_W="${ENERGY_W:-0.5}"
export INIT_MODE="${INIT_MODE:-bottom}"
export FALL_GRACE_STEPS=20 START_GRACE_STEPS=40
export LR="${LR:-3e-4}"
export CHECKPOINT="${CHECKPOINT:-policies/checkpoint-triple-a.pt}"
export OUT="${OUT:-policies/policy-triple-a.json}"
export RUN_NAME="${RUN_NAME:-ft-triple-a-e${NUM_ENVS:-8192}-r${ROLLOUT:-256}-uuu-swing-f${FORCE_LIMIT}-prog-flip}"

exec bash scripts/next-train-triple.sh "$@"
