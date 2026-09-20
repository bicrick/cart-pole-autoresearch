#!/usr/bin/env bash
# One-shot H1 walls-hold specialist (same knobs as continue-triple-b).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TRACK_WALLS="${TRACK_WALLS:-1}"
export FORCE_LIMIT="${FORCE_LIMIT:-40}"
export REWARD_MODE="${REWARD_MODE:-product}"
export PROGRESS_W="${PROGRESS_W:-1.0}"
export FLIP_AUGMENT="${FLIP_AUGMENT:-1}"
export WARMUP_UPDATES="${WARMUP_UPDATES:-0}"
export WARMUP_HANG_START_P="${WARMUP_HANG_START_P:-0.0}"
export HANG_START_P="${HANG_START_P:-0.0}"
export NEAR_GOAL_P="${NEAR_GOAL_P:-1.0}"
export WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0
export GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0
export CART_BARRIER_COEF="${CART_BARRIER_COEF:-10}"
export ENERGY_W="${ENERGY_W:-0.2}"
export INIT_MODE="${INIT_MODE:-near_target}"
export INIT_NOISE="${INIT_NOISE:-0.05}"
export FALL_GRACE_STEPS=20 START_GRACE_STEPS=40
export LR="${LR:-1e-4}"
export ENT="${ENT:-0.035}"
export VEL_COST_COEF="${VEL_COST_COEF:-0.015}"
export CHECKPOINT="${CHECKPOINT:-policies/checkpoint-triple-b.pt}"
export OUT="${OUT:-policies/policy-triple-b.json}"
export RUN_NAME="${RUN_NAME:-ft-triple-b-e${NUM_ENVS:-8192}-r${ROLLOUT:-256}-uuu-walls-hold-h1-f${FORCE_LIMIT}-nt1-in005}"

exec bash scripts/next-train-triple.sh "$@"
