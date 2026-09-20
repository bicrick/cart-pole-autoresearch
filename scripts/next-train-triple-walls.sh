#!/usr/bin/env bash
# P1a — walls-on UUU specialist (product reward, near_target, UUU-first).
# Hard inelastic cart endstops help early upright learning; P1b removes walls later.
# Usage: bash scripts/next-train-triple-walls.sh
# Or: TRACK_WALLS=1 bash scripts/next-train-triple-uuu.sh  (same knobs)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TRACK_WALLS="${TRACK_WALLS:-1}"
export REWARD_MODE="${REWARD_MODE:-product}"
export WARMUP_UPDATES="${WARMUP_UPDATES:-100000}"
export UU_BIAS="${UU_BIAS:-1.0}"
export HANG_START_P="${HANG_START_P:-0.0}"
export WARMUP_HANG_START_P="${WARMUP_HANG_START_P:-0.0}"
export NEAR_GOAL_P="${NEAR_GOAL_P:-0.55}"
export WRONG_EQ_P="${WRONG_EQ_P:-0.0}"
export INIT_MODE="${INIT_MODE:-near_target}"
export INIT_NOISE="${INIT_NOISE:-0.3}"
export CART_BARRIER_COEF="${CART_BARRIER_COEF:-10}"
export W_UP="${W_UP:-5.0}"
export W_DOWN="${W_DOWN:-1.0}"
export FALL_GRACE_STEPS="${FALL_GRACE_STEPS:-20}"
export START_GRACE_STEPS="${START_GRACE_STEPS:-40}"
export FORCE_LIMIT="${FORCE_LIMIT:-40}"
export PROGRESS_W="${PROGRESS_W:-1.0}"
export FLIP_AUGMENT="${FLIP_AUGMENT:-1}"
export ENERGY_W="${ENERGY_W:-0.2}"
export LR="${LR:-1e-4}"
export ENT="${ENT:-0.05}"
export ANNEAL_UPDATES="${ANNEAL_UPDATES:-0}"
export GOAL_SWITCH_P="${GOAL_SWITCH_P:-0.0}"
export FOLD_PAIR_P="${FOLD_PAIR_P:-0.0}"
export NUM_ENVS="${NUM_ENVS:-8192}"
export ROLLOUT="${ROLLOUT:-256}"
export RUN_NAME="${RUN_NAME:-ft-triple-walls-e${NUM_ENVS}-r${ROLLOUT}-uuu-hold-f${FORCE_LIMIT}-bar10-prog1-flip}"
export CHECKPOINT="${CHECKPOINT:-policies/checkpoint-triple-walls.pt}"
export OUT="${OUT:-policies/policy-triple-walls.json}"

exec bash scripts/next-train-triple.sh "$@"
