#!/usr/bin/env bash
# UUU-first hold stage (fawraw M2): product reward, near_target inits, no hang.
# After warmup_updates of pure UUU, soft uu-bias continues; flip to multi-eq via
# next-train-triple.sh once UUU hold looks solid (~align≳0.80).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export REWARD_MODE="${REWARD_MODE:-product}"
export WARMUP_UPDATES="${WARMUP_UPDATES:-120}"
export UU_BIAS="${UU_BIAS:-0.70}"
export HANG_START_P="${HANG_START_P:-0.0}"
export WARMUP_HANG_START_P="${WARMUP_HANG_START_P:-0.0}"
export NEAR_GOAL_P="${NEAR_GOAL_P:-0.55}"
export WRONG_EQ_P="${WRONG_EQ_P:-0.05}"
export INIT_MODE="${INIT_MODE:-near_target}"
export CART_BARRIER_COEF="${CART_BARRIER_COEF:-50}"
export W_UP="${W_UP:-5.0}"
export W_DOWN="${W_DOWN:-1.0}"
export FALL_GRACE_STEPS="${FALL_GRACE_STEPS:-20}"
export RUN_NAME="${RUN_NAME:-ft-triple-uuu-e${NUM_ENVS:-8192}-r${ROLLOUT:-256}-prod}"

exec bash scripts/next-train-triple.sh "$@"
