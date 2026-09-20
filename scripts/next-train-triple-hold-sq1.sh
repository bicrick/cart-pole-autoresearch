#!/usr/bin/env bash
# Square-one H1 hold ONLY — LOCKED recipe (user 2026-09-20).
# Walls-on UUU near-target hold; PPO ENT=0; NO gSDE/RPO/ERA/ATRPO/AVC/ENERGY crank.
# Do not invent levers. Hard kill later owned by overnight @u100–150 if nt_UUU≲0.15 ∧ reward↑.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TRACK_WALLS=1
export FORCE_LIMIT="${FORCE_LIMIT:-40}"
export REWARD_MODE=product
export PROGRESS_W="${PROGRESS_W:-1.0}"
export FLIP_AUGMENT="${FLIP_AUGMENT:-1}"
export WARMUP_UPDATES=0
export WARMUP_HANG_START_P=0.0
export HANG_START_P=0.0
export NEAR_GOAL_P=1.0
export WRONG_EQ_P=0.0
export UU_BIAS=1.0
export ANNEAL_UPDATES=0
export GOAL_SWITCH_P=0.0
export FOLD_PAIR_P=0.0
export CART_BARRIER_COEF="${CART_BARRIER_COEF:-10}"
# Baseline energy term only (no crank — locked). Default train script is 0.15.
export ENERGY_W="${ENERGY_W:-0.15}"
export INIT_MODE=near_target
export INIT_NOISE="${INIT_NOISE:-0.05}"
export FALL_GRACE_STEPS=20
export START_GRACE_STEPS=40
export LR="${LR:-1e-4}"
export ENT=0
export VEL_COST_COEF="${VEL_COST_COEF:-0.015}"
# Explicitly disarm paper levers
export RPO_ALPHA=0
export LOG_STD_FLOOR=0
export USE_SDE=0
export AVG_REWARD=0
export AVC_NU=0
export AVC_EMA_ALPHA=0
export RESET_LOG_STD=""
export NUM_ENVS="${NUM_ENVS:-8192}"
export ROLLOUT="${ROLLOUT:-256}"
export EPISODE_LEN="${EPISODE_LEN:-1200}"
export CHECKPOINT="${CHECKPOINT:-policies/checkpoint-sq1-h1.pt}"
export OUT="${OUT:-policies/policy-sq1-h1.json}"
export RUN_NAME="${RUN_NAME:-sq1-h1-walls-ent0-nt1-in005}"

exec bash scripts/next-train-triple.sh "$@"
