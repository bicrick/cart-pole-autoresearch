#!/usr/bin/env bash
# Loop forever: square-one H1 walls-on hold (sq1b: INIT_NOISE=0.02).
# Disarmed: A/C swing/combo, TQC, gSDE, RPO, ERA, ATRPO, AVC, ENERGY crank.
# Prefer single GPU slot. Hard kill @u100–150 owned by overnight if nt_UUU≲0.15 ∧ reward↑.
# Do NOT resume FAIL quarantine ckpt — cold start via .sq1b-h1-cold-v1 marker.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs policies

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv-tqc/bin/python" ]]; then
    export PYTHON="$ROOT/.venv-tqc/bin/python"
  else
    export PYTHON="python3"
  fi
fi

find_pid() {
  ps -eo pid=,args= | awk '
    /train_triple_tqc\.py/ { print $1; exit }
    /^[ ]*[0-9]+[ ]+python3 train\/train_triple\.py / && (index($0,"sq1b-h1") || index($0,"sq1-h1") || index($0,"checkpoint-sq1b-h1") || index($0,"checkpoint-sq1-h1")) { print $1; exit }
  '
}

start_sq1b_h1() {
  # Quarantine NaN on dedicated sq1b ckpt; never load FAIL-u100 quarantine.
  if [[ -x scripts/quarantine-nan-ckpt.sh ]]; then
    bash scripts/quarantine-nan-ckpt.sh policies/checkpoint-sq1b-h1.pt logs/continue-sq1-h1.log || true
  fi
  # Cold start once for sq1b in002 (tighter ICs). Wipe any leftover live ckpt.
  if [[ ! -f policies/.sq1b-h1-cold-v1 ]]; then
    rm -f policies/checkpoint-sq1b-h1.pt policies/checkpoint-sq1-h1.pt
    # Explicitly do not copy FAIL quarantine weights.
    touch policies/.sq1b-h1-cold-v1
    echo "$(date -u +%FT%TZ) cold-start sq1b-h1 walls-hold ENT=0 INIT_NOISE=0.02; marker set" >> logs/continue-sq1-h1.log
  fi
  nohup env TRACK_WALLS=1 NUM_ENVS="${NUM_ENVS:-8192}" FORCE_LIMIT=40 \
    REWARD_MODE=product PROGRESS_W=1.0 FLIP_AUGMENT=1 \
    WARMUP_UPDATES=0 WARMUP_HANG_START_P=0.0 HANG_START_P=0.0 \
    NEAR_GOAL_P=1.0 WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0 \
    GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0 \
    CART_BARRIER_COEF=10 ENERGY_W=0.15 \
    INIT_MODE=near_target INIT_NOISE=0.02 \
    FALL_GRACE_STEPS=20 START_GRACE_STEPS=40 \
    LR=1e-4 ENT=0 VEL_COST_COEF=0.015 \
    RPO_ALPHA=0 LOG_STD_FLOOR=0 USE_SDE=0 AVG_REWARD=0 AVC_NU=0 AVC_EMA_ALPHA=0 \
    RUN_NAME=sq1b-h1-walls-ent0-nt1-in002 \
    CHECKPOINT=policies/checkpoint-sq1b-h1.pt \
    OUT=policies/policy-sq1b-h1.json \
    bash scripts/next-train-triple-hold-sq1.sh \
    >> logs/train-sq1-h1.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED sq1b-h1 walls-hold ENT=0 INIT_NOISE=0.02 pid=$! RUN_NAME=sq1b-h1-walls-ent0-nt1-in002" >> logs/continue-sq1-h1.log
}

while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for sq1b-h1 pid=$PID" >> logs/continue-sq1-h1.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) sq1b-h1 exited; restarting locked recipe" >> logs/continue-sq1-h1.log
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) sq1b-h1 already running; sleep" >> logs/continue-sq1-h1.log
    sleep 30
    continue
  fi
  start_sq1b_h1
  sleep 60
done
