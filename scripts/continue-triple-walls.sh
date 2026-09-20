#!/usr/bin/env bash
# Loop forever: walls-on UUU hold specialist (P1a). Cold-start via marker.
# Prefer slot A or C natural restart — do not mid-kill live jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_MATCH="${RUN_MATCH:-ft-triple-walls}"
MARKER="${MARKER:-policies/.triple-walls-v1}"
CHECKPOINT="${CHECKPOINT:-policies/checkpoint-triple-walls.pt}"
OUT="${OUT:-policies/policy-triple-walls.json}"
LOG="${LOG:-logs/continue-triple-walls.log}"
TRAIN_LOG="${TRAIN_LOG:-logs/train-triple-walls.log}"
find_pid() { ps -eo pid=,args= | awk -v m="$RUN_MATCH" '/^[ ]*[0-9]+[ ]+python3 train\/train_triple\.py / && index($0,m){print $1; exit}'; }
mkdir -p logs policies
while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for walls python pid=$PID" >> "$LOG"
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) walls job exited; restarting" >> "$LOG"
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) walls already running; sleep" >> "$LOG"
    sleep 30
    continue
  fi
  if [[ "${TRIPLE_WALLS_COLD:-1}" == "1" && ! -f "$MARKER" ]]; then
    rm -f "$CHECKPOINT"
    touch "$MARKER"
    echo "$(date -u +%FT%TZ) cold-start walls UUU (P1a); marker $MARKER" >> "$LOG"
  fi
  nohup env TRACK_WALLS=1 NUM_ENVS="${NUM_ENVS:-8192}" FORCE_LIMIT="${FORCE_LIMIT:-40}" \
    REWARD_MODE=product PROGRESS_W=1.0 FLIP_AUGMENT=1 \
    WARMUP_UPDATES=100000 WARMUP_GOAL=UUU \
    WARMUP_HANG_START_P=0.0 HANG_START_P=0.0 \
    NEAR_GOAL_P=0.55 WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0 \
    GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0 \
    CART_BARRIER_COEF=10 W_UP=5.0 W_DOWN=1.0 ALPHA_TH=0.5 \
    FALL_GRACE_STEPS=20 START_GRACE_STEPS=40 \
    INIT_MODE=near_target ENERGY_W=0.2 LR=1e-4 ENT=0.05 \
    RUN_NAME="${RUN_MATCH}-e${NUM_ENVS:-8192}-r256-uuu-hold-f40-bar10-prog1-flip" \
    CHECKPOINT="$CHECKPOINT" OUT="$OUT" \
    bash scripts/next-train-triple-walls.sh \
    >> "$TRAIN_LOG" 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED walls pid=$!" >> "$LOG"
  sleep 30
done
