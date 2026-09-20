#!/usr/bin/env bash
# Loop forever: triple-C = P1a walls-on UUU (product, near_target-heavy, light hang).
# Staged for next NATURAL restart — do not mid-kill the live nowalls combo job.
# Cold wipe via .triple-c-walls-v1 marker (first walls deploy).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_MATCH='ft-triple-c'
find_pid() { ps -eo pid=,args= | awk -v m="$RUN_MATCH" '/^[ ]*[0-9]+[ ]+python3 train\/train_triple\.py / && index($0,m){print $1; exit}'; }
mkdir -p logs policies
while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for triple-c python pid=$PID" >> logs/continue-triple-c.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) triple-c exited; restarting as walls-on UUU (P1a)" >> logs/continue-triple-c.log
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) triple-c already running; sleep" >> logs/continue-triple-c.log
    sleep 30
    continue
  fi
  if [[ "${TRIPLE_C_COLD:-1}" == "1" && ! -f policies/.triple-c-walls-v1 ]]; then
    rm -f policies/checkpoint-triple-c.pt
    touch policies/.triple-c-walls-v1
    echo "$(date -u +%FT%TZ) cold-start triple-c walls-on UUU P1a; marker .triple-c-walls-v1 set" >> logs/continue-triple-c.log
  fi
  nohup env TRACK_WALLS=1 NUM_ENVS="${NUM_ENVS:-8192}" FORCE_LIMIT=40 \
    REWARD_MODE=product PROGRESS_W=1.0 FLIP_AUGMENT=1 \
    WARMUP_UPDATES=100000 WARMUP_GOAL=UUU \
    WARMUP_HANG_START_P=0.15 HANG_START_P=0.15 \
    NEAR_GOAL_P=0.50 WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0 \
    GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0 \
    CART_BARRIER_COEF=10 W_UP=5.0 W_DOWN=1.0 ALPHA_TH=0.5 \
    FALL_GRACE_STEPS=20 START_GRACE_STEPS=40 \
    INIT_MODE=near_target ENERGY_W=0.25 LR=5e-5 ENT=0.08 \
    RUN_NAME=ft-triple-c-e8192-r256-uuu-walls-combo-f40-bar10-prog1-flip-ent08-lr5e5 \
    CHECKPOINT=policies/checkpoint-triple-c.pt \
    OUT=policies/policy-triple-c.json \
    bash scripts/next-train-triple-walls.sh \
    >> logs/train-triple-c.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-c walls pid=$!" >> logs/continue-triple-c.log
  sleep 30
done
