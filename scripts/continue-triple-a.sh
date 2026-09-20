#!/usr/bin/env bash
# Loop forever: triple-A = P1a walls-on UUU hold (product, near_target, no hang).
# Staged for next NATURAL restart — do not mid-kill the live nowalls swing job.
# Cold wipe via .triple-a-walls-v1 marker (first walls deploy).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_MATCH='ft-triple-a'
find_pid() { ps -eo pid=,args= | awk -v m="$RUN_MATCH" '/^[ ]*[0-9]+[ ]+python3 train\/train_triple\.py / && index($0,m){print $1; exit}'; }
mkdir -p logs policies
while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for triple-a python pid=$PID" >> logs/continue-triple-a.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) triple-a exited; restarting as walls-on UUU (P1a)" >> logs/continue-triple-a.log
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) triple-a already running; sleep" >> logs/continue-triple-a.log
    sleep 30
    continue
  fi
  if [[ "${TRIPLE_A_COLD:-1}" == "1" && ! -f policies/.triple-a-walls-v1 ]]; then
    rm -f policies/checkpoint-triple-a.pt
    touch policies/.triple-a-walls-v1
    echo "$(date -u +%FT%TZ) cold-start triple-a walls-on UUU P1a; marker .triple-a-walls-v1 set" >> logs/continue-triple-a.log
  fi
  nohup env TRACK_WALLS=1 NUM_ENVS="${NUM_ENVS:-8192}" FORCE_LIMIT=40 \
    REWARD_MODE=product PROGRESS_W=1.0 FLIP_AUGMENT=1 \
    WARMUP_UPDATES=100000 WARMUP_GOAL=UUU \
    WARMUP_HANG_START_P=0.0 HANG_START_P=0.0 \
    NEAR_GOAL_P=0.55 WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0 \
    GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0 \
    CART_BARRIER_COEF=10 W_UP=5.0 W_DOWN=1.0 ALPHA_TH=0.5 \
    FALL_GRACE_STEPS=20 START_GRACE_STEPS=40 \
    INIT_MODE=near_target ENERGY_W=0.2 LR=1e-4 ENT=0.05 \
    RUN_NAME=ft-triple-a-e8192-r256-uuu-walls-hold-f40-bar10-prog1-flip-ent05-lr1e4 \
    CHECKPOINT=policies/checkpoint-triple-a.pt \
    OUT=policies/policy-triple-a.json \
    bash scripts/next-train-triple-walls.sh \
    >> logs/train-triple-a.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-a walls pid=$!" >> logs/continue-triple-a.log
  sleep 30
done
