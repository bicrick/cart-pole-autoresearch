#!/usr/bin/env bash
# Loop forever: triple-B = UUU hold/stabilize specialist (near_target high, low hang).
# v7 ent-boost: ENT=0.05 LR=5e-5 (v6 ENT0.03 collapsed mid-stretch ~u90 ent~0.95).
# Cold wipe via .triple-b-entboost-v7 marker (do not mid-run kill v6).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_MATCH='ft-triple-b'
find_pid() { ps -eo pid=,args= | awk -v m="$RUN_MATCH" '/^[ ]*[0-9]+[ ]+python3 train\/train_triple\.py / && index($0,m){print $1; exit}'; }
mkdir -p logs
while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for triple-b python pid=$PID" >> logs/continue-triple-b.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) triple-b exited; restarting" >> logs/continue-triple-b.log
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) triple-b already running; sleep" >> logs/continue-triple-b.log
    sleep 30
    continue
  fi
  if [[ "${TRIPLE_B_COLD:-1}" == "1" && ! -f policies/.triple-b-entboost-v7 ]]; then
    rm -f policies/checkpoint-triple-b.pt
    touch policies/.triple-b-entboost-v7
    echo "$(date -u +%FT%TZ) cold-start triple-b (entboost v7 / f40 / ent0.05 / lr5e-5); marker set" >> logs/continue-triple-b.log
  fi
  nohup env NUM_ENVS="${NUM_ENVS:-8192}" FORCE_LIMIT=40 \
    REWARD_MODE=product PROGRESS_W=1.0 FLIP_AUGMENT=1 \
    WARMUP_UPDATES=100000 WARMUP_GOAL=UUU \
    WARMUP_HANG_START_P=0.05 HANG_START_P=0.05 \
    NEAR_GOAL_P=0.85 WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0 \
    GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0 \
    CART_BARRIER_COEF=10 W_UP=5.0 W_DOWN=1.0 ALPHA_TH=0.5 \
    FALL_GRACE_STEPS=20 START_GRACE_STEPS=40 \
    INIT_MODE=near_target ENERGY_W=0.15 LR=5e-5 ENT=0.05 \
    RUN_NAME=ft-triple-b-e8192-r256-uuu-hold-f40-bar10-prog1-flip-ent05-lr5e5 \
    CHECKPOINT=policies/checkpoint-triple-b.pt \
    OUT=policies/policy-triple-b.json \
    bash scripts/next-train-triple.sh \
    >> logs/train-triple-b.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-b pid=$!" >> logs/continue-triple-b.log
  sleep 30
done
