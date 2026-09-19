#!/usr/bin/env bash
# Loop forever: triple-B = UUU-only forever + wide ICs + forceLimit=60.
# Soft barrier=10, energy_w=0.35, hang=0.3, near_goal=0.25, lr=3e-4 (was 5e-4; entropy collapsed).
# Cold wipe once via v4 marker after hot-lr collapse.
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
  # UUU-wide-f60-lr3e4-v4 cold-start once (v3 hot lr=5e-4 entropy collapsed ~-0.41 @u320).
  if [[ "${TRIPLE_B_COLD:-1}" == "1" && ! -f policies/.triple-b-uuu-wide-f60-lr3e4-v4 ]]; then
    rm -f policies/checkpoint-triple-b.pt
    touch policies/.triple-b-uuu-wide-f60-lr3e4-v4
    echo "$(date -u +%FT%TZ) cold-start triple-b (UUU wide / forceLimit=60 / bar10 e035 hang03 / lr3e-4); marker set" >> logs/continue-triple-b.log
  fi
  # WARMUP_UPDATES huge => hard UUU-only for entire run.
  # init wide + hang 0.3 + near_goal 0.25; no goal switch/fold.
  nohup env NUM_ENVS="${NUM_ENVS:-8192}" FORCE_LIMIT=60 \
    REWARD_MODE=product \
    WARMUP_UPDATES=100000 WARMUP_GOAL=UUU \
    WARMUP_HANG_START_P=0.3 HANG_START_P=0.3 \
    NEAR_GOAL_P=0.25 WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0 \
    GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0 \
    CART_BARRIER_COEF=10 W_UP=5.0 W_DOWN=1.0 ALPHA_TH=0.5 \
    FALL_GRACE_STEPS=20 START_GRACE_STEPS=40 \
    INIT_MODE=wide ENERGY_W=0.35 LR=3e-4 \
    RUN_NAME=ft-triple-b-e8192-r256-uuu-wide-f60-bar10-e035-lr3e4 \
    CHECKPOINT=policies/checkpoint-triple-b.pt \
    OUT=policies/policy-triple-b.json \
    bash scripts/next-train-triple.sh \
    >> logs/train-triple-b.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-b pid=$!" >> logs/continue-triple-b.log
  sleep 30
done
