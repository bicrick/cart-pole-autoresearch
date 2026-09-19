#!/usr/bin/env bash
# Loop forever: triple-B = product variant (higher UP weight + slightly higher lr).
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
  if [[ "${TRIPLE_B_COLD:-1}" == "1" && -f policies/checkpoint-triple-b.pt ]]; then
    if [[ ! -f policies/.triple-b-product-v1 ]]; then
      rm -f policies/checkpoint-triple-b.pt
      touch policies/.triple-b-product-v1
      echo "$(date -u +%FT%TZ) cold-start triple-b (new product reward)" >> logs/continue-triple-b.log
    fi
  fi
  nohup env NUM_ENVS=8192 \
    REWARD_MODE=product HANG_START_P=0.10 WARMUP_HANG_START_P=0.0 \
    WARMUP_UPDATES=80 UU_BIAS=0.45 \
    CART_BARRIER_COEF=50 W_UP=8.0 W_DOWN=1.0 ALPHA_TH=0.5 \
    FALL_GRACE_STEPS=20 INIT_MODE=mixed LR=5e-4 \
    RUN_NAME=ft-triple-b-e8192-r256-prod-wup8-lr5e4 \
    CHECKPOINT=policies/checkpoint-triple-b.pt \
    OUT=policies/policy-triple-b.json \
    bash scripts/next-train-triple.sh \
    >> logs/train-triple-b.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-b pid=$!" >> logs/continue-triple-b.log
  sleep 30
done
