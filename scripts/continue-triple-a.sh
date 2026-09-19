#!/usr/bin/env bash
# Loop forever: triple-A = product + UUU-first + barrier (cold-start OK — reward changed).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_MATCH='ft-triple-a'
find_pid() { ps -eo pid=,args= | awk -v m="$RUN_MATCH" '/^[ ]*[0-9]+[ ]+python3 train\/train_triple\.py / && index($0,m){print $1; exit}'; }
mkdir -p logs
while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for triple-a python pid=$PID" >> logs/continue-triple-a.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) triple-a exited; restarting" >> logs/continue-triple-a.log
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) triple-a already running; sleep" >> logs/continue-triple-a.log
    sleep 30
    continue
  fi
  # Cold start: product reward is incompatible with old additive ckpt scale.
  # Remove stale ckpt once so first launch is cold; subsequent loops resume.
  if [[ "${TRIPLE_A_COLD:-1}" == "1" && -f policies/checkpoint-triple-a.pt ]]; then
    # Only cold-start if ckpt predates product redesign (no 'reward_mode' marker file).
    if [[ ! -f policies/.triple-a-product-v1 ]]; then
      rm -f policies/checkpoint-triple-a.pt
      touch policies/.triple-a-product-v1
      echo "$(date -u +%FT%TZ) cold-start triple-a (new product reward)" >> logs/continue-triple-a.log
    fi
  fi
  nohup env NUM_ENVS=8192 \
    REWARD_MODE=product HANG_START_P=0.10 WARMUP_HANG_START_P=0.0 \
    WARMUP_UPDATES=80 UU_BIAS=0.40 \
    CART_BARRIER_COEF=50 W_UP=5.0 W_DOWN=1.0 ALPHA_TH=0.5 \
    FALL_GRACE_STEPS=20 INIT_MODE=mixed LR=3e-4 \
    RUN_NAME=ft-triple-a-e8192-r256-prod-uuu-bar50 \
    CHECKPOINT=policies/checkpoint-triple-a.pt \
    OUT=policies/policy-triple-a.json \
    bash scripts/next-train-triple.sh \
    >> logs/train-triple-a.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-a pid=$!" >> logs/continue-triple-a.log
  sleep 30
done
