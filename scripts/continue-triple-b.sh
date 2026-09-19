#!/usr/bin/env bash
# Loop forever: wait for triple-B train to exit, then restart (hotter hang / higher lr).
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
  nohup env NUM_ENVS=8192 HANG_START_P=0.60 LR=1e-3 \
    RUN_NAME=ft-triple-b-e8192-r256-hang060-lr1e3 \
    CHECKPOINT=policies/checkpoint-triple-b.pt \
    OUT=policies/policy-triple-b.json \
    bash scripts/next-train-triple.sh \
    >> logs/train-triple-b.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-b pid=$!" >> logs/continue-triple-b.log
  sleep 30
done
