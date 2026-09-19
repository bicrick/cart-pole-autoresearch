#!/usr/bin/env bash
# Loop forever: wait for triple-A train to exit, then restart (normal multi-eq curriculum).
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
  nohup env NUM_ENVS=8192 \
    RUN_NAME=ft-triple-a-e8192-r256-hang-uub055-her01 \
    CHECKPOINT=policies/checkpoint-triple-a.pt \
    OUT=policies/policy-triple-a.json \
    bash scripts/next-train-triple.sh \
    >> logs/train-triple-a.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-a pid=$!" >> logs/continue-triple-a.log
  sleep 30
done
