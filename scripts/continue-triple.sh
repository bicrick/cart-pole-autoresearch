#!/usr/bin/env bash
# Loop forever: generic triple product/UUU-first (see continue-triple-a/b for A/B).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_MATCH='ft-triple-e'
find_pid() { ps -eo pid=,args= | awk -v m="$RUN_MATCH" '/^[ ]*[0-9]+[ ]+python3 train\/train_triple\.py / && index($0,m){print $1; exit}'; }
mkdir -p logs
while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for triple python pid=$PID" >> logs/continue-triple.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) triple exited; restarting" >> logs/continue-triple.log
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) triple already running; sleep" >> logs/continue-triple.log
    sleep 30
    continue
  fi
  nohup env NUM_ENVS=8192 bash scripts/next-train-triple.sh \
    >> logs/train-triple.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple pid=$!" >> logs/continue-triple.log
  sleep 30
done
