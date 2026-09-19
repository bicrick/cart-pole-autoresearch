#!/usr/bin/env bash
# Loop forever: wait for xonly train to exit, then restart from its ckpt.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_MATCH='ft-e16384-r256-xonly'
find_pid() { ps -eo pid=,args= | awk -v m="$RUN_MATCH" '/^[ ]*[0-9]+[ ]+python3 train\/train\.py / && index($0,m){print $1; exit}'; }
while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for xonly python pid=$PID" >> logs/continue-xonly.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) xonly exited; restarting" >> logs/continue-xonly.log
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) xonly already running; sleep" >> logs/continue-xonly.log
    sleep 30
    continue
  fi
  nohup python3 train/train.py \
    --num-envs 16384 --updates 400 --rollout 256 --episode-len 1200 \
    --track-limit 2.4 --reward-clip 8.0 --oob-penalty 20 \
    --align-w 1.5 --energy-w 0.35 --spin-w 0.0003 \
    --center-w 0.06 --center-hold-w 0.30 --warmup-updates 0 --uu-bias 0 \
    --near-goal-p 0 --hang-start-p 0 --wrong-eq-p 0 --fold-pair-p 0 \
    --transition-only --goal-switch-p 0.01 \
    --her-ratio 0.1 --impulse-p 0.01 \
    --logdir runs --run-name ft-e16384-r256-xonly-gsw01 \
    --checkpoint policies/checkpoint-xonly.pt --out policies/policy-xonly.json \
    >> logs/train-xonly.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED xonly pid=$!" >> logs/continue-xonly.log
  sleep 30
done
