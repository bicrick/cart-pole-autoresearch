#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
find_pid() { ps -eo pid=,args= | awk '/^[ ]*[0-9]+[ ]+python3 train\/train\.py .*ft-e16384-r256-nowalls-oob20-edge/{print $1; exit}'; }
PID=$(find_pid || true)
if [[ -n "${PID}" ]]; then
  echo "$(date -u +%FT%TZ) waiting for nowalls python pid=$PID" >> logs/continue-nowalls.log
  while kill -0 "$PID" 2>/dev/null; do sleep 30; done
  echo "$(date -u +%FT%TZ) nowalls exited; restarting" >> logs/continue-nowalls.log
  sleep 5
fi
if [[ -n "$(find_pid || true)" ]]; then
  echo "$(date -u +%FT%TZ) nowalls already running; exit" >> logs/continue-nowalls.log
  exit 0
fi
nohup python3 train/train.py --num-envs 16384 --updates 400 --rollout 256 --episode-len 1200 --track-limit 2.4 --reward-clip 8.0 --oob-penalty 20 --align-w 1.5 --energy-w 0.35 --spin-w 0.0003 --center-w 0.06 --center-hold-w 0.30 --warmup-updates 0 --uu-bias 0.55 --near-goal-p 0.15 --hang-start-p 0.45 --wrong-eq-p 0.25 --goal-switch-p 0.002 --her-ratio 0.1 --impulse-p 0.01 --logdir runs --run-name ft-e16384-r256-nowalls-oob20-edge --checkpoint policies/checkpoint-nowalls.pt --out policies/policy-nowalls.json >> logs/train-nowalls.log 2>&1 &
echo "$(date -u +%FT%TZ) STARTED nowalls pid=$!" >> logs/continue-nowalls.log
