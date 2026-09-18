#!/usr/bin/env bash
set -euo pipefail
cd ~/CartPoleDemo
# kill prior watcher bash processes by reading /proc comm+cmdline without matching this script
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null || true)
  case "$cmd" in
    "bash ./scripts/continue-nowalls.sh"*|"bash ./scripts/continue-edge.sh"*|"bash ./scripts/continue-hot.sh"*)
      kill "$pid" 2>/dev/null || true
      ;;
  esac
done
sleep 1
nohup bash ./scripts/continue-nowalls.sh </dev/null >/dev/null 2>&1 &
nohup bash ./scripts/continue-edge.sh </dev/null >/dev/null 2>&1 &
nohup bash ./scripts/continue-hot.sh </dev/null >/dev/null 2>&1 &
sleep 2
echo "watchers:"
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null || true)
  case "$cmd" in
    "bash ./scripts/continue-"*) echo "$pid $cmd" ;;
  esac
done
echo "logs:"
tail -n 1 logs/continue-nowalls.log 2>/dev/null || true
tail -n 1 logs/continue-edge.log 2>/dev/null || true
tail -n 1 logs/continue-hot.log 2>/dev/null || true
echo "py trainers:"
ps -eo pid=,args= | awk '/^[ ]*[0-9]+[ ]+python3 train\/train\.py/'
