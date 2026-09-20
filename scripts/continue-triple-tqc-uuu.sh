#!/usr/bin/env bash
# Loop TQC UUU specialist. Do NOT enable on L4 until a PPO slot is deliberately freed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHON="${PYTHON:-$ROOT/.venv-tqc/bin/python}"
RUN_MATCH='train_triple_tqc.py'
find_pid() { ps -eo pid=,args= | awk -v m="$RUN_MATCH" 'index($0,m){print $1; exit}'; }
while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for tqc-uuu pid=$PID" >> logs/continue-triple-tqc-uuu.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
  fi
  nohup env FORCE_LIMIT="${FORCE_LIMIT:-40}" TOTAL_STEPS="${TOTAL_STEPS:-300000}" \
    INIT_MODE=near_target HANG_FRAC=0.05 WIDE_FRAC=0.25 \
    PROGRESS_W=1.0 CART_BARRIER_COEF=10 ALPHA_TH=0.5 \
    RUN_NAME=tqc-uuu-f40-hold-wide \
    CHECKPOINT=policies/tqc-triple-uuu.zip \
    bash scripts/next-train-triple-tqc-uuu.sh \
    >> logs/train-triple-tqc-uuu.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED tqc-uuu pid=$!" >> logs/continue-triple-tqc-uuu.log
  sleep 60
done
