#!/usr/bin/env bash
# Loop forever: triple-B slot = Lim TQC UUU specialist (fresh approach).
# Replaces PPO cool-from-boost v8: single-policy PPO near_target/UUU stuck ~0.05
# across cool-ent v6 + entboost v7b (entropy saturated ~3.42). Do not mid-kill
# the live PPO v7b — this script only takes over after that process exits.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs policies

# Prefer the TQC venv when present (box + VM).
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv-tqc/bin/python" ]]; then
    export PYTHON="$ROOT/.venv-tqc/bin/python"
  else
    export PYTHON="python3"
  fi
fi

# Match either legacy PPO B or the TQC trainer so we never double-start.
find_pid() {
  ps -eo pid=,args= | awk '
    /train_triple_tqc\.py/ { print $1; exit }
    /^[ ]*[0-9]+[ ]+python3 train\/train_triple\.py / && index($0,"ft-triple-b") { print $1; exit }
  '
}

while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for slot-B pid=$PID" >> logs/continue-triple-b.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) slot-B exited; starting TQC UUU" >> logs/continue-triple-b.log
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) slot-B already running; sleep" >> logs/continue-triple-b.log
    sleep 30
    continue
  fi

  # One-shot wipe of the old PPO B ckpt so we do not confuse demo sync.
  if [[ ! -f policies/.triple-b-tqc-uuu-v1 ]]; then
    rm -f policies/checkpoint-triple-b.pt
    touch policies/.triple-b-tqc-uuu-v1
    echo "$(date -u +%FT%TZ) cold-start triple-b → TQC UUU v1 (Lim EP7); marker set" >> logs/continue-triple-b.log
  fi

  nohup env FORCE_LIMIT="${FORCE_LIMIT:-40}" TOTAL_STEPS="${TOTAL_STEPS:-300000}" \
    INIT_MODE=near_target INIT_NOISE=0.15 HANG_FRAC=0.05 WIDE_FRAC=0.25 \
    PROGRESS_W=1.0 CART_BARRIER_COEF=10 ALPHA_TH=0.5 \
    RUN_NAME=tqc-uuu-f40-hold-wide \
    CHECKPOINT=policies/tqc-triple-uuu.zip \
    DEVICE="${DEVICE:-auto}" \
    bash scripts/next-train-triple-tqc-uuu.sh \
    >> logs/train-triple-b-tqc.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-b TQC pid=$!" >> logs/continue-triple-b.log
  sleep 60
done
