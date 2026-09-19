#!/usr/bin/env bash
# Loop forever: triple-A = UUU-only forever, swing-from-bottom (INIT_MODE=bottom, hang=1.0).
# forceLimit=40, soft barrier=10, energy_w=0.5, start_grace=40. Cold wipe once via v3 marker.
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
  # UUU-bottom-v3 cold-start once (stage-gate: prior f40 multi-eq left warmup with zero UUU hold).
  if [[ "${TRIPLE_A_COLD:-1}" == "1" && ! -f policies/.triple-a-uuu-bottom-v3 ]]; then
    rm -f policies/checkpoint-triple-a.pt
    touch policies/.triple-a-uuu-bottom-v3
    echo "$(date -u +%FT%TZ) cold-start triple-a (UUU bottom-swing / forceLimit=40 / bar10 e05 hang1); marker set" >> logs/continue-triple-a.log
  fi
  # WARMUP_UPDATES huge => hard UUU-only for entire run.
  # init bottom + hang 1.0 => always hang/swing-from-bottom; no goal switch/fold.
  nohup env NUM_ENVS="${NUM_ENVS:-8192}" FORCE_LIMIT=40 \
    REWARD_MODE=product \
    WARMUP_UPDATES=100000 WARMUP_GOAL=UUU \
    WARMUP_HANG_START_P=1.0 HANG_START_P=1.0 \
    NEAR_GOAL_P=0.0 WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0 \
    GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0 \
    CART_BARRIER_COEF=10 W_UP=5.0 W_DOWN=1.0 ALPHA_TH=0.5 \
    FALL_GRACE_STEPS=20 START_GRACE_STEPS=40 \
    INIT_MODE=bottom ENERGY_W=0.5 LR=3e-4 \
    RUN_NAME=ft-triple-a-e8192-r256-uuu-bottom-f40-bar10-e05 \
    CHECKPOINT=policies/checkpoint-triple-a.pt \
    OUT=policies/policy-triple-a.json \
    bash scripts/next-train-triple.sh \
    >> logs/train-triple-a.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-a pid=$!" >> logs/continue-triple-a.log
  sleep 30
done
