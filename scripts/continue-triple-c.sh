#!/usr/bin/env bash
# Loop forever: triple-C = UUU swing-up specialist (warmup forever) + forceLimit=40.
# near_target + hang≈0.3 (must pump), soft barrier=10, product + energy height bonus.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_MATCH='ft-triple-c'
find_pid() { ps -eo pid=,args= | awk -v m="$RUN_MATCH" '/^[ ]*[0-9]+[ ]+python3 train\/train_triple\.py / && index($0,m){print $1; exit}'; }
mkdir -p logs
while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for triple-c python pid=$PID" >> logs/continue-triple-c.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) triple-c exited; restarting" >> logs/continue-triple-c.log
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) triple-c already running; sleep" >> logs/continue-triple-c.log
    sleep 30
    continue
  fi
  # New swing-up recipe cold-start once (v2 marker); leave A/B markers alone.
  if [[ "${TRIPLE_C_COLD:-1}" == "1" && ! -f policies/.triple-c-force40-swing-v2 ]]; then
    rm -f policies/checkpoint-triple-c.pt
    touch policies/.triple-c-force40-swing-v2
    echo "$(date -u +%FT%TZ) cold-start triple-c (UUU swing-up / forceLimit=40 / bar10 hang03); marker set" >> logs/continue-triple-c.log
  fi
  # WARMUP_UPDATES huge => hard UUU-only for entire run.
  # init near_target + hang 0.3 => hold basin + energy-pump from hang.
  nohup env NUM_ENVS="${NUM_ENVS:-8192}" FORCE_LIMIT=40 \
    REWARD_MODE=product \
    WARMUP_UPDATES=100000 WARMUP_GOAL=UUU \
    WARMUP_HANG_START_P=0.3 HANG_START_P=0.3 \
    NEAR_GOAL_P=0.4 WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0 \
    GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0 \
    CART_BARRIER_COEF=10 W_UP=5.0 W_DOWN=1.0 ALPHA_TH=0.5 \
    FALL_GRACE_STEPS=20 START_GRACE_STEPS=40 \
    INIT_MODE=near_target ENERGY_W=0.35 LR=3e-4 \
    RUN_NAME=ft-triple-c-e8192-r256-uuu-swing-f40-bar10-hang03-nt-e035 \
    CHECKPOINT=policies/checkpoint-triple-c.pt \
    OUT=policies/policy-triple-c.json \
    bash scripts/next-train-triple.sh \
    >> logs/train-triple-c.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-c pid=$!" >> logs/continue-triple-c.log
  sleep 30
done
