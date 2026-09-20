#!/usr/bin/env bash
# Loop forever: triple-C = H1-var walls-on hold (second H1; slightly different lr/ent/noise).
# Choice documented in docs/triple-macro-loop.md — prefer second H1 over combo mush.
# Cold wipe via .triple-c-h1var-walls-v1.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_MATCH='ft-triple-c'
find_pid() { ps -eo pid=,args= | awk -v m="$RUN_MATCH" '/^[ ]*[0-9]+[ ]+python3 train\/train_triple\.py / && index($0,m){print $1; exit}'; }
mkdir -p logs policies
while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for triple-c python pid=$PID" >> logs/continue-triple-c.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) triple-c exited; restarting as H1-var walls-hold" >> logs/continue-triple-c.log
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) triple-c already running; sleep" >> logs/continue-triple-c.log
    sleep 30
    continue
  fi
  bash scripts/quarantine-nan-ckpt.sh policies/checkpoint-triple-c.pt logs/continue-triple-c.log
  if [[ "${TRIPLE_C_COLD:-1}" == "1" && ! -f policies/.triple-c-h1var-walls-v1 ]]; then
    rm -f policies/checkpoint-triple-c.pt
    touch policies/.triple-c-h1var-walls-v1
    echo "$(date -u +%FT%TZ) cold-start triple-c H1-var walls-hold; marker .triple-c-h1var-walls-v1 set" >> logs/continue-triple-c.log
  fi
  # H1-var: same near-only recipe as B, but LR=5e-5 ENT=0.05 INIT_NOISE=0.08.
  nohup env TRACK_WALLS=1 NUM_ENVS="${NUM_ENVS:-8192}" FORCE_LIMIT=40 \
    REWARD_MODE=product PROGRESS_W=1.0 FLIP_AUGMENT=1 \
    WARMUP_UPDATES=0 WARMUP_GOAL=UUU \
    WARMUP_HANG_START_P=0.0 HANG_START_P=0.0 \
    NEAR_GOAL_P=1.0 WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0 \
    GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0 \
    CART_BARRIER_COEF=10 W_UP=5.0 W_DOWN=1.0 ALPHA_TH=0.5 \
    FALL_GRACE_STEPS=20 START_GRACE_STEPS=40 \
    INIT_MODE=near_target INIT_NOISE=0.08 ENERGY_W=0.2 LR=5e-5 ENT=0.05 \
    VEL_COST_COEF=0.015 \
    RUN_NAME=ft-triple-c-e8192-r256-uuu-walls-hold-h1var-f40-nt1-in008-bar10-prog1-flip-ent05-lr5e5 \
    CHECKPOINT=policies/checkpoint-triple-c.pt \
    OUT=policies/policy-triple-c.json \
    bash scripts/next-train-triple-walls.sh \
    >> logs/train-triple-c.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-c H1-var walls-hold pid=$!" >> logs/continue-triple-c.log
  sleep 30
done
