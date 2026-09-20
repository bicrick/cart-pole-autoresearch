#!/usr/bin/env bash
# Loop forever: triple-A = S1 walls-on swing specialist (hang/bottom heavy).
# Cold wipe via .triple-a-s1-walls-v1 (role flip from walls-hold sibling).
# 2026-09-20: LR 3e-4→1e-4 after NaN blowup @u228 (policy_loss~1e29, full ckpt NaN).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_MATCH='ft-triple-a'
find_pid() { ps -eo pid=,args= | awk -v m="$RUN_MATCH" '/^[ ]*[0-9]+[ ]+python3 train\/train_triple\.py / && index($0,m){print $1; exit}'; }
mkdir -p logs policies
while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for triple-a python pid=$PID" >> logs/continue-triple-a.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) triple-a exited; restarting as S1 walls-swing" >> logs/continue-triple-a.log
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) triple-a already running; sleep" >> logs/continue-triple-a.log
    sleep 30
    continue
  fi
  bash scripts/quarantine-nan-ckpt.sh policies/checkpoint-triple-a.pt logs/continue-triple-a.log
  if [[ "${TRIPLE_A_COLD:-1}" == "1" && ! -f policies/.triple-a-s1-walls-v1 ]]; then
    rm -f policies/checkpoint-triple-a.pt
    touch policies/.triple-a-s1-walls-v1
    echo "$(date -u +%FT%TZ) cold-start triple-a S1 walls-swing; marker .triple-a-s1-walls-v1 set" >> logs/continue-triple-a.log
  fi
  # S1: walls-on, hang/bottom heavy. LR=1e-4 (was 3e-4 — NaN @u228).
  nohup env TRACK_WALLS=1 NUM_ENVS="${NUM_ENVS:-8192}" FORCE_LIMIT=40 \
    REWARD_MODE=product PROGRESS_W=1.0 FLIP_AUGMENT=1 \
    WARMUP_UPDATES=0 WARMUP_GOAL=UUU \
    WARMUP_HANG_START_P=0.85 HANG_START_P=0.85 \
    NEAR_GOAL_P=0.10 WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0 \
    GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0 \
    CART_BARRIER_COEF=10 W_UP=5.0 W_DOWN=1.0 ALPHA_TH=0.5 \
    FALL_GRACE_STEPS=20 START_GRACE_STEPS=40 \
    INIT_MODE=bottom INIT_NOISE=0.3 ENERGY_W=0.5 LR=1e-4 ENT=0.05 \
    RUN_NAME=ft-triple-a-e8192-r256-uuu-walls-swing-s1-f40-hang085-prog1-flip-ent05-lr1e4 \
    CHECKPOINT=policies/checkpoint-triple-a.pt \
    OUT=policies/policy-triple-a.json \
    bash scripts/next-train-triple-walls.sh \
    >> logs/train-triple-a.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-a S1 walls-swing LR=1e-4 pid=$!" >> logs/continue-triple-a.log
  sleep 30
done
