#!/usr/bin/env bash
# Loop forever: triple-B = H1 walls-on hold specialist (near-only, tight init noise).
# Phase 1 (done): Lim TQC UUU wide (marker .triple-b-tqc-uuu-v1) — do not relaunch.
# Phase 2: PPO-balance walls → H1 with INIT_NOISE=0.05 (marker .triple-b-h1-noise05-v1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs policies

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv-tqc/bin/python" ]]; then
    export PYTHON="$ROOT/.venv-tqc/bin/python"
  else
    export PYTHON="python3"
  fi
fi

# Match either TQC trainer or PPO B so we never double-start.
find_pid() {
  ps -eo pid=,args= | awk '
    /train_triple_tqc\.py/ { print $1; exit }
    /^[ ]*[0-9]+[ ]+python3 train\/train_triple\.py / && index($0,"ft-triple-b") { print $1; exit }
  '
}

start_tqc_wide() {
  nohup env FORCE_LIMIT="${FORCE_LIMIT:-40}" TOTAL_STEPS="${TOTAL_STEPS:-300000}" \
    INIT_MODE=near_target INIT_NOISE=0.15 HANG_FRAC=0.05 WIDE_FRAC=0.25 \
    PROGRESS_W=1.0 CART_BARRIER_COEF=10 ALPHA_TH=0.5 \
    RUN_NAME=tqc-uuu-f40-hold-wide \
    CHECKPOINT=policies/tqc-triple-uuu.zip \
    DEVICE="${DEVICE:-auto}" \
    bash scripts/next-train-triple-tqc-uuu.sh \
    >> logs/train-triple-b-tqc.log 2>&1 &
  echo "$(date -u +%FT%TZ) STARTED triple-b TQC wide pid=$!" >> logs/continue-triple-b.log
}

start_ppo_h1() {
  bash scripts/quarantine-nan-ckpt.sh policies/checkpoint-triple-b.pt logs/continue-triple-b.log
  # H1 hold catcher: near_target only, no hang, noise=0.05, UUU-biased, force40, walls-on.
  # ENT=0.035 alone DEAD (σ→floor, policy_loss ~8e26 @u75). Live stretch may still be
  # RPO+ERA+ENT=0.02 (cold @11:19Z) — that ENT term is a cookbook mismatch:
  # CleanRL RPO + Zoo Pendulum + Raffin gSDE all use ent_coef=0 (explore via μ-perturb / gSDE,
  # not β). Next restart: RPO α=0.01 + ERA H₀=0.5 + ENT=0 + RESET_LOG_STD (2212.07536 /
  # 2510.08549 / Zoo). Never AR-EAPO MaxEnt / ENT≥0.02 on RPO. gSDE after ENT=0 fails (next marker).
  if [[ ! -f policies/.triple-b-h1-noise05-v1 ]]; then
    rm -f policies/checkpoint-triple-b.pt
    touch policies/.triple-b-h1-noise05-v1
    echo "$(date -u +%FT%TZ) cold-start triple-b → H1 walls-hold noise0.05; marker set" >> logs/continue-triple-b.log
  fi
  if [[ ! -f policies/.triple-b-h1-rpo01-era05-ent0-v1 ]]; then
    touch policies/.triple-b-h1-rpo01-era05-ent0-v1
    # also keep legacy marker so we never re-arm the ENT=0.02 recipe
    touch policies/.triple-b-h1-rpo01-era05-v1
    echo "$(date -u +%FT%TZ) arming H1 RPO=0.01 + ERA H0=0.5 + ENT=0 (CleanRL/Zoo); marker set" >> logs/continue-triple-b.log
  fi
  # H1 gSDE FAILED (2026-09-20 ~07:20 CT): H 1.59→0.58 @u30, nt_UUU flat ~0.04,
  # reward↑/hold≈0 — sample-path gSDE did not stop σ-collapse (macro u30–40 gate).
  # Keep marker so we never re-arm gSDE / ENT≥0.02 / AR-EAPO.
  if [[ ! -f policies/.triple-b-h1-gsde-ent0-v1 ]]; then
    touch policies/.triple-b-h1-gsde-ent0-v1
    rm -f policies/checkpoint-triple-b.pt
    echo "$(date -u +%FT%TZ) arming H1 gSDE sf4 + ENT=0 (RPO+ERA kept); cold ckpt; marker set" >> logs/continue-triple-b.log
  fi
  # ENERGY_W=0.35 stretch (LIVE ~07:23 CT). Babysit to u60–80; if nt still flat+reward↑
  # → overnight touches .triple-b-h1-atrpo-v1 then mid-kills / natural-exits into ATRPO.
  if [[ ! -f policies/.triple-b-h1-energy035-v1 ]]; then
    touch policies/.triple-b-h1-energy035-v1
    rm -f policies/checkpoint-triple-b.pt
    echo "$(date -u +%FT%TZ) arming H1 ENERGY_W=0.35 + RPO+ERA ENT=0 (no gSDE); cold ckpt; marker set" >> logs/continue-triple-b.log
  fi
  if [[ -f policies/.triple-b-h1-atrpo-v1 ]]; then
    # ATRPO-lite (Zhang–Ross 2106.07329): ρ-center + γ-free GAE, ENT=0, EP≥1200.
    # Cold wipe ENERGY_W weights once.
    if [[ ! -f policies/.triple-b-h1-atrpo-cold-v1 ]]; then
      rm -f policies/checkpoint-triple-b.pt
      touch policies/.triple-b-h1-atrpo-cold-v1
      echo "$(date -u +%FT%TZ) ATRPO cold wipe collapsed ENERGY_W ckpt; cold marker set" >> logs/continue-triple-b.log
    fi
    nohup env TRACK_WALLS=1 NUM_ENVS="${NUM_ENVS:-8192}" FORCE_LIMIT=40 \
      REWARD_MODE=product PROGRESS_W=1.0 FLIP_AUGMENT=1 \
      WARMUP_UPDATES=0 WARMUP_GOAL=UUU \
      WARMUP_HANG_START_P=0.0 HANG_START_P=0.0 \
      NEAR_GOAL_P=1.0 WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0 \
      GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0 \
      CART_BARRIER_COEF=10 W_UP=5.0 W_DOWN=1.0 ALPHA_TH=0.5 \
      FALL_GRACE_STEPS=20 START_GRACE_STEPS=40 \
      INIT_MODE=near_target INIT_NOISE=0.05 ENERGY_W=0.35 LR=1e-4 ENT=0 \
      VEL_COST_COEF=0.015 RPO_ALPHA=0.01 LOG_STD_FLOOR=0.5 RESET_LOG_STD=0 \
      USE_SDE=0 AVG_REWARD=1 EPISODE_LEN=1200 \
      RUN_NAME=ft-triple-b-e8192-r256-uuu-walls-hold-h1-f40-nt1-in005-rpo01-era05-ew035-atrpo-ent0-bar10-prog1-flip \
      CHECKPOINT=policies/checkpoint-triple-b.pt \
      OUT=policies/policy-triple-b.json \
      bash scripts/next-train-triple.sh \
      >> logs/train-triple-b-balance.log 2>&1 &
    echo "$(date -u +%FT%TZ) STARTED triple-b H1 ATRPO-lite + ENERGY_W=0.35 RPO+ERA ENT=0 pid=$!" >> logs/continue-triple-b.log
  else
    nohup env TRACK_WALLS=1 NUM_ENVS="${NUM_ENVS:-8192}" FORCE_LIMIT=40 \
      REWARD_MODE=product PROGRESS_W=1.0 FLIP_AUGMENT=1 \
      WARMUP_UPDATES=0 WARMUP_GOAL=UUU \
      WARMUP_HANG_START_P=0.0 HANG_START_P=0.0 \
      NEAR_GOAL_P=1.0 WRONG_EQ_P=0.0 UU_BIAS=1.0 ANNEAL_UPDATES=0 \
      GOAL_SWITCH_P=0.0 FOLD_PAIR_P=0.0 \
      CART_BARRIER_COEF=10 W_UP=5.0 W_DOWN=1.0 ALPHA_TH=0.5 \
      FALL_GRACE_STEPS=20 START_GRACE_STEPS=40 \
      INIT_MODE=near_target INIT_NOISE=0.05 ENERGY_W=0.35 LR=1e-4 ENT=0 \
      VEL_COST_COEF=0.015 RPO_ALPHA=0.01 LOG_STD_FLOOR=0.5 RESET_LOG_STD=0 \
      USE_SDE=0 \
      RUN_NAME=ft-triple-b-e8192-r256-uuu-walls-hold-h1-f40-nt1-in005-rpo01-era05-ew035-ent0-bar10-prog1-flip \
      CHECKPOINT=policies/checkpoint-triple-b.pt \
      OUT=policies/policy-triple-b.json \
      bash scripts/next-train-triple.sh \
      >> logs/train-triple-b-balance.log 2>&1 &
    echo "$(date -u +%FT%TZ) STARTED triple-b H1 walls-hold RPO+ERA ENERGY_W=0.35 ENT=0 (no gSDE) pid=$!" >> logs/continue-triple-b.log
  fi
}

while true; do
  PID=$(find_pid || true)
  if [[ -n "${PID}" ]]; then
    echo "$(date -u +%FT%TZ) waiting for slot-B pid=$PID" >> logs/continue-triple-b.log
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "$(date -u +%FT%TZ) slot-B exited; picking next recipe" >> logs/continue-triple-b.log
    sleep 5
  fi
  if [[ -n "$(find_pid || true)" ]]; then
    echo "$(date -u +%FT%TZ) slot-B already running; sleep" >> logs/continue-triple-b.log
    sleep 30
    continue
  fi

  # First-ever: cold TQC wide (legacy path; already marked on VM — skipped).
  if [[ ! -f policies/.triple-b-tqc-uuu-v1 ]]; then
    rm -f policies/checkpoint-triple-b.pt
    touch policies/.triple-b-tqc-uuu-v1
    echo "$(date -u +%FT%TZ) cold-start triple-b → TQC UUU v1 (Lim EP7); marker set" >> logs/continue-triple-b.log
    start_tqc_wide
    sleep 60
    continue
  fi

  # After wide TQC has been started once: H1 PPO-balance walls-hold.
  start_ppo_h1
  sleep 60
done
