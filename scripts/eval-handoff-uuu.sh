#!/usr/bin/env bash
# X1 smoke: S1 (A) → H1 (B) handoff on walls plant when ckpts exist.
# Usage: bash scripts/eval-handoff-uuu.sh
# Optional: SWING=... CATCHER=... EPISODES=32 DEVICE=cpu
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SWING="${SWING:-policies/checkpoint-triple-a.pt}"
CATCHER="${CATCHER:-policies/checkpoint-triple-b.pt}"
EPISODES="${EPISODES:-32}"
DEVICE="${DEVICE:-cpu}"
OUT="${OUT:-logs/handoff-eval-uuu.json}"

mkdir -p logs

BACKEND=ppo
if [[ ! -f "$CATCHER" ]]; then
  echo "catcher ckpt missing ($CATCHER) — falling back to LQR"
  BACKEND=lqr
fi
SWING_ARG=()
if [[ -f "$SWING" ]]; then
  SWING_ARG=(--swing "$SWING")
else
  echo "swing ckpt missing ($SWING) — zero swing placeholder"
fi

CATCHER_ARG=()
if [[ "$BACKEND" == "ppo" ]]; then
  CATCHER_ARG=(--catcher "$CATCHER" --catcher-backend ppo)
else
  CATCHER_ARG=(--catcher-backend lqr)
fi

exec python3 scripts/handoff_eval.py \
  "${SWING_ARG[@]}" \
  "${CATCHER_ARG[@]}" \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --capture-tol 0.1 \
  --track-walls \
  --out "$OUT"
