#!/usr/bin/env bash
# Behavior cloning from LQR UUU demos — CPU only. Quiet-basin + fall-kill contract.
# Trains MLP [128,128], eval survival @ init_noise 0.01 and 0.02 (target ≥0.90).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/train${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv-tqc/bin/python" ]]; then
    PYTHON="$ROOT/.venv-tqc/bin/python"
  else
    PYTHON="python3"
  fi
fi

DEMOS="${DEMOS:-policies/lqr-uuu-demos.npz}"
OUT="${OUT:-policies/bc-lqr-uuu.pt}"
N_EPISODES="${N_EPISODES:-50}"
EPOCHS="${EPOCHS:-80}"
PASS_THRESHOLD="${PASS_THRESHOLD:-0.90}"

mkdir -p policies logs

echo "=== BC LQR UUU (CPU) ===" >&2
echo "  demos=$DEMOS out=$OUT epochs=$EPOCHS" >&2
echo "  N=$N_EPISODES noises=0.01,0.02 threshold=$PASS_THRESHOLD" >&2

exec "$PYTHON" train/bc_lqr_uuu.py \
  --demos "$DEMOS" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --n-episodes "$N_EPISODES" \
  --pass-threshold "$PASS_THRESHOLD" \
  --arch 128,128 \
  --loss huber \
  --track-walls \
  "$@"
