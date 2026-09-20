#!/usr/bin/env bash
# LQR UUU oracle gate — CPU only. Quiet-basin + fall-kill env contract.
# PASS (≥95% survival @ init_noise 0.01 and 0.02) → dump demos for BC.
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

N_EPISODES="${N_EPISODES:-50}"
DEMO_EPISODES="${DEMO_EPISODES:-80}"
DEMO_OUT="${DEMO_OUT:-policies/lqr-uuu-demos.npz}"
PASS_THRESHOLD="${PASS_THRESHOLD:-0.95}"

mkdir -p policies logs

echo "=== LQR oracle UUU (CPU) ===" >&2
echo "  N=$N_EPISODES noises=0.01,0.02 threshold=$PASS_THRESHOLD" >&2
echo "  demo_out=$DEMO_OUT demo_eps=$DEMO_EPISODES" >&2

exec "$PYTHON" train/lqr_oracle_uuu.py \
  --n-episodes "$N_EPISODES" \
  --demo-episodes "$DEMO_EPISODES" \
  --demo-out "$DEMO_OUT" \
  --pass-threshold "$PASS_THRESHOLD" \
  --track-walls \
  "$@"
