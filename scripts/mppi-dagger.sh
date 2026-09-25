#!/usr/bin/env bash
# G3: DAgger-distill the MPPI teacher into an MLP student, then export it.
#   GOAL=UUU ITERS=8 bash scripts/mppi-dagger.sh
#   RESUME=1 continues from policies/mppi/student-<goal>.pt + its dataset.
set -euo pipefail
cd "$(dirname "$0")/../train"
export PYTHONPATH=.
GOAL="${GOAL:-UUU}"
g="$(echo "$GOAL" | tr '[:upper:]' '[:lower:]')"
CONFIG="${CONFIG:-mppi/configs/$g.json}"
[ -f "$CONFIG" ] || CONFIG=mppi/configs/uuu.json
OUT="../policies/mppi/student-$g"
EXTRA=()
[ "${RESUME:-0}" = "1" ] && EXTRA+=(--resume)
python3 -m mppi.dagger --config "$CONFIG" --goal "$GOAL" --out "$OUT" \
  --iters "${ITERS:-8}" --episodes "${EPISODES:-32}" --batch "${BATCH:-16}" \
  --steps "${STEPS:-1500}" --threads "${THREADS:-10}" ${EXTRA[@]+"${EXTRA[@]}"}
python3 -m mppi.export_student --ckpt "$OUT.pt" ${WEB:+--web}
