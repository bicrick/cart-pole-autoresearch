#!/usr/bin/env bash
# G1 (hold / shove vs LQR) and G2 (swing-up from hang) for one MPPI teacher.
#   GOAL=UUU N=50 bash scripts/mppi-teacher-gates.sh
set -euo pipefail
cd "$(dirname "$0")/../train"
export PYTHONPATH=.
GOAL="${GOAL:-UUU}"
GATE="${GATE:-all}"
N="${N:-50}"
CONFIG="${CONFIG:-mppi/configs/$(echo "$GOAL" | tr '[:upper:]' '[:lower:]').json}"
[ -f "$CONFIG" ] || CONFIG=mppi/configs/uuu.json
python3 -m mppi.teacher_eval --config "$CONFIG" --goal "$GOAL" --gate "$GATE" --n "$N" \
  --batch "${BATCH:-10}" --threads "${THREADS:-8}" \
  --out "../policies/mppi/teacher-$(echo "$GOAL" | tr '[:upper:]' '[:lower:]')-${GATE}.json"
