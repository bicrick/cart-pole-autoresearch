#!/usr/bin/env bash
# Python-backed triple sim for the web demo: the server steps the training plant
# with the MPPI teacher in the loop; the browser only renders and sends inputs.
#   bash scripts/mppi-server.sh               # 4096 samples (gated teacher)
#   SAMPLES=1024 bash scripts/mppi-server.sh  # faster, less reliable swing-up
#   WARM=1 bash scripts/mppi-server.sh        # compile all 8 teachers up front
# Then: (cd web && npm run dev) and open http://localhost:5173/#triple
set -euo pipefail
cd "$(dirname "$0")/../train"
export PYTHONPATH=.
python3 -m mppi.sim_server --samples "${SAMPLES:-4096}" --port "${PORT:-8765}" --threads "${THREADS:-8}" ${WARM:+--warm}
