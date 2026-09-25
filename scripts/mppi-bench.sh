#!/usr/bin/env bash
# G0: fast-dynamics parity against physics_triple + rollout throughput (local Mac).
set -euo pipefail
cd "$(dirname "$0")/../train"
export PYTHONPATH=.
python3 -m mppi.test_dynamics_fast
python3 -m mppi.bench --k "${K:-2048}" --h "${H:-180}"
