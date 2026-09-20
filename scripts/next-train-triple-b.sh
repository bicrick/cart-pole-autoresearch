#!/usr/bin/env bash
# One-shot H1 walls-hold → square-one locked recipe (delegates to hold-sq1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec bash scripts/next-train-triple-hold-sq1.sh "$@"
