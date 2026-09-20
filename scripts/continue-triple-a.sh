#!/usr/bin/env bash
# DISARMED for square-one (2026-09-20): S1 swing must NOT auto-start.
# Sleep forever so any legacy rearm/cron that launches this is harmless.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
echo "$(date -u +%FT%TZ) continue-triple-a DISARMED (square-one H1-only); sleeping" >> logs/continue-triple-a.log
while true; do sleep 3600; done
