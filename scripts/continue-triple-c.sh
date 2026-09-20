#!/usr/bin/env bash
# DISARMED for square-one (2026-09-20): H1-var / second slot must NOT auto-start.
# Prefer single GPU slot. Sleep forever so legacy rearm is harmless.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
echo "$(date -u +%FT%TZ) continue-triple-c DISARMED (square-one H1-only); sleeping" >> logs/continue-triple-c.log
while true; do sleep 3600; done
