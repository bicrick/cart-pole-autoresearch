#!/usr/bin/env bash
# Slot-B watcher → square-one H1 hold ONLY (LOCKED 2026-09-20).
# Old ATRPO/gSDE/TQC ladder DISARMED. Delegates to continue-sq1-h1.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec bash scripts/continue-sq1-h1.sh
