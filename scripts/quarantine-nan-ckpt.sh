#!/usr/bin/env bash
# Usage: bash scripts/quarantine-nan-ckpt.sh <ckpt.pt> [logfile]
# Exit 0 always. Moves ckpt aside if NaN/Inf/unreadable.
set -euo pipefail
ckpt="${1:?ckpt path}"
log="${2:-/dev/null}"
[[ -f "$ckpt" ]] || exit 0
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/policies/quarantine"
if python3 - "$ckpt" <<'PY'
import sys, torch
p = sys.argv[1]
try:
    ck = torch.load(p, map_location="cpu")
except Exception as e:
    print(f"unreadable: {e}")
    sys.exit(2)
sd = ck.get("model", ck) if isinstance(ck, dict) else None
if not isinstance(sd, dict):
    sys.exit(0)
bad = 0
for v in sd.values():
    if hasattr(v, "dtype") and (torch.isnan(v).any() or torch.isinf(v).any()):
        bad += 1
sys.exit(1 if bad else 0)
PY
then
  exit 0
fi
rc=$?
ts=$(date -u +%Y%m%dT%H%M%SZ)
dest="$root/policies/quarantine/$(basename "$ckpt" .pt)-nan-$ts.pt"
mv -f "$ckpt" "$dest"
echo "$(date -u +%FT%TZ) QUARANTINED NaN/unreadable ckpt -> $dest (rc=$rc)" >> "$log"
