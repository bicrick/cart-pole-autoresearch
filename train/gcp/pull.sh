#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-cartpole-demo}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-cartpole-train}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$ROOT/policies" "$ROOT/web/public"
gcloud compute scp --project="$PROJECT" --zone="$ZONE" \
  "$VM:~/CartPoleDemo/policies/policy.json" "$ROOT/policies/policy.json" || true
gcloud compute scp --project="$PROJECT" --zone="$ZONE" \
  "$VM:~/CartPoleDemo/policies/checkpoint.pt" "$ROOT/policies/checkpoint.pt" || true
if [[ -f "$ROOT/policies/policy.json" ]]; then
  cp "$ROOT/policies/policy.json" "$ROOT/web/public/policy.json"
  echo "pulled policy.json"
else
  echo "policy.json not on VM yet"
fi
