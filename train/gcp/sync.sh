#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-cartpole-demo}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-cartpole-train}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" --command="mkdir -p ~/CartPoleDemo"
gcloud compute scp --recurse --project="$PROJECT" --zone="$ZONE" \
  "$ROOT/train" "$ROOT/shared" "$ROOT/scripts" "$ROOT/requirements.txt" \
  "$VM:~/CartPoleDemo/"
# docs when present (paper lessons for the VM copy)
if [[ -d "$ROOT/docs" ]]; then
  gcloud compute scp --recurse --project="$PROJECT" --zone="$ZONE" \
    "$ROOT/docs" "$VM:~/CartPoleDemo/"
fi
