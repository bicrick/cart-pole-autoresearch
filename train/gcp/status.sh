#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-cartpole-demo}"
ZONE="${ZONE:-us-east1-c}"
VM="${VM:-cartpole-train}"

gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" --command="
  tail -n 30 ~/CartPoleDemo/policies/train.log 2>/dev/null || echo 'no log yet'
"
