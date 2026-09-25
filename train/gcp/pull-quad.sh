#!/usr/bin/env bash
# Copy the quad sweep results (policies/mppi/quad/) back from the VM.
set -euo pipefail

PROJECT="${PROJECT:-cartpole-demo}"
ZONE="${ZONE:-us-east1-b}"
VM="${VM:-cartpole-train-od}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$ROOT/policies/mppi"
gcloud compute scp --recurse --tunnel-through-iap --quiet --project="$PROJECT" --zone="$ZONE" \
  "$VM:~/CartPoleDemo/policies/mppi/quad" "$ROOT/policies/mppi/"
ls -la "$ROOT/policies/mppi/quad"
