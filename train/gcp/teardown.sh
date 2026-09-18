#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-cartpole-demo}"
ZONE="${ZONE:-us-east1-c}"
VM="${VM:-cartpole-train}"

gcloud compute instances delete "$VM" --project="$PROJECT" --zone="$ZONE" --quiet
echo "deleted $VM"
