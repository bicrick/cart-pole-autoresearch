#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-cartpole-demo}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-cartpole-train}"

gcloud compute instances delete "$VM" --project="$PROJECT" --zone="$ZONE" --quiet
echo "deleted $VM"
