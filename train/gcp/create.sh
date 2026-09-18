#!/usr/bin/env bash
set -euo pipefail

ACCOUNT="patrickbrown5530@gmail.com"
PROJECT="${PROJECT:-cartpole-demo}"
BILLING="${BILLING:-01EFF4-172B86-5368F2}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-cartpole-train}"
MACHINE="${MACHINE:-n1-standard-8}"
GPU="${GPU:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"

gcloud config set account "$ACCOUNT"
if ! gcloud projects describe "$PROJECT" >/dev/null 2>&1; then
  gcloud projects create "$PROJECT" --name="cartpole-demo"
  gcloud billing projects link "$PROJECT" --billing-account="$BILLING"
fi
gcloud config set project "$PROJECT"
gcloud services enable compute.googleapis.com --project="$PROJECT"

if gcloud compute instances describe "$VM" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1; then
  echo "instance $VM already exists"
  exit 0
fi

gcloud compute instances create "$VM" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --accelerator="type=${GPU},count=${GPU_COUNT}" \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=50GB \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform

echo "created $VM in $PROJECT ($ZONE)"
