#!/usr/bin/env bash
set -euo pipefail
export PATH="/workspace/google-cloud-sdk/bin:$PATH"
export GOOGLE_APPLICATION_CREDENTIALS=/home/box/.config/gcloud/cartpole-bot-cartpole-demo.json
export CLOUDSDK_CORE_PROJECT=cartpole-demo
export CLOUDSDK_CORE_ACCOUNT=cartpole-bot@cartpole-demo.iam.gserviceaccount.com
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
ZONE=${ZONE:-us-east1-c}
VM=${VM:-cartpole-train}
LOGDIR=/workspace/double-cart-pole/tb/gpu-train
mkdir -p "$LOGDIR"
gsutil -m cp "gs://cartpole-demo-413636930404/tb/gpu-train/*" "$LOGDIR/" 2>/dev/null || true
REMOTE=$(gcloud compute ssh "$VM" --zone="$ZONE" --project=cartpole-demo --tunnel-through-iap --quiet --command='find ~/CartPoleDemo/runs -name "events.out.tfevents*" -type f 2>/dev/null | head -5' 2>/dev/null || true)
while IFS= read -r f; do
  [ -n "$f" ] || continue
  gcloud compute scp --tunnel-through-iap --zone="$ZONE" --project=cartpole-demo --quiet \
    "$VM:$f" "$LOGDIR/" >/dev/null 2>&1 || true
done <<< "$REMOTE"
