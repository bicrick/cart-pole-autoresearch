#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-cartpole-demo}"
ZONE="${ZONE:-us-east1-c}"
VM="${VM:-cartpole-train}"
NUM_ENVS="${NUM_ENVS:-8192}"
UPDATES="${UPDATES:-400}"
# Set USE_LEGACY=1 to restore the pre-curriculum command (not recommended).
USE_LEGACY="${USE_LEGACY:-0}"

if [[ "$USE_LEGACY" == "1" ]]; then
  gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap --command="
  set -euo pipefail
  cd ~/CartPoleDemo
  python3 -m pip install -q -r requirements.txt
  mkdir -p policies
  nohup python3 train/train.py --num-envs ${NUM_ENVS} --updates ${UPDATES} \\
    --checkpoint policies/checkpoint.pt --out policies/policy.json \\
    > policies/train.log 2>&1 &
  echo \$! > policies/train.pid
  echo started pid=\$(cat policies/train.pid)
"
else
  gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap --command="
  set -euo pipefail
  cd ~/CartPoleDemo
  python3 -m pip install -q -r requirements.txt
  mkdir -p policies
  nohup env NUM_ENVS=${NUM_ENVS} UPDATES=${UPDATES} bash scripts/next-train.sh \\
    > policies/train.log 2>&1 &
  echo \$! > policies/train.pid
  echo started pid=\$(cat policies/train.pid) recipe=scripts/next-train.sh
"
fi
