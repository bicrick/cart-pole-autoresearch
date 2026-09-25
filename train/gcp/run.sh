#!/usr/bin/env bash
# Run a command in ~/CartPoleDemo/train on the VM, detached, logging to LOG.
#   LOG=policies/mppi/quad/sweep.log bash train/gcp/run.sh "python3 -m mppi.nlink.sweep --n 4 ..."
#   bash train/gcp/run.sh --tail          # follow the log
set -euo pipefail

PROJECT="${PROJECT:-cartpole-demo}"
ZONE="${ZONE:-us-east1-b}"
VM="${VM:-cartpole-train-od}"
LOG="${LOG:-policies/mppi/quad/run.log}"
SSH=(gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap --quiet)

if [[ "${1:-}" == "--tail" ]]; then
  "${SSH[@]}" --command="tail -n 40 -f ~/CartPoleDemo/$LOG"
  exit 0
fi

CMD="$1"
"${SSH[@]}" --command="
  set -euo pipefail
  cd ~/CartPoleDemo
  mkdir -p \$(dirname $LOG)
  cd train
  nohup bash -c 'PYTHONWARNINGS=ignore $CMD' > ~/CartPoleDemo/$LOG 2>&1 &
  echo started pid=\$! log=$LOG
"
