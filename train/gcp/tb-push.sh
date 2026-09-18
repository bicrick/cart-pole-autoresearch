#!/bin/bash
DEST=gs://cartpole-demo-413636930404/tb/gpu-train/
cd ~/CartPoleDemo || exit 1
mkdir -p policies
while true; do
  shopt -s nullglob
  files=(runs/*/events.out.tfevents*)
  if [ ${#files[@]} -gt 0 ]; then
    gsutil -q -m cp "${files[@]}" "$DEST" || true
  fi
  sleep 15
done
