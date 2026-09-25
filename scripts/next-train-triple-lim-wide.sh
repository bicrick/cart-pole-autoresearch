#!/usr/bin/env bash
# One Lim KIEE 2025 upright specialist on equation-(10) starts.
# Does not touch the quiet-basin keepers.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=train
CKPT="${CKPT:-policies/tqc-lim-uuu-wide.zip}"
STEPS="${STEPS:-1000000}"
exec python3 train/train_triple_tqc.py \
  --goal UUU \
  --init-mode wide \
  --total-steps "$STEPS" \
  --learning-starts 10000 \
  --buffer-size 1000000 \
  --batch-size 256 \
  --lr 3e-4 \
  --gamma 0.99 \
  --tau 0.005 \
  --n-critics 3 \
  --n-quantiles 25 \
  --top-quantiles-to-drop 2 \
  --policy-arch 400,300 \
  --critic-arch 512,512,512 \
  --gradient-steps 1 \
  --n-envs 1 \
  --progress-w 0 \
  --ry-scale 1 \
  --cart-barrier-coef 0 \
  --alpha-th 0 \
  --w-up 1 \
  --w-down 1 \
  --sparse-bonus 0 \
  --actor-freeze-steps 0 \
  --no-pin-log-std \
  --ent-coef auto \
  --no-track-walls \
  --force-limit 40 \
  --max-steps 1000 \
  --checkpoint "$CKPT" \
  --run-name lim-uuu-wide \
  --save-freq 100000 \
  --eval-freq 50000 \
  "$@"
