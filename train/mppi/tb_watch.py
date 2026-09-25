#!/usr/bin/env python3
"""Mirror DAgger ``*-log.jsonl`` files into TensorBoard, live.

Polls the logs and writes each new iteration row as scalars under
``runs/distill/<log name>``, grouped as swing/, hold/, train/.

    python -m mppi.tb_watch                       # all policies/mppi/*-log.jsonl
    tensorboard --logdir ../runs/distill
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
TRAIN_KEYS = ("loss", "beta", "samples", "collect_s", "train_s", "eval_s")


def group(key: str) -> str:
    if key in TRAIN_KEYS:
        return f"train/{key}"
    if key.startswith("swing_") or key == "g2_pass":
        return f"swing/{key}"
    return f"hold/{key}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "policies" / "mppi" / "*-log.jsonl"))
    ap.add_argument("--logdir", default=str(ROOT / "runs" / "distill"))
    ap.add_argument("--every", type=float, default=10.0, help="poll interval, seconds")
    args = ap.parse_args()
    pattern = Path(args.glob)
    writers: dict[Path, SummaryWriter] = {}
    seen: dict[Path, int] = {}
    print(f"watching {args.glob} -> {args.logdir}", flush=True)
    while True:
        for path in sorted(pattern.parent.glob(pattern.name)):
            lines = path.read_text().splitlines()
            start = seen.get(path, 0)
            if len(lines) <= start:
                continue
            name = path.name.removesuffix("-log.jsonl")
            w = writers.setdefault(path, SummaryWriter(str(Path(args.logdir) / name)))
            for line in lines[start:]:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                it = int(row.get("iter", start))
                for k, v in row.items():
                    if k != "iter" and isinstance(v, (int, float)):
                        w.add_scalar(group(k), float(v), it)
                start += 1
            w.flush()
            seen[path] = start
            print(f"{name}: {start} iterations", flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
