#!/usr/bin/env python3
"""Export a checkpoint.pt actor to policy.json."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from goals import OBS_DIM
from physics import load_constants
from ppo import ActorCritic, export_actor
from train import POLICY_PATH, WEB_POLICY_PATH
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parents[1] / "policies" / "checkpoint.pt")
    parser.add_argument("--out", type=Path, default=POLICY_PATH)
    args = parser.parse_args()
    constants = load_constants()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    obs_dim = payload.get("obs_dim", OBS_DIM) if isinstance(payload, dict) else OBS_DIM
    model = ActorCritic(obs_dim=obs_dim, hidden=constants["hidden"])
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    spec = export_actor(model, constants)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(spec)
    args.out.write_text(text)
    WEB_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEB_POLICY_PATH.write_text(text)
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
