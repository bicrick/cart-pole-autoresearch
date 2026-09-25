"""Teacher config files: ``{"cost": {...}, "mppi": {...}, "walls": false}``."""

from __future__ import annotations

import json
from pathlib import Path

from mppi.costs import CostConfig, TargetCost
from mppi.dynamics_fast import Plant
from mppi.mppi import MPPI, MPPIConfig

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def read_config(path: str | Path, goal: str | None = None) -> dict:
    raw = json.loads(Path(path).read_text())
    if goal:
        raw.setdefault("cost", {})["goal"] = goal
    return raw


def build_teacher(raw: dict):
    plant = Plant.from_constants(walls=bool(raw.get("walls", False)))
    cost_cfg = CostConfig(**raw.get("cost", {}))
    mppi_cfg = MPPIConfig(**raw.get("mppi", {}))
    ctrl = MPPI(plant, TargetCost(plant, cost_cfg), mppi_cfg)
    meta = {"goal": cost_cfg.goal, "cost": cost_cfg.to_dict(), "mppi": mppi_cfg.to_dict(), "walls": plant.walls}
    return plant, ctrl, meta


def load_teacher(path: str | Path, goal: str | None = None):
    return build_teacher(read_config(path, goal))
