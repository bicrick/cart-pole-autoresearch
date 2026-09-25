"""The 2^n equilibria of an n-link cart pendulum: "U"/"D" per link, cart-most link first."""

from __future__ import annotations

import itertools
import math


def goal_ids(n: int) -> list[str]:
    return ["".join(g) for g in itertools.product("DU", repeat=n)]


def goal_angles(goal: str) -> tuple[float, ...]:
    return tuple(0.0 if ch == "U" else math.pi for ch in goal.upper())
