#!/usr/bin/env python3
"""Python-backed web sim for the triple demo (temporary, until distillation lands).

The server owns the physics: the same ``dynamics_fast`` plant (walls off, void
respawn) the MPPI teacher was trained and gated on, stepped at 120 Hz, with the
full teacher (``MPPI.act``, exactly as in ``teacher_eval``) driving it. The
browser only renders and sends inputs. If the teacher cannot keep up, the sim
runs slower than real time rather than degrading.

    python -m mppi.sim_server --samples 4096 --port 8765

Client -> server (any subset, sent every frame):
    {"started": true, "goal": "UUU", "policy": true, "manual": N,
     "grab": {"body": "tip", "x": m, "y": m} | null}
    {"started": true, "reset": "DDD"}   teleport to an equilibrium (or "hang")
Only a client that has sent ``started: true`` (the user interacted with that
tab) drives the sim; the most recent such client owns it. Others just watch.
Server -> client (~60 Hz):
    {"type": "state", "state": {x, xd, th1, ...}, "force", "goal", "policy",
     "rt": sim-time / wall-time, "ms": last replan ms, "status"}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import threading
import time

import torch
import websockets

from goals_triple import GOAL_ANGLES, GOAL_IDS, parse_goal
from mppi.config import build_teacher, read_config
from mppi.dynamics_fast import Plant, step

KEYS = ("x", "xd", "th1", "th1d", "th2", "th2d", "th3", "th3d")
GRAB_STIFFNESS = 40.0
GRAB_DAMPING = 4.0


def equilibrium(goal: str) -> torch.Tensor:
    s = torch.zeros(1, 8)
    ang = GOAL_ANGLES[parse_goal(goal)]
    s[0, 2], s[0, 4], s[0, 6] = ang[0], ang[1], ang[2]
    return s


def hang() -> torch.Tensor:
    """Respawn pose after leaving the track (matches the web HANGING_TRIPLE)."""
    return torch.tensor([[0.0, 0.0, math.pi + 0.16, 0.28, math.pi - 0.1, -0.18, math.pi + 0.08, 0.12]])


def grab_forces(p: Plant, s: torch.Tensor, grab: dict):
    """Spring-damper pull on a body, as generalized forces (port of web grabForces)."""
    x, xd, t1, w1, t2, w2, t3, w3 = (float(v) for v in s[0])
    c1, s1, c2, s2, c3, s3 = math.cos(t1), math.sin(t1), math.cos(t2), math.sin(t2), math.cos(t3), math.sin(t3)
    body = grab.get("body")
    px, py, vx, vy = x, 0.0, xd, 0.0
    if body in ("lower", "mid", "tip"):
        px += p.l1 * s1
        py += p.l1 * c1
        vx += p.l1 * c1 * w1
        vy -= p.l1 * s1 * w1
    if body in ("mid", "tip"):
        px += p.l2 * s2
        py += p.l2 * c2
        vx += p.l2 * c2 * w2
        vy -= p.l2 * s2 * w2
    if body == "tip":
        px += p.l3 * s3
        py += p.l3 * c3
        vx += p.l3 * c3 * w3
        vy -= p.l3 * s3 * w3
    fx = GRAB_STIFFNESS * (float(grab["x"]) - px) - GRAB_DAMPING * vx
    fy = GRAB_STIFFNESS * (float(grab["y"]) - py) - GRAB_DAMPING * vy
    q1 = p.l1 * c1 * fx - p.l1 * s1 * fy if body in ("lower", "mid", "tip") else 0.0
    q2 = p.l2 * c2 * fx - p.l2 * s2 * fy if body in ("mid", "tip") else 0.0
    q3 = p.l3 * c3 * fx - p.l3 * s3 * fy if body == "tip" else 0.0
    return tuple(torch.tensor([v]) for v in (fx, q1, q2, q3))


class Sim(threading.Thread):
    def __init__(self, config: str, samples: int):
        super().__init__(daemon=True)
        self.config = config
        self.samples = samples
        self.plant = Plant.from_constants(walls=False)
        self.teachers: dict = {}
        self.lock = threading.Lock()
        self.inputs = {"goal": "DDD", "policy": False, "manual": 0.0, "grab": None, "reset": None}
        self.state = equilibrium("DDD")
        self.snapshot: dict = {}
        self.status = "starting"
        self.ms = 0.0
        self.rt = 1.0

    def teacher(self, goal: str):
        if goal not in self.teachers:
            self.status = f"compiling {goal} teacher"
            raw = read_config(self.config, goal)
            raw["mppi"]["n_samples"] = self.samples
            _, ctrl, _ = build_teacher(raw)
            ctrl.reset(1)
            ctrl.act(equilibrium(goal))
            self.teachers[goal] = ctrl
        return self.teachers[goal]

    def update(self, msg: dict) -> None:
        with self.lock:
            for k in ("goal", "policy", "manual", "grab", "reset"):
                if k in msg:
                    self.inputs[k] = msg[k]

    def run(self) -> None:
        torch.set_grad_enabled(False)
        p = self.plant
        dt = p.dt
        goal, driving, ctrl = None, False, None
        wall0, sim_t = time.perf_counter(), 0.0
        while True:
            t_step = time.perf_counter()
            with self.lock:
                inp = dict(self.inputs)
                self.inputs["reset"] = None
            want = str(inp["goal"]).upper()
            if want not in GOAL_IDS:
                want = "DDD"
            if inp["reset"]:
                r = str(inp["reset"]).upper()
                self.state = hang() if r == "HANG" else equilibrium(r if r in GOAL_IDS else "DDD")
                if ctrl is not None:
                    ctrl.reset(1)
            if want != goal:
                goal = want
                ctrl = self.teacher(goal)
                ctrl.reset(1)
            if bool(inp["policy"]) != driving:
                driving = bool(inp["policy"])
                ctrl.reset(1)
            u = 0.0
            if driving:
                t0 = time.perf_counter()
                u = float(ctrl.act(self.state)[0])
                if ctrl._tick % ctrl.cfg.knot == 1:
                    self.ms = (time.perf_counter() - t0) * 1e3
            u = max(-p.force_limit, min(p.force_limit, u + float(inp["manual"] or 0.0)))
            extra = grab_forces(p, self.state, inp["grab"]) if inp["grab"] else None
            self.state = step(p, self.state, torch.tensor([u]), extra)
            if abs(float(self.state[0, 0])) > p.track_limit or not torch.isfinite(self.state).all():
                self.state = hang()
                ctrl.reset(1)
            sim_t += dt
            wall = time.perf_counter() - wall0
            if wall > 2.0:
                self.rt = 0.9 * self.rt + 0.1 * (sim_t / wall)
                wall0, sim_t = time.perf_counter(), 0.0
            self.status = f"{goal} · mppi {self.samples}" if driving else f"{goal} · policy off"
            self.snapshot = {
                "type": "state",
                "state": dict(zip(KEYS, (float(v) for v in self.state[0]))),
                "force": u,
                "goal": goal,
                "policy": driving,
                "rt": round(self.rt, 2),
                "ms": round(self.ms, 1),
                "status": self.status,
            }
            spare = dt - (time.perf_counter() - t_step)
            if spare > 0:
                time.sleep(spare)


async def serve(args) -> None:
    sim = Sim(args.config, args.samples)
    if args.warm:
        for g in GOAL_IDS:
            print(f"compiling {g} ...", flush=True)
            sim.teacher(g)
    sim.start()
    clients: set = set()

    owner = {"ws": None}

    async def handler(ws):
        clients.add(ws)
        try:
            async for raw in ws:
                msg = json.loads(raw)
                # Only the tab the user is interacting with drives the sim;
                # idle tabs (not started) just watch.
                if msg.get("started"):
                    owner["ws"] = ws
                if owner["ws"] is ws or msg.get("started"):
                    sim.update(msg)
        finally:
            clients.discard(ws)
            if owner["ws"] is ws:
                owner["ws"] = None

    async def broadcast():
        while True:
            if clients and sim.snapshot:
                msg = json.dumps(sim.snapshot)
                await asyncio.gather(*(c.send(msg) for c in list(clients)), return_exceptions=True)
            await asyncio.sleep(1 / 60)

    async with websockets.serve(handler, args.host, args.port):
        print(f"triple sim server on ws://{args.host}:{args.port} ({args.samples} samples)", flush=True)
        await broadcast()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mppi/configs/uuu.json")
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--warm", action="store_true", help="compile all 8 teachers before serving")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
