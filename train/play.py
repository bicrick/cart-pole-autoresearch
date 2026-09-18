#!/usr/bin/env python3
"""Pygame viewer for the live checkpoint + real physics.

Mouse: grab the cart or either pole and drag / flick.
A/D or arrows: shove the cart.
Click a goal chip, or 1-4 / Tab: UU UD DU DD.
P: toggle the policy. The file is reloaded if checkpoint.pt changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pygame
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from goals import GOAL_ANGLES, GOAL_IDS, GOAL_INDEX, OBS_DIM, at_goal, conditioned_obs  # noqa: E402
from physics import (  # noqa: E402
    grab_forces,
    load_constants,
    normalize_obs,
    observe,
    step,
    tip_positions,
)
from ppo import ActorCritic, tanh_action  # noqa: E402


def make_obs(state, goal_ids, constants):
    """Same 16-d UVFA obs the trainer uses."""
    return normalize_obs(conditioned_obs(observe(state), goal_ids), constants)


PAPER = (251, 251, 250)
INK = (33, 35, 45)
INK_MUTED = (90, 94, 107)
LINK = (37, 99, 235)
RAIL = (232, 232, 230)
CART = (42, 45, 56)
GHOST = (160, 164, 176)

STIFFNESS = 55.0
DAMPING = 6.0
HIT = {"cart": 0.16, "lower": 0.12, "upper": 0.12}
BODY_ID = {"cart": 0, "lower": 1, "upper": 2}
DEFAULT_CKPT = ROOT / "policies" / "checkpoint.pt"
DEFAULT_JSON = ROOT / "policies" / "policy.json"


def _first_linear_in(state_dict):
    for key, value in state_dict.items():
        if key.endswith("weight") and value.ndim == 2:
            return int(value.shape[1])
    return None


def load_actor(constants, device, ckpt_path, json_path):
    hidden = constants["hidden"]
    meta = {"update": None, "path": None}

    if ckpt_path.is_file():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        in_dim = payload.get("obs_dim") if isinstance(payload, dict) else None
        in_dim = in_dim or _first_linear_in(state)
        if in_dim != OBS_DIM:
            print(f"checkpoint obs_dim={in_dim} != {OBS_DIM}; not loading", flush=True)
        else:
            model = ActorCritic(obs_dim=OBS_DIM, hidden=hidden).to(device)
            model.load_state_dict(state)
            model.eval()
            update = payload.get("update") if isinstance(payload, dict) else None
            meta = {"update": update, "path": ckpt_path}
            label = f"live checkpoint @ {update}" if update is not None else f"live {ckpt_path.name}"
            print(f"loaded {label} ({OBS_DIM}d) from {ckpt_path}", flush=True)
            return model, label, meta

    if json_path.is_file():
        spec = json.loads(json_path.read_text())
        if int(spec.get("obs_dim", 0)) != OBS_DIM:
            print(f"policy.json obs_dim={spec.get('obs_dim')} != {OBS_DIM}; not loading", flush=True)
            return None, "physics only", meta
        model = ActorCritic(obs_dim=OBS_DIM, hidden=hidden).to(device)
        layers = [layer for layer in spec.get("layers", []) if layer.get("type") == "linear"]
        linears = [m for m in model.actor if isinstance(m, torch.nn.Linear)]
        if len(layers) == len(linears) and len(layers[0]["weight"][0]) == OBS_DIM:
            with torch.no_grad():
                for module, layer in zip(linears, layers):
                    module.weight.copy_(torch.tensor(layer["weight"], device=device))
                    module.bias.copy_(torch.tensor(layer["bias"], device=device))
            model.eval()
            meta = {"update": None, "path": json_path}
            print(f"loaded policy.json ({OBS_DIM}d)", flush=True)
            return model, f"policy.json ({OBS_DIM}d)", meta
    return None, "physics only", meta


class Camera:
    def __init__(self, width, height, scale=180.0):
        self.width = width
        self.height = height
        self.scale = scale
        self.x = 0.0
        self.rail_y_frac = 0.62

    def follow(self, x):
        self.x += 0.14 * (x - self.x)

    def world_to_screen(self, wx, wy):
        sx = self.width * 0.5 + (wx - self.x) * self.scale
        sy = self.height * self.rail_y_frac - wy * self.scale
        return int(sx), int(sy)

    def screen_to_world(self, sx, sy):
        wx = self.x + (sx - self.width * 0.5) / self.scale
        wy = (self.height * self.rail_y_frac - sy) / self.scale
        return wx, wy


def pick_body(world, cart, lower, upper):
    wx, wy = world
    candidates = [
        ("cart", ((wx - cart[0]) ** 2 + (wy - cart[1]) ** 2) ** 0.5, HIT["cart"]),
        ("lower", ((wx - lower[0]) ** 2 + (wy - lower[1]) ** 2) ** 0.5, HIT["lower"]),
        ("upper", ((wx - upper[0]) ** 2 + (wy - upper[1]) ** 2) ** 0.5, HIT["upper"]),
    ]
    hits = [c for c in candidates if c[1] <= c[2]]
    if hits:
        return min(hits, key=lambda c: c[1])[0]
    nearest = min(candidates, key=lambda c: c[1])
    return nearest[0] if nearest[1] < 0.32 else None


def tips_xy(state, constants):
    p1x, p1y, p2x, p2y = tip_positions(state, constants)
    x = float(state[0, 0])
    return (x, 0.0), (float(p1x[0]), float(p1y[0])), (float(p2x[0]), float(p2y[0]))


def ghost_tips(cart_x, goal_name, constants):
    th1, th2 = GOAL_ANGLES[GOAL_INDEX[goal_name]].tolist()
    l1 = float(constants["poleLength1"])
    l2 = float(constants["poleLength2"])
    p1x = cart_x + l1 * torch.sin(torch.tensor(th1)).item()
    p1y = l1 * torch.cos(torch.tensor(th1)).item()
    p2x = p1x + l2 * torch.sin(torch.tensor(th2)).item()
    p2y = p1y + l2 * torch.cos(torch.tensor(th2)).item()
    return (cart_x, 0.0), (p1x, p1y), (p2x, p2y)


def goal_chip_rects(width):
    chip_w, chip_h, gap = 72, 28, 8
    total = 4 * chip_w + 3 * gap
    x0 = width - total - 16
    y0 = 12
    return [
        (name, pygame.Rect(x0 + i * (chip_w + gap), y0, chip_w, chip_h))
        for i, name in enumerate(GOAL_IDS)
    ]


def draw(
    screen,
    font,
    small,
    camera,
    state,
    constants,
    grab,
    goal_name,
    policy_on,
    source,
    force_n,
):
    screen.fill(PAPER)
    cart, lower, upper = tips_xy(state, constants)
    camera.follow(cart[0])
    track = float(constants.get("trackLimit", 2.4))
    left = camera.world_to_screen(-track, 0)
    right = camera.world_to_screen(track, 0)
    pygame.draw.line(screen, RAIL, left, right, 2)
    pygame.draw.line(screen, INK_MUTED, (left[0], left[1] - 14), (left[0], left[1] + 14), 3)
    pygame.draw.line(screen, INK_MUTED, (right[0], right[1] - 14), (right[0], right[1] + 14), 3)

    g0, g1, g2 = ghost_tips(cart[0], goal_name, constants)
    gc = camera.world_to_screen(*g0)
    gp1 = camera.world_to_screen(*g1)
    gp2 = camera.world_to_screen(*g2)
    pygame.draw.lines(screen, GHOST, False, [gc, gp1, gp2], 3)
    pygame.draw.circle(screen, GHOST, gp1, 5)
    pygame.draw.circle(screen, GHOST, gp2, 6)

    c = camera.world_to_screen(*cart)
    p1 = camera.world_to_screen(*lower)
    p2 = camera.world_to_screen(*upper)
    pygame.draw.lines(screen, INK, False, [c, p1, p2], 6)

    cart_w = int(0.28 * camera.scale)
    cart_h = int(0.12 * camera.scale)
    rect = pygame.Rect(c[0] - cart_w // 2, c[1] - cart_h // 3, cart_w, cart_h)
    pygame.draw.rect(screen, CART, rect, border_radius=4)
    if grab == "cart":
        pygame.draw.rect(screen, LINK, rect, width=3, border_radius=4)
    pygame.draw.circle(screen, LINK if grab == "lower" else INK, p1, max(7, int(0.045 * camera.scale)))
    pygame.draw.circle(screen, LINK if grab == "upper" else INK, p2, max(8, int(0.055 * camera.scale)))

    if grab:
        mx, my = pygame.mouse.get_pos()
        origin = {"cart": c, "lower": p1, "upper": p2}[grab]
        pygame.draw.line(screen, LINK, origin, (mx, my), 2)

    gid = torch.tensor([GOAL_INDEX[goal_name]], dtype=torch.long)
    hit = bool(at_goal(state, gid)[0])
    x, xd, th1, _, th2, _ = [float(v) for v in state[0]]
    lines = [
        f"{source}    policy {'ON' if policy_on else 'off'}    goal {goal_name}"
        + ("    at goal" if hit else ""),
        "click a chip or 1-4 / Tab to change desired pose     mouse: flick     A/D: shove cart     P: policy",
        f"x={x:+.2f}  th1={th1:+.2f}  th2={th2:+.2f}  u={force_n:+.1f} N",
    ]
    for i, text in enumerate(lines):
        screen.blit(font.render(text, True, INK if i == 0 else INK_MUTED), (16, 12 + i * 20))

    chips = goal_chip_rects(camera.width)
    for name, chip in chips:
        on = name == goal_name
        pygame.draw.rect(screen, LINK if on else RAIL, chip, border_radius=6)
        pygame.draw.rect(screen, INK if on else INK_MUTED, chip, width=1, border_radius=6)
        label = small.render(name, True, PAPER if on else INK)
        screen.blit(label, label.get_rect(center=chip.center))
    return chips


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive cart-double-pendulum")
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--policy-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--goal", default="UU", choices=GOAL_IDS)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=560)
    return parser.parse_args()


def main():
    args = parse_args()
    constants = load_constants()
    device = torch.device("cpu")
    ckpt_path = args.checkpoint
    json_path = args.policy_json
    model, source, meta = (
        (None, "physics only", {"update": None, "path": None})
        if args.no_policy
        else load_actor(constants, device, ckpt_path, json_path)
    )
    policy_on = model is not None
    goal_name = args.goal
    goal_id = torch.tensor([GOAL_INDEX[goal_name]], dtype=torch.long)
    watched = ckpt_path if ckpt_path.is_file() else json_path
    mtime = watched.stat().st_mtime if watched.is_file() else 0.0

    state = torch.tensor([[0.0, 0.0, 0.08, 0.0, -0.05, 0.0]], dtype=torch.float32)
    pygame.init()
    pygame.display.set_caption("double cart-pole  ·  live checkpoint")
    screen = pygame.display.set_mode((args.width, args.height))
    font = pygame.font.SysFont("menlo,monaco,consolas,dejavusansmono", 15)
    small = pygame.font.SysFont("menlo,monaco,consolas,dejavusansmono", 14)
    clock = pygame.time.Clock()
    camera = Camera(args.width, args.height)
    grab = None
    chips = []
    fmax = float(constants["forceLimit"])
    dt = float(constants["dt"])
    acc = 0.0
    last_force = 0.0

    running = True
    while running:
        acc += clock.tick(60) / 1000.0
        if watched.is_file() and watched.stat().st_mtime > mtime + 0.05:
            mtime = watched.stat().st_mtime
            model, source, meta = load_actor(constants, device, ckpt_path, json_path)
            policy_on = model is not None
            print(f"reloaded {source}", flush=True)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p and model is not None:
                    policy_on = not policy_on
                elif event.key == pygame.K_TAB:
                    nxt = (GOAL_INDEX[goal_name] + 1) % len(GOAL_IDS)
                    goal_name = GOAL_IDS[nxt]
                    goal_id = torch.tensor([nxt], dtype=torch.long)
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    goal_name = GOAL_IDS[event.key - pygame.K_1]
                    goal_id = torch.tensor([GOAL_INDEX[goal_name]], dtype=torch.long)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked = False
                for name, chip in chips:
                    if chip.collidepoint(event.pos):
                        goal_name = name
                        goal_id = torch.tensor([GOAL_INDEX[name]], dtype=torch.long)
                        clicked = True
                        break
                if not clicked:
                    world = camera.screen_to_world(*event.pos)
                    cart, lower, upper = tips_xy(state, constants)
                    grab = pick_body(world, cart, lower, upper)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                grab = None

        keys = pygame.key.get_pressed()
        manual = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            manual -= fmax
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            manual += fmax

        while acc >= dt:
            extra = None
            if grab is not None:
                wx, wy = camera.screen_to_world(*pygame.mouse.get_pos())
                target = torch.tensor([[wx, wy]], dtype=torch.float32)
                body = torch.tensor([BODY_ID[grab]], dtype=torch.long)
                extra = grab_forces(state, target, body, STIFFNESS, DAMPING, constants)
            force = torch.zeros(1)
            if policy_on and model is not None:
                with torch.no_grad():
                    obs = make_obs(state, goal_id, constants)
                    force = tanh_action(model.deterministic(obs), fmax).reshape(1)
            force = (force + manual).clamp(-fmax, fmax)
            last_force = float(force[0])
            state = step(state, force, extra_q=extra, constants=constants)
            acc -= dt

        chips = draw(
            screen,
            font,
            small,
            camera,
            state,
            constants,
            grab,
            goal_name,
            policy_on,
            source,
            last_force,
        )
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
