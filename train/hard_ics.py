"""Oversample early-death ICs for W2 hold widen.

Baek-style: densify the 40-step catch misses, not mix easy 0.02 ICs.
Bank is raw 8-D plant state. Left–right mirrors are included.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = Path(__file__).resolve().parent
for p in (_ROOT, _TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def flip_plant_state(z: np.ndarray) -> np.ndarray:
    return -np.asarray(z, dtype=np.float32)


def load_bank(path: str | Path, device: torch.device) -> torch.Tensor:
    data = np.load(path)
    z = np.asarray(data["states"], dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 8:
        raise ValueError(f"hard-ic bank must be (N,8), got {z.shape}")
    bank = np.concatenate([z, flip_plant_state(z)], axis=0)
    return torch.as_tensor(bank, device=device, dtype=torch.float32)


def mix_hard_ics(
    states: torch.Tensor,
    bank: torch.Tensor,
    frac: float,
    jitter: float = 0.002,
) -> torch.Tensor:
    """Replace a random subset of rows with bank ICs plus tiny jitter."""
    if bank is None or frac <= 0 or bank.numel() == 0:
        return states
    n = int(states.shape[0])
    take = torch.rand(n, device=states.device) < float(frac)
    if not bool(take.any()):
        return states
    k = int(take.sum().item())
    idx = torch.randint(0, int(bank.shape[0]), (k,), device=states.device)
    picked = bank[idx].to(states.device)
    if jitter > 0:
        picked = picked + jitter * (2.0 * torch.rand_like(picked) - 1.0)
    out = states.clone()
    out[take] = picked
    return out


def dump_deaths(
    checkpoint: str,
    out: str,
    *,
    init_noise: float = 0.04,
    n_episodes: int = 50,
    seed: int = 0,
    device: str = "cpu",
) -> Path:
    from sb3_contrib import TQC

    from envs.triple_gym import TriplePendulumUUUEnv

    path = Path(checkpoint)
    load = str(path.with_suffix("") if path.suffix == ".zip" else path)
    model = TQC.load(load, device=device)
    env = TriplePendulumUUUEnv(
        force_limit=40.0,
        max_steps=1000,
        track_walls=True,
        init_mode="near_target",
        init_noise=float(init_noise),
        hang_frac=0.0,
        wide_frac=0.0,
        progress_w=0.0,
        seed=seed,
    )
    deaths = []
    for i in range(int(n_episodes)):
        obs, _ = env.reset(seed=seed + i)
        z0 = env._state.detach().cpu().numpy().reshape(-1).astype(np.float32)
        last = {}
        for _ in range(1000):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, last = env.step(act)
            if term or trunc:
                break
        if not last.get("is_success"):
            deaths.append(z0)
    if not deaths:
        raise RuntimeError("no deaths to dump")
    states = np.stack(deaths, axis=0)
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_p,
        states=states,
        init_noise=np.array(init_noise),
        source=np.array(checkpoint),
        n_episodes=np.array(n_episodes),
    )
    print(
        f"HARD ICS → {out_p} deaths={len(deaths)}/{n_episodes} "
        f"noise={init_noise} (+mirrors at load)",
        flush=True,
    )
    return out_p


def parse_args():
    p = argparse.ArgumentParser(description="Dump early-death ICs from a TQC hold zip")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="policies/hard-ics-n004-0.04.npz")
    p.add_argument("--init-noise", type=float, default=0.04)
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dump_deaths(
        args.checkpoint,
        args.out,
        init_noise=args.init_noise,
        n_episodes=args.n_episodes,
        seed=args.seed,
        device=args.device,
    )
