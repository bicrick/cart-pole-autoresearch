#!/usr/bin/env python3
"""TQC trainer for cart-triple UUU specialist (Lim KIEE 2025 path).

First of eight EP specialists. Product reward + Gym plant wrapper.
Do not kill live PPO a/b/c — run on a free L4 when ready (prefer slot C).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = Path(__file__).resolve().parent
for p in (_ROOT, _TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from envs.triple_gym import make_triple_uuu_env  # noqa: E402


def _parse_arch(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="TQC UUU specialist (Lim 8×TQC)")
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--top-quantiles-to-drop", type=int, default=2)
    p.add_argument("--n-quantiles", type=int, default=25)
    p.add_argument("--n-critics", type=int, default=3)
    # Lim Table 1: policy 400→300, critic 3×512
    p.add_argument("--policy-arch", type=str, default="400,300")
    p.add_argument("--critic-arch", type=str, default="512,512,512")
    p.add_argument("--force-limit", type=float, default=40.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--track-limit", type=float, default=2.4)
    p.add_argument("--init-mode", type=str, default="near_target")
    p.add_argument("--init-noise", type=float, default=0.15)
    p.add_argument("--hang-frac", type=float, default=0.05)
    p.add_argument("--wide-frac", type=float, default=0.25)
    p.add_argument("--progress-w", type=float, default=1.0)
    p.add_argument("--cart-barrier-coef", type=float, default=10.0)
    p.add_argument("--alpha-th", type=float, default=0.5)
    p.add_argument("--w-up", type=float, default=5.0)
    p.add_argument("--w-down", type=float, default=1.0)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="policies/tqc-triple-uuu.zip")
    p.add_argument("--save-freq", type=int, default=25_000)
    p.add_argument("--smoke", action="store_true", help="Tiny run for box smoke test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.total_steps = min(int(args.total_steps), 2_500)
        args.learning_starts = min(int(args.learning_starts), 256)
        args.buffer_size = min(int(args.buffer_size), 10_000)
        args.save_freq = 5_000
        args.n_envs = 1

    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import TQC

    run_name = args.run_name or (
        f"tqc-uuu-f{args.force_limit:g}-"
        f"nt{args.init_noise:g}-w{args.wide_frac:g}-h{args.hang_frac:g}"
    )
    log_path = Path(args.logdir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def _make():
        env = make_triple_uuu_env(
            force_limit=args.force_limit,
            max_steps=args.max_steps,
            track_limit=args.track_limit,
            init_mode=args.init_mode,
            init_noise=args.init_noise,
            hang_frac=args.hang_frac,
            wide_frac=args.wide_frac,
            progress_w=args.progress_w,
            cart_barrier_coef=args.cart_barrier_coef,
            alpha_th=args.alpha_th,
            w_up=args.w_up,
            w_down=args.w_down,
            device="cpu",
            seed=args.seed,
        )
        return Monitor(env)

    env = DummyVecEnv([_make for _ in range(max(1, args.n_envs))])
    eval_env = DummyVecEnv([_make])

    policy_kwargs = dict(
        net_arch=dict(pi=_parse_arch(args.policy_arch), qf=_parse_arch(args.critic_arch)),
        n_critics=args.n_critics,
        n_quantiles=args.n_quantiles,
    )

    meta = {
        "algo": "TQC",
        "goal": "UUU",
        "source": "Lim/Ju/Lee KIEE 2025 — 8 specialists; this is EP7/UUU",
        "hypers": {
            "lr": args.lr,
            "gamma": args.gamma,
            "tau": args.tau,
            "buffer": args.buffer_size,
            "batch": args.batch_size,
            "n_critics": args.n_critics,
            "n_quantiles": args.n_quantiles,
            "top_quantiles_to_drop_per_net": args.top_quantiles_to_drop,
            "policy_arch": args.policy_arch,
            "critic_arch": args.critic_arch,
            "force_limit": args.force_limit,
        },
        "init": {
            "mode": args.init_mode,
            "noise": args.init_noise,
            "hang_frac": args.hang_frac,
            "wide_frac": args.wide_frac,
        },
    }
    (log_path / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)

    model = TQC(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        top_quantiles_to_drop_per_net=args.top_quantiles_to_drop,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(Path(args.logdir)),
    )

    save_every = max(args.save_freq // max(args.n_envs, 1), 1)
    callbacks = [
        CheckpointCallback(
            save_freq=save_every,
            save_path=str(ckpt_path.parent / f"{ckpt_path.stem}_ckpts"),
            name_prefix=ckpt_path.stem,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_path.parent / f"{ckpt_path.stem}_best"),
            log_path=str(log_path / "eval"),
            eval_freq=save_every,
            n_eval_episodes=3 if args.smoke else 5,
            deterministic=True,
        ),
    ]

    t0 = time.time()
    model.learn(
        total_timesteps=int(args.total_steps),
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=False,
    )
    model.save(str(ckpt_path.with_suffix("")))  # SB3 adds .zip
    elapsed = time.time() - t0
    print(
        f"DONE steps={args.total_steps} elapsed_s={elapsed:.1f} "
        f"ckpt={ckpt_path} log={log_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
