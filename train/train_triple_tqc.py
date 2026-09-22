#!/usr/bin/env python3
"""TQC trainer for cart-triple UUU hold — fawraw M2 / Lim product recipe.

Faithful reverse-eng of fawraw training/configs/m2_upright_tqc.yaml on our plant:
  target UUU only, init_mode near_target, init_noise 0.05,
  quiet rates ±0.01, fall-kill |θ|>0.6, progress_w=0,
  TQC lr=3e-4 buffer=200k batch=256 tau=0.005 gamma=0.99 net=[128,128]
  n_critics=3 n_quantiles=20 top_drop=2, 150k steps, ep~1000,
  product reward, inelastic walls ON (our plant training wheels).

Do not start a long train from this module without an explicit launch script /
GPU green-light. Smoke: --smoke or SMOKE=1 on scripts/next-train-triple-m2-hold.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = Path(__file__).resolve().parent
for p in (_ROOT, _TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from envs.triple_vec import make_triple_uuu_vec  # noqa: E402
from goals_triple import GOAL_IDS, parse_goal  # noqa: E402
from tqc_bc_anchor import TQCWithBCAnchor  # noqa: E402
from tqc_ver import VERReplayBuffer  # noqa: E402


def resolve_device(name: str) -> str:
    """auto → cuda, else Apple Silicon MPS, else cpu."""
    import torch

    if name and name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _parse_arch(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="TQC hold — fawraw M2 recipe, one Lim specialist (default UUU)"
    )
    # fawraw m2_upright_tqc.yaml defaults
    p.add_argument("--total-steps", type=int, default=150_000)
    p.add_argument(
        "--learning-starts",
        type=int,
        default=0,
        help="SB3 random-action warmup. Must be 0 when BC-warm-starting a hold "
        "or the first 1k steps are ±Fmax and knock UUU over.",
    )
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--top-quantiles-to-drop", type=int, default=2)
    p.add_argument("--n-quantiles", type=int, default=20, help="fawraw M2=20; Lim often 25")
    p.add_argument("--n-critics", type=int, default=3)
    p.add_argument("--policy-arch", type=str, default="128,128")
    p.add_argument("--critic-arch", type=str, default="128,128")
    p.add_argument("--force-limit", type=float, default=40.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--track-limit", type=float, default=2.4)
    p.add_argument(
        "--track-walls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inelastic walls ON (our plant). Lim/fawraw use rail limits; we use walls as training wheels.",
    )
    p.add_argument(
        "--goal",
        "--target-ep",
        dest="goal",
        type=str,
        default="UUU",
        help="Lim specialist target (DDD…UUU or 0..7). Default UUU. "
        "Not a one-hot UVFA — one net per EP.",
    )
    p.add_argument("--init-mode", type=str, default="near_target")
    p.add_argument(
        "--init-noise",
        type=float,
        default=0.05,
        help="Quiet-basin angle/cart half-range; rates xd/ω FIXED ±0.01 (fawraw M2).",
    )
    p.add_argument(
        "--init-noise-min",
        type=float,
        default=0.0,
        help="If >0 and < --init-noise, train ICs sample the half-range uniformly in "
        "[min, max]. Eval env stays at fixed --init-noise. W2 mix: 0.02 with max 0.04.",
    )
    p.add_argument("--hang-frac", type=float, default=0.0, help="M2 hold: 0 (no hang mix)")
    p.add_argument("--wide-frac", type=float, default=0.0, help="M2 hold: 0 (no wide mix)")
    p.add_argument("--progress-w", type=float, default=0.0,
                    help="M2 hold: 0. M4 swing: 1, on weighted error² (not cos-align).")
    p.add_argument("--cart-barrier-coef", type=float, default=10.0)
    p.add_argument(
        "--reward-mode",
        type=str,
        default="product",
        help="product = Lim hold. m4 = fawraw swing cost (error² + vel + cart + barrier).",
    )
    p.add_argument("--vel-cost-coef", type=float, default=0.02,
                    help="M4 swing only. Probe yaml 0.02.")
    p.add_argument("--cart-cost-coef", type=float, default=0.2,
                    help="M4 swing only. Probe yaml 0.2.")
    p.add_argument("--transition-bonus", type=float, default=0.0,
                    help="M4 one-shot arrival bonus. Probe yaml 200. Hold stays 0.")
    p.add_argument("--transition-tol", type=float, default=0.3,
                    help="M4 arrival tolerance (rad). Probe yaml 0.3. Enter-gate stays 0.03.")
    p.add_argument("--transition-steps", type=int, default=100,
                    help="Consecutive in-tol steps before the M4 bonus. Probe yaml 100.")
    p.add_argument("--alpha-th", type=float, default=0.5)
    p.add_argument("--w-up", type=float, default=5.0)
    p.add_argument("--w-down", type=float, default=1.0)
    p.add_argument(
        "--sparse-bonus",
        type=float,
        default=1.0,
        help="W1 stay pressure: add this when at_goal. 0 disables.",
    )
    p.add_argument(
        "--log-std-init",
        type=float,
        default=-4.0,
        help="Actor log_std pin after BC / at init. -4 → σ≈0.018 on [-1,1].",
    )
    p.add_argument(
        "--ent-coef",
        type=str,
        default="0.001",
        help="TQC entropy temperature. Float or 'auto'. Hold recipe: small fixed.",
    )
    p.add_argument(
        "--target-entropy",
        type=str,
        default="auto",
        help="Only used when --ent-coef auto. Hold: try -0.1.",
    )
    p.add_argument(
        "--actor-freeze-steps",
        type=int,
        default=20_000,
        help="Keep BC actor frozen while critics learn. 0 = never freeze.",
    )
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="TQC updates per vec-step. Set equal to --n-envs to keep 1 update per env-step.",
    )
    p.add_argument("--env-device", type=str, default="", help="Physics device (default: same as --device)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="policies/tqc-m2-uuu-hold.zip")
    p.add_argument("--save-freq", type=int, default=25_000)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--smoke", action="store_true", help="Tiny run for box smoke test")
    p.add_argument(
        "--bc-checkpoint",
        type=str,
        default="",
        help="Optional BC .pt (policies/bc-lqr-uuu.pt) to warm-start TQC actor (latent_pi + mu). "
        "log_std left at TQC init. Arch must match --policy-arch.",
    )
    p.add_argument(
        "--bc-reg-coef",
        type=float,
        default=0.0,
        help="TD3+BC-style MSE to the snapped BC actor, scaled by |Q|. "
        "0 disables. Hold stay after unfreeze: 2.5.",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Load an existing TQC zip and continue (W2+). Does not overwrite --checkpoint.",
    )
    p.add_argument(
        "--ver",
        action="store_true",
        help="Baek VER: store left–right flipped (obs, action) in the replay buffer.",
    )
    p.add_argument(
        "--hard-ic-path",
        type=str,
        default="",
        help="NPZ of early-death 8-D ICs (train/hard_ics.py). Train env only.",
    )
    p.add_argument(
        "--hard-ic-frac",
        type=float,
        default=0.0,
        help="Fraction of train resets drawn from --hard-ic-path (+ mirrors).",
    )
    return p.parse_args()


class AtGoalRateCallback:
    """Lazy-import BaseCallback subclass factory — keeps import light for --help/smoke import."""

    @staticmethod
    def make(log_every: int = 5, verbose: int = 1):
        from stable_baselines3.common.callbacks import BaseCallback

        class _CB(BaseCallback):
            """Log episode-end survival (primary) + at_goal (TB + stdout)."""

            def __init__(self):
                super().__init__(verbose)
                self._surv: list[float] = []
                self._goal: list[float] = []
                self._log_every = log_every

            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    if "episode" not in info:
                        continue
                    # PRIMARY M2: survival (is_success); also log final at_goal.
                    surv: Any = info.get("is_success", info.get("survival_success"))
                    goal: Any = info.get("at_goal_final", info.get("at_goal"))
                    if surv is None and goal is None:
                        continue
                    if surv is not None:
                        self._surv.append(1.0 if surv else 0.0)
                    if goal is not None:
                        self._goal.append(1.0 if goal else 0.0)
                    if len(self._surv) >= self._log_every:
                        srate = sum(self._surv) / len(self._surv)
                        grate = (
                            sum(self._goal) / len(self._goal) if self._goal else float("nan")
                        )
                        self.logger.record("rollout/success_rate", srate)  # primary=survival
                        self.logger.record("rollout/survival_success", srate)
                        self.logger.record("rollout/at_goal", grate)
                        print(
                            f"[M2 hold] PRIMARY survival_success={srate:.3f} "
                            f"at_goal={grate:.3f} "
                            f"(last {len(self._surv)} eps, step={self.num_timesteps})",
                            flush=True,
                        )
                        self._surv.clear()
                        self._goal.clear()
                return True

        return _CB()


def freeze_actor_callback(freeze_steps: int, log_std_init: float):
    from stable_baselines3.common.callbacks import BaseCallback

    class _Freeze(BaseCallback):
        """Train critics on BC hold data before touching the actor."""

        def __init__(self):
            super().__init__(verbose=1)
            self.freeze_steps = int(freeze_steps)
            self.log_std_init = float(log_std_init)
            self._frozen = False

        def _actor_params(self):
            return list(self.model.actor.parameters())

        def _freeze(self) -> None:
            for p in self._actor_params():
                p.requires_grad = False
            self._frozen = True
            print(
                f"[hold] actor FROZEN for first {self.freeze_steps} steps (critics only)",
                flush=True,
            )

        def _unfreeze(self) -> None:
            for p in self._actor_params():
                p.requires_grad = True
            pin_actor_log_std(self.model, self.log_std_init)
            self._frozen = False
            print(f"[hold] actor UNFROZEN at step {self.num_timesteps}", flush=True)

        def _on_training_start(self) -> None:
            if self.freeze_steps > 0:
                self._freeze()

        def _on_step(self) -> bool:
            if self._frozen and self.num_timesteps >= self.freeze_steps:
                self._unfreeze()
            return True

    return _Freeze()


def pin_actor_log_std(model, log_std_init: float) -> None:
    """Force near-deterministic actions: log_std(s) ≈ log_std_init.

    SB3 uses a Linear log_std head. Random init ≈ σ=1 on [-1,1] and immediately
    knocks a BC/LQR hold over. Zero the weights and set the bias.
    """
    import torch

    actor = model.policy.actor
    log_std = getattr(actor, "log_std", None)
    if log_std is None:
        return
    with torch.no_grad():
        if hasattr(log_std, "weight") and log_std.weight is not None:
            log_std.weight.zero_()
        if hasattr(log_std, "bias") and log_std.bias is not None:
            log_std.bias.fill_(float(log_std_init))
        elif torch.is_tensor(log_std):
            log_std.fill_(float(log_std_init))
    print(f"[hold] pinned actor log_std ≈ {log_std_init} (σ≈{2.71828 ** float(log_std_init):.4f})", flush=True)


def load_bc_into_actor(model, bc_path: str, log_std_init: float = -4.0) -> dict:
    """Warm-start TQC actor (latent_pi + mu) from BCActor .pt checkpoint.

    BC saves Sequential net.0/2/4 (Linear layers); TQC Actor has latent_pi.0/2 + mu.
    log_std is pinned low so the cloned hold is not destroyed by σ≈1 noise.
    """
    import torch

    path = Path(bc_path)
    if not path.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"BC ckpt must be dict with state_dict: {path}")
    bc_sd = payload["state_dict"]
    arch = list(payload.get("arch") or [])
    actor = model.policy.actor
    actor_sd = actor.state_dict()

    # Map BC Linear indices (0,2,4,...) → latent_pi then mu
    bc_linear_idxs = sorted(
        int(k.split(".")[1])
        for k in bc_sd
        if k.startswith("net.") and k.endswith(".weight")
    )
    if len(bc_linear_idxs) < 2:
        raise ValueError(f"BC net has too few Linear layers: {bc_linear_idxs}")

    mapped: list[str] = []
    with torch.no_grad():
        # Hidden layers → latent_pi even indices 0,2,4,...
        for i, bc_i in enumerate(bc_linear_idxs[:-1]):
            dest_i = i * 2
            for suffix in ("weight", "bias"):
                src_k = f"net.{bc_i}.{suffix}"
                dst_k = f"latent_pi.{dest_i}.{suffix}"
                if src_k not in bc_sd:
                    raise KeyError(src_k)
                if dst_k not in actor_sd:
                    raise KeyError(
                        f"TQC actor missing {dst_k}; check --policy-arch vs BC arch={arch}"
                    )
                if actor_sd[dst_k].shape != bc_sd[src_k].shape:
                    raise ValueError(
                        f"shape mismatch {src_k}{tuple(bc_sd[src_k].shape)} → "
                        f"{dst_k}{tuple(actor_sd[dst_k].shape)}"
                    )
                actor_sd[dst_k].copy_(bc_sd[src_k])
                mapped.append(f"{src_k}→{dst_k}")
        # Final Linear → mu
        bc_last = bc_linear_idxs[-1]
        for suffix in ("weight", "bias"):
            src_k = f"net.{bc_last}.{suffix}"
            dst_k = f"mu.{suffix}"
            if actor_sd[dst_k].shape != bc_sd[src_k].shape:
                raise ValueError(
                    f"shape mismatch {src_k}{tuple(bc_sd[src_k].shape)} → "
                    f"{dst_k}{tuple(actor_sd[dst_k].shape)}"
                )
            actor_sd[dst_k].copy_(bc_sd[src_k])
            mapped.append(f"{src_k}→{dst_k}")
        actor.load_state_dict(actor_sd)
        pin_actor_log_std(model, log_std_init)

    report = {
        "bc_path": str(path),
        "bc_arch": arch,
        "obs_dim": payload.get("obs_dim"),
        "act_dim": payload.get("act_dim"),
        "mapped": mapped,
        "log_std_init": float(log_std_init),
    }
    print(f"[warm-start] loaded BC actor from {path}", flush=True)
    print(f"[warm-start] mapped {len(mapped)} tensors: {mapped}", flush=True)
    return report


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.total_steps = min(int(args.total_steps), 2_500)
        args.learning_starts = min(int(args.learning_starts), 256)
        args.buffer_size = min(int(args.buffer_size), 10_000)
        args.save_freq = 5_000
        args.eval_freq = 1_000
        args.n_eval_episodes = 3
        args.n_envs = max(1, min(int(args.n_envs), 4))
        args.gradient_steps = min(int(args.gradient_steps), args.n_envs)

    args.device = resolve_device(args.device)
    args.env_device = resolve_device(args.env_device or args.device)
    goal_id = parse_goal(args.goal)
    goal_name = GOAL_IDS[goal_id]

    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    walls_tag = "walls" if args.track_walls else "nowalls"
    run_name = args.run_name or (
        f"m2-hold-{goal_name.lower()}-f{args.force_limit:g}-{walls_tag}-"
        f"nt{args.init_noise:g}-tqc"
    )
    log_path = Path(args.logdir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def _make_vec(
        n: int,
        seed: int,
        noise_min: float = 0.0,
            hard_ic_path: str = "",
            hard_ic_frac: float = 0.0,
        ):
        return make_triple_uuu_vec(
            goal=goal_name,
            n_envs=n,
            force_limit=args.force_limit,
            max_steps=args.max_steps,
            track_limit=args.track_limit,
            track_walls=bool(args.track_walls),
            init_mode=args.init_mode,
            init_noise=args.init_noise,
            init_noise_min=float(noise_min),
            hang_frac=args.hang_frac,
            wide_frac=args.wide_frac,
            progress_w=args.progress_w,
            cart_barrier_coef=args.cart_barrier_coef,
            alpha_th=args.alpha_th,
            w_up=args.w_up,
            w_down=args.w_down,
            sparse_bonus=args.sparse_bonus,
            reward_mode=args.reward_mode,
            vel_cost_coef=args.vel_cost_coef,
            cart_cost_coef=args.cart_cost_coef,
            transition_bonus=args.transition_bonus,
            transition_tol=args.transition_tol,
            transition_steps=args.transition_steps,
            device=args.env_device,
            seed=seed,
            hard_ic_path=hard_ic_path,
            hard_ic_frac=hard_ic_frac,
            # fawraw M4: bottom swing-up must not angle-fall at step 0.
            # Near-target hold keeps the fall kill.
            angle_fall=(args.init_mode or "near_target").lower() != "bottom",
        )

    env = _make_vec(
        max(1, args.n_envs),
        args.seed,
        noise_min=args.init_noise_min,
        hard_ic_path=args.hard_ic_path,
        hard_ic_frac=float(args.hard_ic_frac),
    )
    eval_env = _make_vec(1, args.seed + 1, noise_min=0.0)
    if args.hard_ic_path:
        print(
            f"[hold] hard-IC oversample path={args.hard_ic_path} "
            f"frac={args.hard_ic_frac:g} (eval env uniform)",
            flush=True,
        )

    # Shared [128,128] for pi and qf (fawraw M2); n_critics / n_quantiles via policy_kwargs
    policy_kwargs = dict(
        net_arch=dict(pi=_parse_arch(args.policy_arch), qf=_parse_arch(args.critic_arch)),
        n_critics=args.n_critics,
        n_quantiles=args.n_quantiles,
        log_std_init=float(args.log_std_init),
    )
    try:
        ent_coef: float | str = float(args.ent_coef)
    except ValueError:
        ent_coef = args.ent_coef
    try:
        target_entropy: float | str = float(args.target_entropy)
    except ValueError:
        target_entropy = args.target_entropy

    meta = {
        "algo": "TQC",
        "goal": goal_name,
        "goal_id": goal_id,
        "recipe": "fawraw M2 hold (m2_upright_tqc.yaml) + Lim one specialist per EP",
        "source": "fawraw/triple-pendulum-sim2real + Lim/Ju/Lee KIEE 2025 product reward",
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
            "track_walls": bool(args.track_walls),
            "track_limit": args.track_limit,
            "max_steps": args.max_steps,
            "total_steps": args.total_steps,
            "n_envs": args.n_envs,
            "gradient_steps": args.gradient_steps,
            "device": args.device,
            "env_device": args.env_device,
            "reward": args.reward_mode,
            "vel_cost_coef": args.vel_cost_coef,
            "cart_cost_coef": args.cart_cost_coef,
            "transition_bonus": args.transition_bonus,
            "transition_tol": args.transition_tol,
            "transition_steps": args.transition_steps,
            "sparse_bonus": args.sparse_bonus,
            "log_std_init": args.log_std_init,
            "ent_coef": args.ent_coef,
            "bc_reg_coef": args.bc_reg_coef,
            "ver": bool(args.ver),
            "hard_ic_path": args.hard_ic_path or None,
            "hard_ic_frac": float(args.hard_ic_frac),
            "resume_from": args.resume_from or None,
        },
        "init": {
            "mode": args.init_mode,
            "noise": args.init_noise,
            "noise_min": args.init_noise_min,
            "hang_frac": args.hang_frac,
            "wide_frac": args.wide_frac,
            "quiet_basin": "θ ~ target ±init_noise; x ±init_noise; xd,ω FIXED ±0.01; fall UP 0.6 / DOWN 1.5",
            "progress_w": args.progress_w,
        },
        "eval_primary": {
            "m2_hold": "survival_success (ep_len >= 0.8*max_steps, no fall/oob)",
            "also_log": ["at_goal", "rollout/survival_success", "rollout/at_goal"],
        },
    }
    (log_path / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"=== M2 {goal_name} hold (fawraw M2 + Lim specialist) ===", flush=True)
    print(f"device={args.device} env_device={args.env_device} n_envs={args.n_envs}", flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    print(
        "Primary metrics: rollout/success_rate, rollout/at_goal, eval/success_rate "
        "(not ep_rew_mean alone).",
        flush=True,
    )

    resume = (args.resume_from or "").strip()
    if resume:
        load = Path(resume)
        load_s = str(load.with_suffix("") if load.suffix == ".zip" else load)
        model = TQCWithBCAnchor.load(load_s, env=env, device=args.device)
        model.bc_reg_coef = float(args.bc_reg_coef)
        model.learning_starts = int(args.learning_starts)
        model.gradient_steps = max(1, int(args.gradient_steps))
        model.batch_size = int(args.batch_size)
        model.tensorboard_log = str(Path(args.logdir))
        print(f"[hold] resumed TQC from {resume}", flush=True)
        pin_actor_log_std(model, args.log_std_init)
        if args.ver:
            model.replay_buffer = VERReplayBuffer(
                model.buffer_size,
                model.observation_space,
                model.action_space,
                device=model.device,
                n_envs=model.n_envs,
            )
            print("[hold] VER replay buffer attached after resume", flush=True)
    else:
        rb_kwargs: dict[str, Any] = {}
        if args.ver:
            rb_kwargs["replay_buffer_class"] = VERReplayBuffer
        model = TQCWithBCAnchor(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            top_quantiles_to_drop_per_net=args.top_quantiles_to_drop,
            gradient_steps=max(1, int(args.gradient_steps)),
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(Path(args.logdir)),
            bc_reg_coef=float(args.bc_reg_coef),
            **rb_kwargs,
        )
        pin_actor_log_std(model, args.log_std_init)
        if args.ver:
            print("[hold] VER replay flip ON (each transition stored + mirrored)", flush=True)
        if args.bc_checkpoint:
            ws = load_bc_into_actor(model, args.bc_checkpoint, log_std_init=args.log_std_init)
            meta["warm_start"] = ws
            (log_path / "run_meta.json").write_text(json.dumps(meta, indent=2))
    if float(args.bc_reg_coef) > 0:
        model.snapshot_actor_anchor()

    save_every = max(args.save_freq // max(args.n_envs, 1), 1)
    eval_every = max(args.eval_freq // max(args.n_envs, 1), 1)
    callbacks = [
        freeze_actor_callback(args.actor_freeze_steps, args.log_std_init),
        AtGoalRateCallback.make(log_every=5 if not args.smoke else 1, verbose=1),
        CheckpointCallback(
            save_freq=save_every,
            save_path=str(ckpt_path.parent / f"{ckpt_path.stem}_ckpts"),
            name_prefix=ckpt_path.stem,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_path.parent / f"{ckpt_path.stem}_best"),
            log_path=str(log_path / "eval"),
            eval_freq=eval_every,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            verbose=1,
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
    print(
        "Check TB: rollout/success_rate (=survival PRIMARY), rollout/at_goal, eval/success_rate",
        flush=True,
    )


if __name__ == "__main__":
    main()
