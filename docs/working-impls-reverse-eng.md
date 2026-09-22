# Working triple-pendulum implementations — reverse engineering

Last updated: 2026-09-21  
Purpose: stop circling. Copy what already held UUU / hit 56. W2 grind is paused; findings in `docs/triple-uuu-hold-findings.md`.

## Who actually succeeded

| Source | Code? | Algo | What worked | Honest status |
|---|---|---|---|---|
| **Lim / Ju / Lee KIEE 2025** | No public repo | **TQC**, **8 EP specialists**, product reward | **All 56 on hardware** | Gold standard. PDF: ecsl.inha.ac.kr |
| **fawraw/triple-pendulum-sim2real** | **Yes** (open) | **TQC** (sb3-contrib), MuJoCo | M2 UUU stabilize; M3 8-EP ~72.5% sim; M4 56 **not done** (needs swing→hold handoff) | Best open recipe to copy |
| **Baek et al. EAAI 2024** | No official repo | **SAC + VER** (left-right mirror) | Hardware **swing-up to top** (1 goal) | Validates off-policy + product + symmetry |
| **Glück Automatica 2013** | Classical | Energy FF + TV-LQR | Experimental DDD→UUU | Not RL; proves plant is controllable with handoff |
| SustcFYF/RL-python-cpp | Yes | RL + C++ sim | Underactuated tip | Different plant; lower priority |

## fawraw M2 hold recipe (copy this first)

From `training/configs/m2_upright_tqc.yaml` + `sim/envs/triple_pendulum_env.py`:

- **Task:** stabilize EP7 UUU only — “starting state already near upright; policy only rejects perturbations”
- `init_mode: near_target`, `init_noise: 0.05`
- **Quiet rates (critical):** `qvel[:] ~ U(±0.01)` fixed — **not** scaled by `init_noise`. Angles/cart `±init_noise`.
- **Fall-kill:** any UP-target link `|θ_err| > 0.6` → terminate. (DOWN links use 1.5 when multi-EP.)
- `progress_reward_coef: 0` (hold recipe — no swing progress term)
- **Algo: TQC** (not PPO), lr 3e-4, γ 0.99, τ 0.005, buffer 200k, batch 256
- Net `[128,128]`, 3 critics, 20 quantiles, drop top 2
- 150k timesteps, 1000-step episodes
- **Primary success (M2 hold):** survival — `ep_len >= 0.8 * max_steps` without fall/oob. Also log final `at_goal`.

## Why our earlier M2 TQC failed (env mismatch, not “TQC bad”)

| We did | fawraw M2 | Gap |
|---|---|---|
| `near_target` with **ω ± init_noise×5**, **xd ± init_noise×2** | rates **fixed ±0.01** | Blew the quiet basin; LQR itself fails there |
| No angle fall-kill in gym wrapper | `|θ|>0.6` terminate | Episodes wandered; success signal muddy |
| `progress_w=1` on a hold task | `progress_w=0` | Swing-shaped reward on a stabilize task |
| PPO / UVFA zoo before this | TQC specialist + staged milestones | Wrong process (secondary) |

**New path (do in order, no GPU until oracle PASS):**

1. **Env contract** — quiet rates ±0.01, fall-kill 0.6, `progress_w=0`, walls ON (`train/envs/triple_gym.py`)
2. **LQR oracle gate** — ≥95% survival @ `init_noise∈{0.01,0.02}`, N≥50; dump demos
3. **BC** from `policies/lqr-uuu-demos.npz`
4. **Short TQC fine-tune** on the same contract (only after BC)

Do **not** resume ATRPO/gSDE/ENT ladder. Do **not** start GCP VM until oracle PASS + user green-light.

## Implemented here

1. **Env contract** — `train/envs/triple_gym.py` (+ matching `train_triple.random_states` near_target). Assert: `python3 train/test_quiet_basin_ics.py`
2. **LQR oracle** — `train/lqr_oracle_uuu.py` / `scripts/lqr-oracle-uuu.sh`  
   ```bash
   bash scripts/lqr-oracle-uuu.sh          # CPU; PASS → policies/lqr-uuu-demos.npz
   ```
3. **BC from LQR demos** — `train/bc_lqr_uuu.py` / `scripts/bc-lqr-uuu.sh` → `policies/bc-lqr-uuu.pt` (PASS @ 0.01/0.02)
4. **M2 hold trainer** — `train/train_triple_tqc.py` defaults: quiet basin, `progress_w=0`, survival primary. Launch: `scripts/next-train-triple-m2-hold.sh` (walls ON). Smoke only until oracle PASS + green-light.
5. **Eval primary (M2):** `rollout/success_rate` = **survival_success**; also `rollout/at_goal`.


## Oracle → BC status (2026-09-20)

| Gate | Status | Notes |
|---|---|---|
| **LQR oracle** | **PASS** | survival 1.0 @ init_noise 0.01 & 0.02 (N=50); demos `policies/lqr-uuu-demos.npz` (80 eps, actions in Newtons) |
| **BC** | **PASS** | `train/bc_lqr_uuu.py` / `scripts/bc-lqr-uuu.sh` — MLP [128,128] on 11-D `observe` features, actions Newtons→[-1,1]; ckpt `policies/bc-lqr-uuu.pt` |
| BC eval @ 0.01 | survival **1.00** / at_goal **1.00** (N=50) | |
| BC eval @ 0.02 | survival **1.00** / at_goal **1.00** (N=50) | |
| **TQC FT** | ready (not started) | needs user GPU green-light; do **not** start GCP yet |

```bash
bash scripts/bc-lqr-uuu.sh   # CPU; train + eval → policies/bc-lqr-uuu.pt
```

## Lim recipe (for later, after hold)

- TQC, **one policy per equilibrium** (8 heads), product reward ∈[0,1]
- Wide random ICs for transition capability
- Train until return plateaus ~700–800 / 1000
- Transitions “for free” by switching specialists — not one UVFA for 56

## Sources

- https://github.com/fawraw/triple-pendulum-sim2real
- http://ecsl.inha.ac.kr/publication/KIEE2025_b.pdf
- Baek EAAI 2024 doi:10.1016/j.engappai.2023.107518
