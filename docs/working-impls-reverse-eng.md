# Working triple-pendulum implementations — reverse engineering

Last updated: 2026-09-20 ~13:10 CT  
Purpose: stop circling. Copy what already held UUU / hit 56.

## Who actually succeeded

| Source | Code? | Algo | What worked | Honest status |
|---|---|---|---|---|
| **Lim / Ju / Lee KIEE 2025** | No public repo | **TQC**, **8 EP specialists**, product reward | **All 56 on hardware** | Gold standard. PDF: ecsl.inha.ac.kr |
| **fawraw/triple-pendulum-sim2real** | **Yes** (open) | **TQC** (sb3-contrib), MuJoCo | M2 UUU stabilize; M3 8-EP ~72.5% sim; M4 56 **not done** (needs swing→hold handoff) | Best open recipe to copy |
| **Baek et al. EAAI 2024** | No official repo | **SAC + VER** (left-right mirror) | Hardware **swing-up to top** (1 goal) | Validates off-policy + product + symmetry |
| **Glück Automatica 2013** | Classical | Energy FF + TV-LQR | Experimental DDD→UUU | Not RL; proves plant is controllable with handoff |
| SustcFYF/RL-python-cpp | Yes | RL + C++ sim | Underactuated tip | Different plant; lower priority |

## fawraw M2 hold recipe (copy this first)

From `training/configs/m2_upright_tqc.yaml` (verbatim intent):

- **Task:** stabilize EP7 UUU only — “starting state already near upright; policy only rejects perturbations”
- `init_mode: near_target`, `init_noise: 0.05`
- **Algo: TQC** (not PPO), lr 3e-4, γ 0.99, τ 0.005, buffer 200k, batch 256
- Net `[128,128]`, 3 critics, 20 quantiles, drop top 2
- 150k timesteps, 1000-step episodes
- Success = hold metric, not mean reward

## Lim recipe (for later, after hold)

- TQC, **one policy per equilibrium** (8 heads), product reward ∈[0,1]
- Wide random ICs for transition capability
- Train until return plateaus ~700–800 / 1000
- Transitions “for free” by switching specialists — not one UVFA for 56

## Why our 2 days failed (map to their recipe)

| We did | They did | Gap |
|---|---|---|
| PPO on-policy hold | **TQC / SAC** off-policy | Wrong algo family for continuous force hold |
| One UVFA + entropy/ATRPO zoo | **Specialist + staged milestones** | Wrong architecture / process |
| Optimized TB reward | **Success / at-goal hold** | Reward↑ hold≈0 = visit≠hold |
| `near_target` with **ω±0.1 and cart floors** | Quiet near-upright basin | Our “tighten INIT_NOISE” was partly a **no-op** |
| Mixed swing+hold same net | **Handoff** (fawraw M4) / Lim specialists | Need separate catcher |

## Implemented here (faithful reimpl, not knob zoo)

1. **Quiet-basin ICs** — done in `train/train_triple.py` (`random_states` near_target / near_goal) and `train/envs/triple_gym.py`. `init_noise` scales θ, ω (×5), x/xd (×2) with **1e-3 absolute floors only** (legacy cart≥0.05 / ω≥0.1 removed). Assert: `python3 train/test_quiet_basin_ics.py`.
2. **M2 hold trainer** — `train/train_triple_tqc.py` defaults match fawraw yaml; product reward; walls ON for our plant.
3. **Launch script:** [`scripts/next-train-triple-m2-hold.sh`](../scripts/next-train-triple-m2-hold.sh)
   ```bash
   # smoke (CPU, few steps) — no VM
   SMOKE=1 bash scripts/next-train-triple-m2-hold.sh
   # full GPU train — only when user green-lights
   DEVICE=cuda bash scripts/next-train-triple-m2-hold.sh
   ```
4. **Eval primary:** `rollout/success_rate`, `rollout/at_goal`, `eval/success_rate` (logged prominently; not ep_rew_mean alone).
5. **Only after M2 hold works:** M3 multi-EP / Lim 8 specialists; then swing→handoff (fawraw M4).

Do **not** resume ATRPO/gSDE/ENT ladder.

## Sources

- https://github.com/fawraw/triple-pendulum-sim2real
- http://ecsl.inha.ac.kr/publication/KIEE2025_b.pdf
- Baek EAAI 2024 doi:10.1016/j.engappai.2023.107518
