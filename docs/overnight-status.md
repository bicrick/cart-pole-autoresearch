# Triple overnight status — 2026-09-20 ~12:45 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~14%/5.5GB (train pid **4586** @~100% CPU); TB http://34.148.138.48:6006/ **OK**
- **Square-one LIVE (sq1b):** single slot `sq1b-h1-walls-ent0-nt1-in002` via `scripts/continue-sq1-h1.sh` (INIT_NOISE=0.02). A/C DISARMED. Walls on, near_goal_p=1, hang=0, ENT=0, force40, product. Cold `.sq1b-h1-cold-v1`.
- **Meters (~u40–43):**
  - nt_at_goal/UUU **0.0368→0.0335 flat** — ≪0.15 kill line / ≪0.80 gate
  - nt_align/UUU **−0.075→−0.042** (still negative)
  - nt_at_goal/DDD **0.89→0.30↓**; OOB=0
  - eval/reward **52→228↑** / rollout_reward **−0.97→0.51↑**
  - train/entropy **1.44→0.13↓** (σ collapse under ENT=0)
- **Diagnosis:** early visit≠hold (reward↑ / hold≈0) — same class as in005 FAIL, but **under** hard-kill window (u100–150).
- **This fire GO:** **KEEP**. Leave alone. Next: watch to u100–150; if nt≲0.15 + reward↑ → STOP + quarantine + next simpler wheel.
- NEED_USER_PING: **yes** (think-out-loud)
