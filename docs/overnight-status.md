# Triple overnight status — 2026-09-20 ~11:43 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~15%/5.5GB (train pid **2288** @99% CPU); TB http://34.148.138.48:6006/ 
- **Square-one LIVE (locked):** single slot `sq1-h1-walls-ent0-nt1-in005` via `scripts/continue-sq1-h1.sh`. A/C DISARMED. Walls on, near_goal_p=1, init_noise=0.05, hang=0, ENT=0, force40, product.
- **Meters (~u67 train / eval@u60):**
  - nt_at_goal/UUU **0.034→0.036 flat** (peak 0.038@u50) — ≪0.15 kill line
  - nt_align/UUU **−0.065→−0.037** (still negative; dip −0.080@u50)
  - eval/reward **14.9→226↑** / rollout_reward **−0.97→0.53↑** — reward climbing
  - train/entropy **1.44→−0.24↓↓** (σ collapse under ENT=0)
  - nt_at_goal/DDD **0.90→0.33↓**; OOB=0; policy_loss healthy
- **Diagnosis:** classic **visit≠hold** (reward↑ / hold≈0). Same FAIL class as pre-halt ATRPO/AVC day. Under hard-kill window (**u100–150**); do **not** mid-kill or invent levers.
- **This fire GO:** leave alone. Next fires: watch to u100–150; if nt≲0.15 + reward↑ → STOP + quarantine + ping (no same-day paper lever).
- NEED_USER_PING: **yes** (think-out-loud)
