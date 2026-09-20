# Triple overnight status — 2026-09-20 ~12:55 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~13%/5.5GB (train pid **4586**); TB http://34.148.138.48:6006/ **OK**
- **Square-one LIVE (sq1b):** single slot `sq1b-h1-walls-ent0-nt1-in002` via `scripts/continue-sq1-h1.sh` (INIT_NOISE=0.02). A/C DISARMED. Walls on, near_goal_p=1, hang=0, ENT=0, force40, product. Cold `.sq1b-h1-cold-v1`.
- **Meters (~u80):**
  - nt_at_goal/UUU **0.037→0.036 flat** (0.032–0.038) — ≪0.15 kill / ≪0.80 gate
  - nt_align/UUU **−0.075→−0.032**
  - nt_at_goal/DDD **0.89→0.27↓**; OOB=0
  - eval/reward **52→225↑**
  - train/entropy **1.44→−0.30↓** (σ collapse under ENT=0)
- **Diagnosis:** visit≠hold (reward↑ / hold≈0 / entropy dive) — same class as in005; still under u100–150 kill window (~4 min to u100).
- **This fire GO:** **KEEP**. Next fire: expect hard kill @u100–150 if still flat → launch sq1c INIT_NOISE=0.01.
- NEED_USER_PING: **yes** (think-out-loud)
