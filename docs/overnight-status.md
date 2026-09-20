# Triple overnight status — 2026-09-20 ~02:02 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~99%/15.6GB; TB http://34.148.138.48:6006/ HTTP 200
- **Roles:** A=S1 pid **145281**/w rearmed; B=H1 pid **145275**/w rearmed after ENT stage; C=H1-var pid **145280**/w **146717**. Markers unchanged. TRACK_WALLS=1 all.
- **Meters (~u40–48, ~25min post cold @06:33Z / 01:33 CT):**
  - A S1: reward 222→137 hang_align/UUU **-0.98→-0.11** hang_at_goal~0.002 nt_UUU~0.03 OOB=0 ent **1.45→2.01↑** — healthy swing climb
  - B H1: reward~220 flat nt_UUU~0.034 nt_align **-0.07→-0.007** OOB=0 ent **1.45→0.73↓** (~0.01/u) rollout_rew↑ — **entropy collapse + reward↑/hold≈0**; do **not** mid-kill
  - C H1-var: reward 221→186 nt_align **-0.06→0.05** nt_UUU~0.036 OOB=0 ent **1.46→2.01↑** — healthier hold exploration than B
- **This fire GO:** Stage B `ENT` 0.02→**0.035** (+ RUN_NAME `-ent035`) in `continue-triple-b.sh` / `next-train-triple-b.sh` for **next natural restart** (continues from ckpt; VEL_COST=0.015 already staged). Live train PIDs undisturbed. Handoff smoke still deferred (H1 nt cold).
- NEED_USER_PING: **no**
