# Triple overnight status — 2026-09-20 ~03:55 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~97%/16.6GB; TB http://34.148.138.48:6006/ HTTP 200; uptime ~2d
- **Roles:** A=S1 pid **153439** (LR=1e-4 cold restart @08:46Z); B=H1 pid **145275** / w **147599**; C=H1-var pid **145280** / w **146717**. TRACK_WALLS=1 all.
- **Meters:**
  - A S1: ~**u16** hang_align/UUU **−0.97→−0.83** (climbing), policy_loss sane, ent **~1.68↑**. NaN quarantine held; LR=1e-4 restart **healthy** so far.
  - B H1: ~**u258** nt_UUU **~0.054** nt_align **~0.16** reward~179 ent **≈−0.24** (still collapsing). ENT=0.035+VEL_COST staged; **no mid-kill**; stretch ETA ~**05:15 CT**.
  - C H1-var: ~**u250** nt_UUU **~0.050** nt_align **~0.15** ent **~1.50** — leave alone.
- **This fire GO:** confirm A healthy only; leave B/C alone. No code change.
- NEED_USER_PING: **yes** (standing think-out-loud every fire)
