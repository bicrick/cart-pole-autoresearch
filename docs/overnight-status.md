# Triple overnight status — 2026-09-20 ~07:45 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~99%/16.5GB; TB http://34.148.138.48:6006/ HTTP 200
- **Roles:** A=S1 pid **166936**; B=H1 **ENERGY_W=0.35 RPO+ERA ENT=0 (no gSDE)** pid **166805**; C=H1-var pid **158583**. TRACK_WALLS=1. Watchers continue-a/b/c.
- **Meters:**
  - A S1: ~**u26** hang_align/UUU **~−0.047**, nt_UUU~0.057, ent **~1.14**, OOB=0 — leave alone (new stretch after u400). Marker `.triple-a-s1-walls-v1`.
  - B H1 ENERGY_W=0.35: ~**u38** H **1.77→0.64↓↓**, nt_UUU **~0.035 flat**, nt_rew **228→243↑**, rollout −1→0.45↑, OOB=0 — same reward↑/hold≈0 class; **babysit to u60–80** (no mid-kill yet). Marker `.triple-b-h1-energy035-v1`.
  - C H1-var: ~**u259** nt_UUU **~0.058** ent **~0.85** OOB=0 — leave alone. Marker `.triple-c-h1var-walls-v1`.
- **This fire GO:** staged **ATRPO-lite** (`--avg-reward`: ρ-center + γ-free GAE) in `train_triple.py` + `next-train-triple.sh` + continue-b behind marker `.triple-b-h1-atrpo-v1` (**not armed** — live ENERGY_W keeps cooking). Left A/B/C alone.
- NEED_USER_PING: **yes**
