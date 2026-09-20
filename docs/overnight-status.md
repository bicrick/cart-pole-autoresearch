# Triple overnight status — 2026-09-20 ~01:44 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~99%/15.6GB; TB http://34.148.138.48:6006/ HTTP 200
- **Roles confirmed:** A=S1 pid **145281**/w **145239**; B=H1 pid **145275**/w **145240**; C=H1-var pid **145280**/w **145241**. Markers `.triple-a-s1-walls-v1` / `.triple-b-h1-noise05-v1` / `.triple-c-h1var-walls-v1`. TRACK_WALLS=1 all.
- **Meters (~u15–18, ~10min post cold @06:33Z / 01:33 CT):**
  - A S1: reward~222 hang_align/UUU -0.98→-0.80 nt_UUU~0.035 OOB=0 ent~1.69↑ — healthy early swing
  - B H1: reward~225 nt_UUU~0.034 nt_align/UUU~-0.03 OOB=0 ent 1.45→1.12 (watch) — healthy early hold; **no mid-kill**
  - C H1-var: reward~221 nt_UUU~0.033 OOB=0 ent~1.74 — healthy early
- **This fire GO:** Wired `VEL_COST_COEF` through `next-train-triple.sh`; set 0.015 on continue-b/c for **next natural restart** (soft-land backlog). Live PIDs undisturbed. Handoff smoke deferred (ckpts cold/~u15).
- Reward↑/hold≈0 = cold product reward, not yet hacking. No void TQC. No Mac.
- NEED_USER_PING: **no**
