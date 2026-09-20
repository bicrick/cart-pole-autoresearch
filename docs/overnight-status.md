# Triple overnight status — 2026-09-20 ~07:05 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~99%/16GB; TB http://34.148.138.48:6006/ HTTP 200
- **Roles:** A=S1 pid **153439**; B=H1 **RPO+ERA+gSDE ENT=0** pid **164727** (mid-kill @12:03Z from ENT=0 σ-collapse); C=H1-var pid **158583**. TRACK_WALLS=1. Watchers continue-a/b/c.
- **Meters:**
  - A S1: ~**u360**/400 hang_align/UUU **−0.037**, nt_UUU~0.056, ent **~1.22**, OOB=0 — leave alone. ETA ~0.3h.
  - B H1 prior ENT=0 **KILLED @u30**: ent 1.70→0.61↓↓ (faster than ENT=0.02), nt 0.039→0.033 flat, rollout_reward 0.36→0.45↑ (reward↑/hold≈0). ERA floor alone ≠ exploration.
  - B H1 **gSDE sf=4 ENT=0 LIVE** @u1: ent **1.59**, nt_UUU **0.045**, `--use-sde --sde-sample-freq 4`. Marker `.triple-b-h1-gsde-ent0-v1`. Commit `a7a764f`. ETA ~2.4h.
  - C H1-var: ~**u183** nt_UUU **~0.055** ent **~0.92** OOB=0 — leave alone. ETA ~1.3h.
- **This fire GO:** diagnosed B ENT=0 σ-death early; shipped Raffin/Zoo gSDE; mid-killed B into gSDE cold start. Left A/C alone.
- NEED_USER_PING: **yes**
