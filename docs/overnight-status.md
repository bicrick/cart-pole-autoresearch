# Triple overnight status — 2026-09-20 ~05:51 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~99%/16.6GB; TB http://34.148.138.48:6006/ HTTP 200; uptime ~2d 3h
- **Roles:** A=S1 pid **153439** (LR=1e-4); B=H1 pid **158009** (**ENT=0.035 + VEL_COST=0.015** @~u62–65); C=H1-var pid **158583** (ENT=0.05 + VEL_COST @~u50). TRACK_WALLS=1 all. Watchers continue-a/b/c.
- **Meters:**
  - A S1: ~**u228** hang_align/UUU **−0.97→−0.027** (wobble −0.009@u150…−0.036@u210…−0.027@u220), hang_at_goal~0.003, nt~0.057 nt_align~0.23, ent **~1.49**, policy_loss **~0.01** (no explode), OOB=0 — **past prior NaN crash point u228, still clean**.
  - B H1 ENT035: ~**u62** ent **−0.80→−1.23↓↓** (still diving ~0.007/u); nt_UUU **~0.052 flat**; nt_align~0.19; reward~168 flat; OOB=0. Babysit window →~u80 still open — **no mid-kill**.
  - C H1-var: ~**u50** nt_UUU **~0.049** nt_align~0.16 ent **~1.08** reward~74 OOB=0. Leave alone.
- **This fire GO:** left A/B/C live alone. **Staged** next H1 recipe: CleanRL **RPO α=0.01** + ERA soft **H₀=0.5** log_std floor (ENT=0.02, keep VEL_COST) in `ppo.py` / `train_triple` / `continue-triple-b` (marker `.triple-b-h1-rpo01-era05-v1`). Rearmed B watcher only so natural exit (or post-u80 call) picks RPO+ERA up — **did not mid-kill B**.
- NEED_USER_PING: **yes** (standing think-out-loud every fire)
