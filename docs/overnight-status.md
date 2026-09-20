# Triple overnight status — 2026-09-20 ~05:40 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~99%/16.2GB; TB http://34.148.138.48:6006/ HTTP 200; uptime ~2d 2h
- **Roles:** A=S1 pid **153439** (LR=1e-4); B=H1 pid **158009** (**ENT=0.035 + VEL_COST=0.015** @u45); C=H1-var pid **158583** (natural restart @10:22Z, ENT=0.05 + VEL_COST=0.015 @u32). TRACK_WALLS=1 all. Watchers continue-a/b/c armed.
- **Meters:**
  - A S1: ~**u210** hang_align/UUU **−0.97→−0.016** (recovered from −0.037@u170), hang_at_goal~0.006, nt~0.053 nt_align~0.239, ent **~1.54**, policy_loss **~−0.01** (no explode), OOB=0 — **past prior NaN crash zone u150–228, still clean**.
  - B H1 ENT035: ~**u45** ent **−0.80→−1.055↓** (σ still diving from dead-σ ckpt); nt_UUU **~0.050 flat**; nt_align~0.16; reward~171 flat; OOB=0. **Babysit to ~u80** before RPO/ERA call.
  - C H1-var: ~**u32** nt_UUU **~0.049** nt_align~0.145 ent **~1.10** reward~74 OOB=0 (ENT=0.05 + VEL_COST). Fresh stretch — leave alone.
- **This fire GO:** left A/B/C alone (A surviving NaN window; B still inside ENT035 babysit; C early healthy). No mid-kill. No code change.
- NEED_USER_PING: **yes** (standing think-out-loud every fire)
