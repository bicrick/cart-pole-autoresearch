# Triple overnight status — 2026-09-20 ~05:18 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~99%/14.5GB; TB http://34.148.138.48:6006/ HTTP 200; uptime ~2d
- **Roles:** A=S1 pid **153439** (LR=1e-4); B=H1 pid **158009** (**ENT=0.035 + VEL_COST=0.015 just restarted** @10:16Z from u400 ckpt); C=H1-var pid **145280**. TRACK_WALLS=1 all. Watchers continue-a/b/c armed.
- **Meters:**
  - A S1: ~**u164** hang_align/UUU **−0.97→−0.018** (was −0.009@u150; slight wobble), hang_at_goal~0.005, nt~0.042 nt_align~0.225, ent **~1.71**, policy_loss **−0.008** (recovered from +0.028@u155), OOB=0 — **inside prior NaN window, looking OK**.
  - B H1: prior stretch ended **u400** nt_UUU **~0.050** nt_align **~0.17** ent **≈−0.79↓↓** reward~0.50 (dead σ + visit≠hold). **Natural restart LIVE** run `…-ent035-…` pid 158009 @u1 (loaded u400 ckpt; σ still dead at start — expected).
  - C H1-var: ~**u395** nt_UUU **~0.051** nt_align **~0.171** ent **~1.16** — ENT=0.05 still null for hold; leave alone (~5 upd to stretch end).
- **This fire GO:** babysat B stretch end → confirmed ENT=0.035 + VEL_COST=0.015 + walls continue landed (no mid-kill). Left A/C alone. No code change.
- NEED_USER_PING: **yes** (standing think-out-loud every fire)
