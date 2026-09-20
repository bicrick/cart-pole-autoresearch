# Triple overnight status — 2026-09-20 ~01:35 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4; TB http://34.148.138.48:6006/
- **This fire (GO authorized):** Training breakup executed — roles flipped to true H1/S1.
  - **A = S1** walls-swing (hang/bottom, `INIT_MODE=bottom`, hang_p=0.85, progress on)
  - **B = H1** walls-hold tight (`NEAR_GOAL_P=1`, `INIT_NOISE=0.05`, warmup 0)
  - **C = H1-var** walls-hold (lr=5e-5 ent=0.05 noise=0.08) — second H1, not combo mush
  - **X1:** `scripts/handoff_eval.py` + `scripts/eval-handoff-uuu.sh` staged (smoke when A/B ckpts exist)
  - Added PPO `--init-noise` (wired through `next-train-triple.sh`)
- Cold markers: `.triple-a-s1-walls-v1` / `.triple-b-h1-noise05-v1` / `.triple-c-h1var-walls-v1`
- Kill list: no void TQC relaunch; no Mac; prefer correct roles over early ~u50 flat walls runs
- NEED_USER_PING: **no** (user said Go — executing)
