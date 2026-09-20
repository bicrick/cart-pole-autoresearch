# Triple overnight status — 2026-09-20 ~01:35 CT

- VM `cartpole-train-od` RUNNING us-east1-b L4 ~98%/10.5GB; TB http://34.148.138.48:6006/ HTTP 200
- **Commit:** `96695cd` on `main` (feat: training breakup — S1/H1/H1-var + init-noise + X1 smoke)
- **This fire (GO authorized):** Training breakup LIVE — roles flipped cold.
  - **A = S1** walls-swing — pid **145281** / watcher **145239** — run `ft-triple-a-…-walls-swing-s1-…` — `INIT_MODE=bottom` hang_p=0.85 near=0.10 warmup=0 TRACK_WALLS=1 ENT=0.05 LR=3e-4. Marker `.triple-a-s1-walls-v1`
  - **B = H1** walls-hold tight — pid **145275** / watcher **145240** — run `ft-triple-b-…-walls-hold-h1-…-in005-…` — near_goal_p=1 hang=0 warmup=0 **`INIT_NOISE=0.05`** TRACK_WALLS=1 ENT=0.02 LR=1e-4. Marker `.triple-b-h1-noise05-v1`
  - **C = H1-var** walls-hold — pid **145280** / watcher **145241** — run `ft-triple-c-…-walls-hold-h1var-…-in008-…` — near=1 hang=0 **`INIT_NOISE=0.08`** LR=5e-5 ENT=0.05 TRACK_WALLS=1. Marker `.triple-c-h1var-walls-v1` (second H1, not combo)
  - **X1:** `scripts/handoff_eval.py` + `scripts/eval-handoff-uuu.sh` staged (smoke when A/B ckpts exist)
  - PPO `--init-noise` wired through `next-train-triple.sh`
- Scripts md5-match box↔VM. No void TQC. No Mac.
- NEED_USER_PING: **no**
