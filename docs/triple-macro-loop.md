# Triple pendulum — macro loop

Last updated: 2026-09-19 ~21:31 CT  
Owner: overnight routine (every 15m). Edit this file when the next micro-task changes.

## Overarching goal

Ship a cracked cart-**triple**-pendulum policy that can:

1. Swing up and **hold UUU** from hard starts (hang / near-target),
2. Stabilize / switch among all **8** discrete equilibria,
3. Eventually cover all **56** directed transitions for the interactive demo,

on the no-walls plant (`forceLimit` ≥ 40N), with TensorBoard + checkpoints mirroring to the user's local demo. Budget: **no hard $30 cap** — keep training until UUU works; prefer one L4 on-demand, don't stack GPU VMs.

## Macro phases (advance only when the gate clears)

| Phase | Gate to leave | What "done" looks like |
|---|---|---|
| **P0 — Plant + meters** | Physics/tests green; product reward + flip-augment + `eval/near_target` + `eval/hang` logging | Already mostly done on `main` |
| **P1 — UUU hold** | `eval/near_target/at_goal/UUU` ≳ **0.80** and align ≳ **0.90** for ≥1 stretch; entropy not collapsed | Current focus |
| **P2 — UUU swing** | `eval/hang/at_goal/UUU` ≳ **0.50** (then chase 0.70+) without killing hold | After P1 |
| **P3 — Multi-eq (8)** | min over 8 eqs `at_goal` ≳ **0.55**, align ≳ **0.75** | After P2 |
| **P4 — 56 transitions** | Directed A→B≠A curriculum; eval all 56 pairs | After P3 |
| **P5 — Demo polish** | Best ckpt → local + web triple demo | After a keeper exists |

## Current phase + next micro-task

- **Phase:** P1 — UUU hold (near-target meter is the truth; harsh `eval/*` stays secondary)
- **Next micro-task:** **build & smoke TQC UUU specialist (Lim 8×TQC path)** — `train/envs/triple_gym.py` + `train/train_triple_tqc.py` + `scripts/next-train-triple-tqc-uuu.sh` are on main; smoke passed on box. Next: scp `.venv-tqc` recipe to VM and **swap L4 slot C** onto TQC UUU (near_target + wide IC mix, product reward, Lim hypers). Do **not** mid-kill A/B PPO until C handoff is staged. Gate: TQC `eval/near_target/at_goal/UUU` ≳ 0.50 within ~300k steps, then promote.
- **Kill list:** no double/xonly on the L4; all 3 slots = triple-a/b/c
- **Do not:** mid-kill improving runs; rewrite plant without a paper-backed reason; stack a second GPU VM

## Ranked backlog (pull from top when a stretch ends)

1. **TQC UUU specialist (Lim)** — off-policy; swap slot C when smoke+scp ready (CURRENT). Hold recipe: BaRC/ladder widen near_target ICs (mastery-gated) + optional CSAC-QI ∫θ steady-state; keep product+progress as task
2. Two-policy handoff (TQC/energy swing → TQC/LQR hold) once UUU hold works — capture_tol ≤ measured ≤0.1 rad (never fawraw 0.3 default); latch+vel gate
3. Remaining 7 EP specialists (Lim 8×TQC) after UUU holds
4. Cool-ent / entropy floor on leftover PPO slots (A/B) until C swap — do not expand PPO
5. True E→E_UUU energy swing only if TQC hold still flat after full stretch
6. Force probe 40→60 only if OOB≈0 and plant feels underpowered


## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.

