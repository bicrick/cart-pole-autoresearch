# Triple pendulum — macro loop

Last updated: 2026-09-19 ~22:30 CT  
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
- **Live (~22:30 CT):** VM `cartpole-train-od` RUNNING us-east1-b (L4 ~99%/15.6GB, up ~45h). TB http://34.148.138.48:6006/
  - **A** PPO cool-ent v6 swing f50 ~**u030**/400 nt~**0.036** — NaN TB crash fix confirmed on VM (`if last_policy is not None`); continue-a restarted onto patch @03:15Z; past u1 crash zone, leave alone
  - **B** PPO entboost v7b hold f40 ~**u380**/400 nt~**0.050** align_UUU~0.24 — still flat; continue-b TQC-staged (waiting pid 119552) → Lim TQC UUU (**ETA ~22:41 CT**, ~20u×~33s). `.venv-tqc` + `train_triple_tqc.py` ready (sb3_contrib 2.9.0)
  - **C** PPO entboost v7 combo f40 ~**u090** nt~**0.04** — leave alone
- **Next micro-task:** **do not touch B/C.** Wait for B natural exit; `continue-triple-b` launches TQC UUU (marker `.triple-b-tqc-uuu-v1` on first start). Next fire: confirm B is `train_triple_tqc` and meter timesteps/`ep_rew_mean`. If TQC flat after ~150k → two-policy handoff. No mid-kill A/C.
- **Kill list:** no double/xonly on the L4; slots = A PPO / B→TQC / C PPO (until C stretch ends)
- **Do not:** mid-kill improving runs; more cool-ent / entboost PPO knobs; stack a second GPU VM; put TQC on C while B handoff is the plan; start `continue-triple-tqc-uuu` in parallel with continue-b
- **NEED_USER_PING:** no (B not yet on TQC; gate ≪0.80; A patch verified healthy)

## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (shipping now on slot B) — off-policy path that hit hardware
2. **Two-policy handoff** (stage only if TQC flat ~150k) — hold: TQC-UUU ckpt or LQR \(Q_\theta\sim100,R\sim0.01\) / PPO-balance near_target≤0.1 rad; swing: separate net (slot A / energy); enter: \(\|\phi_i\|<0.1\) rad **and** \(\|\omega\|_\infty<1\) rad/s (+ optional \(\bar c>0.9\) / E-gate), **never** tol=0.3; latch + hysteresis exit ~0.25 rad; swing soft-landing + LPF τ≈0.3 before handoff (fawraw/DiffSwing/ResearchSquare; research 21:56)
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU TQC proves hold
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
