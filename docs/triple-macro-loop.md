# Triple pendulum — macro loop

Last updated: 2026-09-19 ~22:15 CT  
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
- **Live (~22:15 CT):** VM `cartpole-train-od` RUNNING us-east1-b (L4 ~99%/13.5–14.5GB, up ~43h). TB http://34.148.138.48:6006/
  - **A** PPO cool-ent v6 swing f50 — was crash-looping on `last_policy is None` TB log after all-NaN minibatch skip (4× NotImplementedError); fixed in `train_triple.py` this fire; continue-a will restart onto patched code. Prior healthy stretch peaked nt~0.06 @u400 then degraded; current restart early.
  - **B** PPO entboost v7b hold f40 ~**u360**/400 nt~**0.048** align_UUU~0.23 — still flat; continue-b TQC-staged since 02:37Z, waiting natural exit → Lim TQC UUU (**ETA ~22:38 CT**, ~40u×~34s)
  - **C** PPO entboost v7 combo f40 ~**u060** nt~**0.03** (cold-started 02:38Z) — leave alone
- **Next micro-task:** **do not touch B/C.** Leave B PPO undisturbed until exit; `continue-triple-b` launches TQC UUU (marker `.triple-b-tqc-uuu-v1`, venv `.venv-tqc`). Once TQC is live, track timesteps/`ep_rew_mean` + any UUU hold eval. If TQC flat after ~150k steps → two-policy handoff. A: patched NaN-skip TB crash; confirm A stays up after continue restart (do not mid-kill B/C).
- **Kill list:** no double/xonly on the L4; slots = A PPO / B→TQC / C PPO (until C stretch ends)
- **Do not:** mid-kill improving runs; more cool-ent / entboost PPO knobs; stack a second GPU VM; put TQC on C while B handoff is the plan
- **NEED_USER_PING:** no (B not yet on TQC; gate ≪0.80; A crash fixed in-flight, not a user decision)

## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (shipping now on slot B) — off-policy path that hit hardware
2. **Two-policy handoff** (stage only if TQC flat ~150k) — hold: TQC-UUU ckpt or LQR \(Q_\theta\sim100,R\sim0.01\) / PPO-balance near_target≤0.1 rad; swing: separate net (slot A / energy); enter: \(\|\phi_i\|<0.1\) rad **and** \(\|\omega\|_\infty<1\) rad/s (+ optional \(\bar c>0.9\) / E-gate), **never** tol=0.3; latch + hysteresis exit ~0.25 rad; swing soft-landing + LPF τ≈0.3 before handoff (fawraw/DiffSwing/ResearchSquare; research 21:56)
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU TQC proves hold
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
