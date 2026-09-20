# Triple pendulum — macro loop

Last updated: 2026-09-19 ~23:01 CT  
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
- **Live (~23:01 CT):** VM `cartpole-train-od` RUNNING us-east1-b (L4 ~98%/10.4GB, up ~44h). TB http://34.148.138.48:6006/
  - **A** PPO cool-ent v6 swing f50 ~**u110**/400 nt_at_goal/UUU~**0.034** align_UUU~−0.08 — entropy TB stalled after u10 (~0.20); leave alone (no mid-kill)
  - **B** **TQC UUU live** (`train_triple_tqc.py` pid 133757) since **03:37Z / ~22:37 CT** — marker `.triple-b-tqc-uuu-v1`; run `tqc-uuu-f40-hold-wide_2`; ~**45.5k**/300k steps, `ep_rew_mean` −182→**+247** (climbing), success_rate 0, fps~33, ent_coef~0.014; eval@25k mean_rew~535 success 0. **Not flat** — keep metering to ~150k
  - **C** PPO entboost v7 combo f40 ~**u150**/400 nt~**0.043** align_UUU~0.18 ent~3.42 — leave alone
- **Next micro-task:** **meter TQC only.** Track B timesteps / `ep_rew_mean` / success / next eval. Leave A/C alone. If TQC still flat after ~**150k** steps → stage two-policy handoff (do not spam entropy knobs). If real UUU hold lift (success or hold eval) → stage next EP specialist. No mid-kill.
- **Kill list:** no double/xonly on the L4; slots = A PPO / **B TQC** / C PPO (until C stretch ends)
- **Do not:** mid-kill improving runs; more cool-ent / entboost PPO knobs; stack a second GPU VM; put TQC on C while B is the specialist slot; relaunch parallel `continue-triple-tqc-uuu` (B already owns TQC)
- **NEED_USER_PING:** **no** (B-on-TQC already reported ~22:40; gate still ≪0.80; TQC reward climbing)

## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (**LIVE on slot B** since 22:37 CT) — meter to ~150k before declaring dead. Hold mirror ladder if early-flat (do not mid-kill): (a) M2 tighten `INIT_NOISE=0.05 HANG_FRAC=0 WIDE_FRAC=0`; (b) `ry_scale=1.0` Lim cart term; (c) **∫x obs** (Lim x₉ / LQI) + curriculum `eval/near_target/*` meters on TQC; (d) Baek VER replay flip (TQC-native); (e) optional fawraw M2 arch `[128,128]` / buffer 200k / 150k; (f) **`n_steps=3`** (SB3 NStepReplay / Raffin) after tighten, before M2-arch shrink / dead call (research 23:01)
2. **Two-policy handoff** (stage only if TQC flat ~150k) — hold: TQC-UUU ckpt or LQR \(Q_\theta\sim100,R\sim0.01\) (or IC_ASET 2025 PI/VI-in-LQR catcher) / PPO-balance near_target≤0.1 rad; swing: separate net (slot A / energy); enter: \(\|\phi_i\|<0.1\) rad **and** \(\|\omega\|_\infty<1\) rad/s (+ optional \(\bar c>0.9\) / E-gate), **never** tol=0.3; latch + hysteresis exit ~0.25 rad; swing soft-landing + LPF τ≈0.3 before handoff (fawraw/DiffSwing/ResearchSquare; research 21:56)
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU TQC proves hold
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
