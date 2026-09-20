# Triple pendulum — macro loop

Last updated: 2026-09-19 ~23:57 CT  
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
- **Live (~23:57 CT):** VM `cartpole-train-od` RUNNING us-east1-b (L4 ~97%/11.4GB). TB http://34.148.138.48:6006/
  - **A** PPO swing f50 ~**u260**/400 nt_at_goal/UUU~**0.032** align_UUU~−0.054 — still flat on hold; leave alone (no mid-kill)
  - **B** **TQC UUU live** (`train_triple_tqc.py` pid 133757) since **03:37Z / ~22:37 CT** — run `tqc-uuu-f40-hold-wide_2`; ~**157.5k**/300k steps (ckpt @**150k** + best_model), `ep_rew_mean` −258→541→**+555** (marginal +14 since 125k; crawl not hold), success_rate **0**, ent_coef~0.013; eval@150k mean_rew~**582** (534→517→543→563→569→582) success **0**. **≥150k + success 0 + no hold lift** → stage handoff (do not mid-kill; let stretch run toward 300k)
  - **C** PPO combo f40 ~**u260**/400 nt_at_goal/UUU~**0.050** align_UUU~0.22 ent~3.4 — leave alone
- **Next micro-task:** **stage two-policy handoff** (do **not** mid-kill TQC — leave B running to stretch end ~300k). Prep hold catcher (TQC-UUU ckpt @150k / LQR \(Q_\theta\sim100,R\sim0.01\) / PPO-balance near_target≤0.1 rad) + swing net (slot A) + enter gate \(\|\phi_i\|<0.1\) & \(\|\omega\|_\infty<1\) + latch/hysteresis ~0.25 + soft-landing LPF. Leave A/C alone until their stretches end. If TQC suddenly shows success>0 before stretch end → abort handoff staging and stage EP specialist instead. No mid-kill.
- **Kill list:** no double/xonly on the L4; slots = A PPO / **B TQC** / C PPO (until C stretch ends)
- **Do not:** mid-kill improving runs; more cool-ent / entboost PPO knobs; stack a second GPU VM; put TQC on C while B is the specialist slot; relaunch parallel `continue-triple-tqc-uuu` (B already owns TQC)
- **NEED_USER_PING:** **no** (quiet; gate still ≪0.80; TQC past 150k flat on hold — handoff staged in-doc only, no kill)

## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (**LIVE on slot B** since 22:37 CT; **≥150k / success 0 @23:57 CT** — hold dead for this stretch; do not mid-kill, handoff staged). Hold mirror ladder leftovers (post-stretch only): (a) M2 tighten `INIT_NOISE=0.05 HANG_FRAC=0 WIDE_FRAC=0`; (b) `ry_scale=1.0` Lim cart term; (c) **∫x obs** (Lim x₉ / LQI) + curriculum `eval/near_target/*` meters on TQC; (d) Baek VER replay flip (TQC-native); (e) optional fawraw M2 arch `[128,128]` / buffer 200k / 150k; (f) **`n_steps=3`** (SB3 NStepReplay / Raffin) after tighten; (g) **`use_sde=True sde_sample_freq=4`** (Zoo PyBullet TQC / Raffin) after n_steps, before M2-arch shrink / dead call (research 23:27). After nt mastery: BaRC **expand-with-ω** (nonzero link vel + off-centre x), not angle-noise alone (fawraw M4 basin)
2. **Two-policy handoff** (**STAGED @23:57 CT** — TQC ≥150k flat on hold; implement when B stretch ends / next free slot) — hold: TQC-UUU ckpt or LQR \(Q_\theta\sim100,R\sim0.01\) (or IC_ASET 2025 PI/VI-in-LQR catcher) / PPO-balance near_target≤0.1 rad; swing: separate net (slot A / energy); enter: \(\|\phi_i\|<0.1\) rad **and** \(\|\omega\|_\infty<1\) rad/s (+ optional \(\bar c>0.9\) / E-gate), **never** tol=0.3; latch + hysteresis exit ~0.25 rad; swing soft-landing + LPF τ≈0.3 before handoff (fawraw/DiffSwing/ResearchSquare; research 21:56)
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU TQC proves hold
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
