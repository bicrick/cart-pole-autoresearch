# Triple pendulum — macro loop

Last updated: 2026-09-20 ~01:15 CT
Owner: overnight routine (every 15m). Edit this file when the next micro-task changes.

## Overarching goal

Ship a cracked cart-**triple**-pendulum policy that can:

1. Swing up and **hold UUU** from hard starts (hang / near-target),
2. Stabilize / switch among all **8** discrete equilibria,
3. Eventually cover all **56** directed transitions for the interactive demo,

on the no-walls plant (`forceLimit` ≥ 40N), with TensorBoard + checkpoints mirroring to the user's local demo. Budget: **no hard $30 cap** — keep training until UUU works; prefer one L4 on-demand, don't stack GPU VMs.

## Standing duty — self-debug (user 2026-09-20)

Patrick should not have to poke the demo to find our bugs. Every overnight/research fire must hunt failure modes and stage fixes:

- Reward↑ while hold/success≈0 → reward hacking / flopping
- Tiny |x| + thrashing angles → center / void-death farming (nowalls)
- Entropy collapse or saturation; weird OOB; dead catch basin
- Wrong plant phase (nowalls before walls-on worked — same lesson as double)

When found: name root cause, update Next micro-task, implement carefully, push, arm natural restart. Harden the checklist so that class of miss does not repeat.

## Macro phases (advance only when the gate clears)

| Phase | Gate to leave | What "done" looks like |
|---|---|---|
| **P0 — Plant + meters** | Physics/tests green; product reward + flip-augment + `eval/near_target` + `eval/hang` logging | Already mostly done on `main` |
| **P1a — Walls-on UUU** | Walls-on plant: `eval/near_target/at_goal/UUU` ≳ **0.80** and align ≳ **0.90** for ≥1 stretch; entropy not collapsed | **LIVE on A** (just cold-started) |
| **P1b — Nowalls FT** | Same meters on **void** plant (walls off), starting from P1a ckpt | After P1a gate |
| **P1 — UUU hold (overall)** | P1a then P1b both clear (void is the ship plant) | Current focus |
| **P2 — UUU swing** | `eval/hang/at_goal/UUU` ≳ **0.50** (then chase 0.70+) without killing hold | After P1 |
| **P3 — Multi-eq (8)** | min over 8 eqs `at_goal` ≳ **0.55**, align ≳ **0.75** | After P2 |
| **P4 — 56 transitions** | Directed A→B≠A curriculum; eval all 56 pairs | After P3 |
| **P5 — Demo polish** | Best ckpt → local + web triple demo | After a keeper exists |

## Current phase + next micro-task

- **Phase:** P1a — walls-on UUU (**A+C LIVE**; **B fixing to walls**)
- **Live (~01:15 CT):** VM `cartpole-train-od` RUNNING us-east1-b (L4 ~99%/13.5GB). TB http://34.148.138.48:6006/
  - **A** **P1a walls-on UUU LIVE** (pid 142589, ~u40/400) — `track_walls=True`; nt_UUU~**0.033** align_UUU~**+0.06** (climbing from −0.07); entropy~**1.84** healthy; **oob=0**. Early but align direction better than dead void stretch.
  - **B** was **void PPO-balance** (pid 142965, ~u14, `track_walls=False`) — **mis-arm**. TQC did exit (~06:07Z) after reward-hack (ep_rew~567 success≈0). Restarting B as **walls-on PPO-balance** (marker `.triple-b-ppo-balance-walls-v1`, `TRACK_WALLS=1`).
  - **C** **P1a walls-on combo LIVE** (pid 143053, ~u1+) — `track_walls=True`, ent0.08/lr5e-5, hang0.15; marker `.triple-c-walls-v1`. Prior void ended nt~0.049. Leave alone.
- **Diagnosis this fire:**
  1. **Wrong plant on B (self-caught):** continue-b launched PPO-balance with default void — violates P1a walls-first. Fixed script + cold walls restart (not babysitting void balance).
  2. **A walls early OK:** reward↑ but hold still ~0 (expected <u50); align climbing +; entropy not collapsed; oob=0 (walls). No center-farm signal.
  3. TQC reward-hack closed (natural exit done). Do not relaunch wide TQC.
- **Catch-basin:** TQC zip dead; LQR tiny RoA; PPO-balance walls-on is the live catcher path.
- **Watchers:** continue-a/b/c alive. B script patched for `TRACK_WALLS=1`.
- **Next micro-task:** (1) Babysit **A+C walls-on** — watch nt_UUU/align/entropy/oob through first 100u; no mid-kill. (2) Confirm **B walls-balance** cold-start healthy. (3) After P1a gate (nt≳0.80 align≳0.90) → P1b nowalls FT.
- **Kill list:** no double/xonly; slots = **A walls** / **B walls-balance** / **C walls-combo**
- **Do not:** mid-kill A/C; relaunch wide TQC; stack GPU; treat TQC zip as catcher; run B void during P1a
- **NEED_USER_PING:** **no** — first walls ping already owed/sent prior fire; this fire is ops fix (B walls) + early babysit. Ping on gate clear or new strategy flip.

## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (**DONE / exited**; basin **dead** @150k; reward-hack confirmed — do not relaunch). Hold mirror ladder leftovers (post-stretch only, **after** walls PPO-balance try): (a) M2 tighten `INIT_NOISE=0.05 HANG_FRAC=0 WIDE_FRAC=0`; (b) `ry_scale=1.0`; (c) **∫x obs**; (d) Baek VER flip; (e) optional fawraw M2 arch; (f) `n_steps=3`; (g) `use_sde=True sde_sample_freq=4`.
2. **Two-policy handoff** (**CODE STAGED + basin measured**) — catcher order: **skip TQC zip** → LQR only tiny → **PPO-balance walls-on LIVE on B**. Enter \(\|\phi_i\|<0.1\) & \(\|\omega\|_\infty<1\); latch + exit 0.25 + dwell; LPF τ≈0.3. Ops leftovers: (i) `handoff_eval.py` smoke capture_tol=**0.1**; (ii) PPO-balance `--init-noise` 0.05 + `--vel-cost-coef` 0.01–0.02; (iii) if thin → LQI ∫x; (iv) **P1a: PPO-balance `TRACK_WALLS=1`** (**DONE** this fire — void v1 killed, walls-v1 armed); (v) P1a flat-eval ladder if A stays flat (natural exit only): bar10→50 if rail-park, ENERGY_W 0.2→0.35 if visit≠hold, never mid-kill / hang / force60 mid-P1a.
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU hold actually works
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
