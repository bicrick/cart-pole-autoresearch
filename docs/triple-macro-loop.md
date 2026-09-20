# Triple pendulum — macro loop

Last updated: 2026-09-20 ~00:53 CT
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

- **Phase:** P1a — walls-on UUU (**A LIVE** as of ~00:53 CT)
- **Live (~00:53 CT):** VM `cartpole-train-od` RUNNING us-east1-b (L4 ~97%/9.4GB, up ~1d 22h). TB http://34.148.138.48:6006/
  - **A** **P1a walls-on UUU LIVE** (pid 142589, cold-start 05:53Z) — `track_walls=True`, product/near_target, hang=0, f40, ent0.05/lr1e-4; run `…uuu-walls-hold-f40…`; marker `.triple-a-walls-v1` set. Prior void swing ended u400 nt_at_goal/UUU~**0.039** align~−0.05 (dead). No scalars yet (just started).
  - **B** **TQC UUU** (pid 133757) — ~**273k**/300k; `ep_rew_mean`~**567** flat; eval@250k mean_rew~**566** success **0**; rollout success **0.01**. **Diagnosis: reward hacking / flopping** (rew↑ hold≈0). ETA ~10–15m → PPO-balance (watcher armed).
  - **C** PPO combo f40 ~**u367**/400 nt_at_goal/UUU~**0.056** align_UUU~**0.24**; entropy~**3.42** healthy; oob≈0. Flat hold — leave alone. ETA ~10m → walls-on cold-start (`.triple-c-walls-v1` not yet; watcher waiting).
- **Diagnosis this fire:**
  1. **Wrong plant phase (confirmed):** void A/C stretches stayed flat on UUU hold (A final 0.039 / C 0.056) — same double lesson; **walls-on now live on A**.
  2. **TQC reward hack (ongoing):** B ep_rew~567 with success≈0 — do **not** extend TQC; natural exit → PPO-balance catcher.
  3. No center-farming signal yet on walls plant (too early); watch |x|/oob once A has evals.
- **Catch-basin (CPU, staged prior):** TQC@150k **0/9 dead**; LQR only tiny RoA — skip TQC zip as catcher; PPO-balance next on B.
- **Watchers:** continue-a/b/c alive (a/c @05:45Z, b @05:17Z). A already flipped to walls. C→walls on natural exit. B→PPO-balance on TQC exit.
- **Next micro-task:** (1) **Babysit A walls-on** — first eval window (~u10–20); watch nt_UUU / align / entropy / oob; do not mid-kill. (2) C natural exit → walls-on. (3) B TQC finish → PPO-balance. (4) After P1a gate (nt≳0.80 align≳0.90) → P1b nowalls FT.
- **Kill list:** no double/xonly; slots = **A walls LIVE** / **B TQC→PPO-balance** / **C walls (next)**
- **Do not:** mid-kill A/B/C; more cool-ent PPO knobs; stack GPU; relaunch wide TQC on B; treat TQC zip as hold catcher
- **NEED_USER_PING:** **YES** — first walls-on live (A cold-start ~00:53 CT). Gate still ≪0.80; no further ping unless gate clears or strategy flip.

## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (**LIVE on slot B** → finishing; basin **dead** @150k; reward-hack confirmed). Hold mirror ladder leftovers (post-stretch only, **after** PPO-balance try): (a) M2 tighten `INIT_NOISE=0.05 HANG_FRAC=0 WIDE_FRAC=0`; (b) `ry_scale=1.0`; (c) **∫x obs**; (d) Baek VER flip; (e) optional fawraw M2 arch; (f) `n_steps=3`; (g) `use_sde=True sde_sample_freq=4`.
2. **Two-policy handoff** (**CODE STAGED + basin measured**) — catcher order: **skip TQC zip** → LQR only tiny → **PPO-balance next on B exit**. Enter \(\|\phi_i\|<0.1\) & \(\|\omega\|_\infty<1\); latch + exit 0.25 + dwell; LPF τ≈0.3. Ops leftovers: (i) `handoff_eval.py` smoke capture_tol=**0.1**; (ii) PPO-balance `--init-noise` 0.05 + `--vel-cost-coef` 0.01–0.02; (iii) if thin → LQI ∫x.
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU hold actually works
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
