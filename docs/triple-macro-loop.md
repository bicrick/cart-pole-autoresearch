# Triple pendulum — macro loop

Last updated: 2026-09-20 ~00:15 CT  
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
- **Live (~00:15 CT):** VM `cartpole-train-od` RUNNING us-east1-b (L4 ~88%/11.4GB). TB http://34.148.138.48:6006/
  - **A** PPO swing f50 ~**u300**/400 nt_at_goal/UUU~**0.038** align_UUU~−0.07 — flat; leave alone (no mid-kill). ETA stretch end ~15–20m.
  - **B** **TQC UUU live** (`train_triple_tqc.py`) since 03:37Z — run `tqc-uuu-f40-hold-wide_2`; ~**189k**/300k steps (ckpts @175k + best@150k), `ep_rew_mean`~**561** (flat crawl), success_rate **0**, ent_coef~0.0125; eval@175k mean_rew~**551** (dip from 582@150k) success **0**.
  - **C** PPO combo f40 ~**u289**/400 nt_at_goal/UUU~**0.056** align_UUU~0.23 ent~3.42 — leave alone.
- **Catch-basin (CPU, staged this fire):**
  - TQC@150k: **0/9 dead** (`docs/basins/basin-tqc150k.json`) — do **not** use as hold catcher.
  - LQR soft Qθ=100 / stiff Qθ=1e3: only **0.1 rad @ ω=0** survives (`docs/basins/basin-lqr-*.json`). RoA too small for enter-gate 0.1 alone under velocity.
- **Code staged:** `train/lqr_uuu.py`, `train/handoff.py` (enter 0.1 / exit 0.25 / dwell / LPF τ=0.3), `scripts/measure_catch_basin.py`.
- **Next micro-task:** (1) leave B TQC to ~300k (no mid-kill); `continue-triple-b` now auto-starts **PPO-balance** catcher on natural exit (not another wide TQC). (2) On A exit keep swing specialist. (3) Next free cycle: wire `handoff_eval` smoke (swing-A + LQR/PPO-balance catcher) + widen LQR (∫x LQI / multi-link ICs) if PPO-balance also thin. If TQC success>0 before stretch end → abort handoff, stage EP specialist instead.
- **Kill list:** no double/xonly on the L4; slots = A PPO / **B TQC→PPO-balance** / C PPO
- **Do not:** mid-kill improving runs; more cool-ent / entboost PPO knobs; stack a second GPU VM; put TQC on C; relaunch parallel wide TQC on B
- **NEED_USER_PING:** **no** (quiet; gate ≪0.80; TQC basin dead confirms handoff path — no kill)

## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (**LIVE on slot B** → finishing stretch; basin **dead** @150k). Hold mirror ladder leftovers (post-stretch only, **after** PPO-balance try): (a) M2 tighten `INIT_NOISE=0.05 HANG_FRAC=0 WIDE_FRAC=0`; (b) `ry_scale=1.0`; (c) **∫x obs**; (d) Baek VER flip; (e) optional fawraw M2 arch; (f) `n_steps=3`; (g) `use_sde=True sde_sample_freq=4`.
2. **Two-policy handoff** (**CODE STAGED + basin measured @00:15 CT**) — catcher order now data-backed: **skip TQC zip** → LQR only tiny → **PPO-balance next on B exit**. Enter \(\|\phi_i\|<0.1\) & \(\|\omega\|_\infty<1\); latch + exit 0.25 + dwell; LPF τ≈0.3. Modules: `train/handoff.py`, `train/lqr_uuu.py`, `scripts/measure_catch_basin.py`.
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU hold actually works
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
