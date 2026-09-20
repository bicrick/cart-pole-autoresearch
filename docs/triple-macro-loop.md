# Triple pendulum — macro loop

Last updated: 2026-09-20 ~00:34 CT  
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
- **Live (~00:28 CT):** VM `cartpole-train-od` RUNNING us-east1-b (L4 ~98%/11.4GB, up ~1d 21h). TB http://34.148.138.48:6006/
  - **A** PPO swing f50 ~**u340**/400 nt_at_goal/UUU~**0.033** align_UUU~−0.06 ent~0.20 — flat/worse; leave alone (no mid-kill). ETA stretch end ~10–12m.
  - **B** **TQC UUU live** (`train_triple_tqc.py` pid 133757) since 03:37Z — run `tqc-uuu-f40-hold-wide_2`; ~**221k**/300k steps (ckpts thru 200k + best@150k), `ep_rew_mean`~**568** (flat crawl), rollout success **0.01** (noise), ent_coef~0.0125; eval@200k mean_rew~**577** success **0**. ETA ~30–40m to 300k.
  - **C** PPO combo f40 ~**u319**/400 nt_at_goal/UUU~**0.048** align_UUU~0.19 ent~3.42 — leave alone. ETA ~25m.
- **Catch-basin (CPU, staged prior fire):**
  - TQC@150k: **0/9 dead** (`docs/basins/basin-tqc150k.json`) — do **not** use as hold catcher.
  - LQR soft Qθ=100 / stiff Qθ=1e3: only **0.1 rad @ ω=0** survives (`docs/basins/basin-lqr-*.json`). RoA too small for enter-gate 0.1 alone under velocity.
- **Code staged:** `train/lqr_uuu.py`, `train/handoff.py` (enter 0.1 / exit 0.25 / dwell / LPF τ=0.3), `scripts/measure_catch_basin.py`. `scripts/continue-triple-b.sh` on VM **md5-matches box** (PPO-balance on natural TQC exit; marker `.triple-b-tqc-uuu-v1` present).
- **Next micro-task:** (1) leave B TQC to ~300k (no mid-kill); watcher will auto-start **PPO-balance** catcher on natural exit (not another wide TQC). (2) On A exit keep swing specialist via continue-a. (3) On C exit keep combo. (4) Next free cycle after B→balance starts: wire `handoff_eval` smoke (swing-A + LQR/PPO-balance catcher) + widen LQR (∫x LQI / multi-link ICs) if PPO-balance also thin. If TQC eval success>0 before stretch end → abort handoff, stage EP specialist instead. (Rollout success 0.01 alone is **not** that signal.)
- **Kill list:** no double/xonly on the L4; slots = A PPO / **B TQC→PPO-balance** / C PPO
- **Do not:** mid-kill improving runs; more cool-ent / entboost PPO knobs; stack a second GPU VM; put TQC on C; relaunch parallel wide TQC on B
- **NEED_USER_PING:** **no** (quiet; gate ≪0.80; TQC still hold-flat; handoff path armed)

## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (**LIVE on slot B** → finishing stretch; basin **dead** @150k). Hold mirror ladder leftovers (post-stretch only, **after** PPO-balance try): (a) M2 tighten `INIT_NOISE=0.05 HANG_FRAC=0 WIDE_FRAC=0`; (b) `ry_scale=1.0`; (c) **∫x obs**; (d) Baek VER flip; (e) optional fawraw M2 arch; (f) `n_steps=3`; (g) `use_sde=True sde_sample_freq=4`.
2. **Two-policy handoff** (**CODE STAGED + basin measured**) — catcher order data-backed: **skip TQC zip** → LQR only tiny → **PPO-balance next on B exit**. Enter \(\|\phi_i\|<0.1\) & \(\|\omega\|_\infty<1\) (+ opt \(|\tilde E|<\varepsilon\)); latch + exit 0.25 + dwell; LPF τ≈0.3. Modules: `train/handoff.py`, `train/lqr_uuu.py`, `scripts/measure_catch_basin.py`. **Ops leftovers (research 00:34):** (i) wire `scripts/handoff_eval.py` smoke — capture_tol=**0.1** (not fawraw 0.35), pin reset options, report handoff@/reached/held/cartx/stab%; (ii) PPO-balance cold-start: add `--init-noise` to `train_triple.py` (gym default 0.15 is a no-op for shell `INIT_NOISE`), start at **0.05** then BaRC-expand; set `--vel-cost-coef` **0.01–0.02** (SARS −ẋ²); (iii) if balance thin → LQI ∫x with Q_ξ≈**0.1** + per-link basin grid (ResearchSquare RoA ~5° confirms 0.1@ω=0).
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU hold actually works
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
