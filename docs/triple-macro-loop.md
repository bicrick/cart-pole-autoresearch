# Triple pendulum — macro loop

Last updated: 2026-09-20 ~00:45 CT
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
| **P1a — Walls-on UUU** | Walls-on plant: `eval/near_target/at_goal/UUU` ≳ **0.80** and align ≳ **0.90** for ≥1 stretch; entropy not collapsed | **Priority new experiment** (staged on A/C next natural restart) |
| **P1b — Nowalls FT** | Same meters on **void** plant (walls off), starting from P1a ckpt | After P1a gate |
| **P1 — UUU hold (overall)** | P1a then P1b both clear (void is the ship plant) | Current focus |
| **P2 — UUU swing** | `eval/hang/at_goal/UUU` ≳ **0.50** (then chase 0.70+) without killing hold | After P1 |
| **P3 — Multi-eq (8)** | min over 8 eqs `at_goal` ≳ **0.55**, align ≳ **0.75** | After P2 |
| **P4 — 56 transitions** | Directed A→B≠A curriculum; eval all 56 pairs | After P3 |
| **P5 — Demo polish** | Best ckpt → local + web triple demo | After a keeper exists |

## Current phase + next micro-task

- **Phase:** P1a — walls-on UUU (priority); void nowalls jobs still finishing their stretches (no mid-kill)
- **Live (~00:45 CT):** VM `cartpole-train-od` RUNNING us-east1-b (L4 ~99%/11.4GB, up ~1d 22h). TB http://34.148.138.48:6006/
  - **A** PPO swing f50 ~**u380**/400 nt_at_goal/UUU~**0.031** align_UUU~−0.07 — flat/dead; leave alone. ETA stretch end **~5m** → walls-on cold-start via rearmed watcher.
  - **B** **TQC UUU live** (pid 133757) — ~**253k**/300k; `ep_rew_mean`~**567** flat; eval@250k mean_rew~**566** success **0**; rollout success **0.01**. ETA ~20–25m → PPO-balance.
  - **C** PPO combo f40 ~**u340**/400 nt_at_goal/UUU~**0.054** align_UUU~0.23 — flat; leave alone. ETA ~20m → walls-on cold-start.
- **Catch-basin (CPU, staged prior fire):**
  - TQC@150k: **0/9 dead** (`docs/basins/basin-tqc150k.json`) — do **not** use as hold catcher.
  - LQR soft Qθ=100 / stiff Qθ=1e3: only **0.1 rad @ ω=0** survives (`docs/basins/basin-lqr-*.json`). RoA too small for enter-gate 0.1 alone under velocity.
- **Code staged + VM synced (00:44 CT):** walls physics (`trackWalls` clamp x, xd=0) + `continue-triple-a/c.sh` + `next-train-triple-walls.sh` md5-match box↔VM. **A/C continue watchers rearmed** (pids fresh @05:44Z) waiting on live trainers — next natural exit → P1a walls-on UUU cold-start (markers `.triple-a-walls-v1` / `.triple-c-walls-v1`). B watcher unchanged (TQC→PPO-balance; `.triple-b-tqc-uuu-v1` present). Handoff modules still staged.
- **Next micro-task:** (1) leave B TQC to ~300k (no mid-kill) → PPO-balance. (2) **A then C** natural exit → **P1a walls-on** (already armed). (3) On first walls-on live → ping user. (4) After P1a gate → P1b nowalls FT. Do not mid-kill current A/C/B.
- **Kill list:** no double/xonly on the L4; slots = **A walls-on (next)** / **B TQC→PPO-balance** / **C walls-on (next)**
- **Do not:** mid-kill live A/C/B; more cool-ent / entboost PPO knobs; stack a second GPU VM; put TQC on C; relaunch parallel wide TQC on B
- **NEED_USER_PING:** **no** yet (walls armed but not live; gate ≪0.80; TQC eval success still 0). Ping on walls-on first live after natural start.

## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (**LIVE on slot B** → finishing stretch; basin **dead** @150k). Hold mirror ladder leftovers (post-stretch only, **after** PPO-balance try): (a) M2 tighten `INIT_NOISE=0.05 HANG_FRAC=0 WIDE_FRAC=0`; (b) `ry_scale=1.0`; (c) **∫x obs**; (d) Baek VER flip; (e) optional fawraw M2 arch; (f) `n_steps=3`; (g) `use_sde=True sde_sample_freq=4`.
2. **Two-policy handoff** (**CODE STAGED + basin measured**) — catcher order data-backed: **skip TQC zip** → LQR only tiny → **PPO-balance next on B exit**. Enter \(\|\phi_i\|<0.1\) & \(\|\omega\|_\infty<1\) (+ opt \(|\tilde E|<\varepsilon\)); latch + exit 0.25 + dwell; LPF τ≈0.3. Modules: `train/handoff.py`, `train/lqr_uuu.py`, `scripts/measure_catch_basin.py`. **Ops leftovers (research 00:34):** (i) wire `scripts/handoff_eval.py` smoke — capture_tol=**0.1** (not fawraw 0.35), pin reset options, report handoff@/reached/held/cartx/stab%; (ii) PPO-balance cold-start: add `--init-noise` to `train_triple.py` (gym default 0.15 is a no-op for shell `INIT_NOISE`), start at **0.05** then BaRC-expand; set `--vel-cost-coef` **0.01–0.02** (SARS −ẋ²); (iii) if balance thin → LQI ∫x with Q_ξ≈**0.1** + per-link basin grid (ResearchSquare RoA ~5° confirms 0.1@ω=0).
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU hold actually works
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
