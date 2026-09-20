# Triple pendulum — macro loop

Last updated: 2026-09-20 ~01:28 CT
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


## Training breakup (how we actually ship UUU)

Do **not** train one mega-policy to swing + hold + recover. Split by **role**, plant phase, then compose.

| Stage | Who trains | Plant | ICs / recipe | Gate to leave | Slot bias |
|---|---|---|---|---|---|
| **H1 — Hold specialist** | Balance-only policy (PPO or tight TQC) | **Walls on** | Near-upright only (`near_goal_p≈1`, noise≲0.05–0.15, **no hang**). Product reward, center OK. | `near_target/at_goal/UUU` ≳ **0.80**, align ≳ **0.90** | **B** (live) |
| **S1 — Swing specialist** | Swing-only policy | **Walls on** | Hang / bottom heavy; progress reward; deliver toward UUU (align↑), not required to hold forever | Hang align climbing; can reach enter-gate often | **A** |
| **X1 — Handoff** | Glue S1→H1 | Walls on | Enter \|φ\|≲0.1 & ω small; latch; exit 0.25; dwell; optional LPF. Eval `handoff_eval` smoke | Hang→hold success rate useful (e.g. ≳0.3 then chase) | CPU eval + one GPU slot when ready |
| **H2 / S2 — Void FT** | Fine-tune H1 then S1+X1 from walls keepers | **Walls off** | Same roles; hard oob_penalty; no center-only farming | Same meters on void plant | After H1 (hold first!) |
| **E8 — 8 specialists** | One policy per EP (Lim) | Walls then void | Clone H1 recipe per eq | Each EP hold gate | After UUU path works |
| **T56 — Transitions** | Directed A→B or shared conditional | Void | Only after E8 / solid multi-eq | 56-pair eval | Last |

**Live mapping (2026-09-20):** B = H1 walls-balance; A = walls-hold (treat as H1 sibling until hold gate, then retarget to S1); C = walls-combo exploration (kill/repurpose if it stays flat past ~u150). Do not relaunch wide void TQC.

**Compose rule:** never void-FT a policy that cannot hold on walls. Never ask swing to also be the catcher.

## Macro phases (advance only when the gate clears)

| Phase | Gate to leave | What "done" looks like |
|---|---|---|
| **P0 — Plant + meters** | Physics/tests green; product reward + flip-augment + `eval/near_target` + `eval/hang` logging | Already mostly done on `main` |
| **P1a — Walls-on UUU** | Walls-on plant: `eval/near_target/at_goal/UUU` ≳ **0.80** and align ≳ **0.90** for ≥1 stretch; entropy not collapsed | **LIVE on A+B+C** (walls confirmed) |
| **P1b — Nowalls FT** | Same meters on **void** plant (walls off), starting from P1a ckpt | After P1a gate |
| **P1 — UUU hold (overall)** | P1a then P1b both clear (void is the ship plant) | Current focus |
| **P2 — UUU swing** | `eval/hang/at_goal/UUU` ≳ **0.50** (then chase 0.70+) without killing hold | After P1 |
| **P3 — Multi-eq (8)** | min over 8 eqs `at_goal` ≳ **0.55**, align ≳ **0.75** | After P2 |
| **P4 — 56 transitions** | Directed A→B≠A curriculum; eval all 56 pairs | After P3 |
| **P5 — Demo polish** | Best ckpt → local + web triple demo | After a keeper exists |

## Current phase + next micro-task

- **Phase:** P1a — walls-on UUU (**A+B+C ALL LIVE walls-on**)
- **Live (~01:25 CT):** VM `cartpole-train-od` RUNNING us-east1-b (L4 ~98–99%/14.2GB of 22.5GB; uptime ~46.5h). TB http://34.148.138.48:6006/ HTTP 200. continue-a/b/c watchers alive. Scripts md5-match box.
  - **A** walls-on UUU hold LIVE (pid **142589**, ~**u50–56**/400) — `--track-walls` / `TRACK_WALLS=1`; marker `.triple-a-walls-v1`. **nt_UUU~0.036** (flat early); **align_UUU~+0.123** (from −0.07 @u1 → +0.06 @u40 → +0.12 @u50); train reward peaked ~229 @u10 then settling ~155; **entropy~1.94** healthy; **oob_rate=0**. Hang UUU still ~0.
  - **B** **CONFIRMED walls-on PPO-balance** LIVE (pid **143661**, ~**u10** train / eval@u1) — run `…-walls-balance…`; marker `.triple-b-ppo-balance-walls-v1` (06:19Z); `track_walls=True`; nt_UUU~**0.037** align~**−0.075**; ent~**1.45**; **oob=0**. Prior void balance (pids 142965/143446) exited; TQC exited earlier (reward-hack) — do not relaunch.
  - **C** walls-on combo LIVE (pid **143053**, ~**u10–20**/400) — `track_walls=True`; marker `.triple-c-walls-v1`; nt_UUU~**0.0345** align~**−0.055** (slightly up from −0.064); reward~227 @u10; **entropy~2.25**; **oob=0**. Early — leave alone.
- **Diagnosis this fire (~01:25 CT):**
  1. **Plant phase OK:** all three live jobs have `--track-walls` + `TRACK_WALLS=1`. **B is walls-on now** (prior void mis-arm closed).
  2. **Reward↑ hold≈0:** A reward up then settling while hold still ~0.036 — **expected <u100**, not classic hack (align climbing strongly +, unlike TQC reward-hack with success≈0). C/B too early to call.
  3. **Center farm:** oob=0 on walls plant; no thrash/void-death signal. No action.
  4. **Entropy:** A~1.94 / B~1.45 / C~2.25 — not collapsed, not saturated (~3.4 was prior bad pattern).
  5. **Actions:** none — leave A/B/C cooking; no mid-kill; no TQC relaunch; no code change (continue-b already walls-v1 on box+VM, md5 match).
- **Catch-basin:** TQC zip dead; LQR tiny RoA; **PPO-balance walls-on on B** is the live catcher path.
- **Watchers:** continue-a (142090) / continue-b (143647) / continue-c (142091) alive.
- **Next micro-task:** (1) Babysit **A+B+C walls-on** through ~u100 — watch nt_UUU / align_UUU / entropy / oob; no mid-kill. (2) If A nt stays flat past ~u150 with align plateau, stage natural-exit ladder (bar10→50 / ENERGY_W↑) — never mid-kill. (3) After P1a gate (nt≳0.80 align≳0.90) → P1b nowalls FT from best walls ckpt.
- **Kill list:** no double/xonly; slots = **A walls-hold** / **B walls-balance** / **C walls-combo**
- **Do not:** mid-kill healthy stretches; relaunch wide TQC; stack GPU; treat TQC zip as catcher; run void during P1a
- **NEED_USER_PING:** **no** — first walls live already reported; B walls confirm is ops babysit; gate not clear; no strategy flip.

## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (**DONE / exited**; basin **dead** @150k; reward-hack confirmed — do not relaunch). Hold mirror ladder leftovers (post-stretch only, **after** walls PPO-balance try): (a) M2 tighten `INIT_NOISE=0.05 HANG_FRAC=0 WIDE_FRAC=0`; (b) `ry_scale=1.0`; (c) **∫x obs**; (d) Baek VER flip; (e) optional fawraw M2 arch; (f) `n_steps=3`; (g) `use_sde=True sde_sample_freq=4`.
2. **Two-policy handoff** (**CODE STAGED + basin measured**) — catcher order: **skip TQC zip** → LQR only tiny → **PPO-balance walls-on LIVE on B**. Enter \(\|\phi_i\|<0.1\) & \(\|\omega\|_\infty<1\); latch + exit 0.25 + dwell; LPF τ≈0.3. Ops leftovers: (i) `handoff_eval.py` smoke capture_tol=**0.1**; (ii) PPO-balance `--init-noise` 0.05 + `--vel-cost-coef` 0.01–0.02; (iii) if thin → LQI ∫x; (iv) **P1a: PPO-balance `TRACK_WALLS=1`** (**DONE** this fire — void v1 killed, walls-v1 armed); (v) P1a flat-eval ladder if A stays flat (natural exit only): bar10→50 if rail-park, ENERGY_W 0.2→0.35 if visit≠hold, never mid-kill / hang / force60 mid-P1a.
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU hold actually works
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
