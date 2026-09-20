# Triple pendulum — macro loop

Last updated: 2026-09-20 ~04:23 CT
Owner: overnight routine (every 15m). Edit this file when the next micro-task changes.

## Overarching goal

Ship a cracked cart-**triple**-pendulum policy that can:

1. Swing up and **hold UUU** from hard starts (hang / near-target),
2. Stabilize / switch among all **8** discrete equilibria,
3. Eventually cover all **56** directed transitions for the interactive demo,

on the no-walls plant (`forceLimit` ≥ 40N), with TensorBoard + checkpoints mirroring to the user's local demo. Budget: **no hard $30 cap** — keep training until UUU works; prefer one L4 on-demand, don't stack GPU VMs.

## Think out loud (user 2026-09-20)

Overnight **always** pings Patrick each 15m fire: meters, missing/wrong/how, what we did, what's next. No silent babysit-only runs. Research pings only on material findings.

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

**Live mapping (2026-09-20 ~04:23 CT):** **A = S1** walls-swing (**LR=1e-4** cold restart @08:46Z — hang_align **−0.97→−0.059 @u60**, ent~2.24↑ @u62, policy_loss sane, no 2nd NaN); **B = H1** walls-hold (tight `INIT_NOISE=0.05`, live ENT=0.02 **dead≈−0.42 @u307**, **ENT=0.035 staged** for natural restart ~05:14 CT); **C = H1-var** walls-hold (ENT=0.05 live, ent~1.36, **nt still ~0.046** — ENT null for hold). Combo mush retired. X1 handoff_eval staged. `VEL_COST=0.015` + B `ENT=0.035` apply on **natural restart only**. Do not relaunch wide void TQC. **Do not** put AR-EAPO MaxEnt on H1 (MaxEnt fights upright stay — 2503.15290).

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
- **Phase:** P1a — walls-on UUU via **role split** (S1 / H1 / H1-var), not one mega-policy
- **Live (2026-09-20 ~04:23 CT):** VM `cartpole-train-od` RUNNING us-east1-b (L4 ~99%/16.6GB, uptime ~2d). TB http://34.148.138.48:6006/ HTTP 200. Watchers continue-a/b/c.
  - **A = S1 walls-swing** — prior NaN @u228 quarantined; **LR=1e-4 cold restart pid 153439** @08:46Z. ~**u62** hang_align/UUU **−0.97→−0.059** (@u60), hang_at_goal~0.002, ent **~2.24↑**, policy_loss sane — **still climbing; passed u50 clean, watch through ~u80**. Marker `.triple-a-s1-walls-v1`.
  - **B = H1 walls-hold (tight)** — pid **145275** / w **147599** — near_goal_p=1.0, hang=0, `INIT_NOISE=0.05`, product, walls, **live ENT=0.02** (next restart **ENT=0.035** + VEL_COST=0.015). Marker `.triple-b-h1-noise05-v1`. ~**u307**/400 nt_UUU **~0.052** nt_align/UUU **~0.170** ent **≈−0.42** (deeper dead). **Entropy dead + visit≠hold; do not mid-kill** — stretch ETA ~**05:14 CT** → ENT bump.
  - **C = H1-var walls-hold** — pid **145280** / w **146717** — LR=5e-5 ENT=0.05 INIT_NOISE=0.08. Marker `.triple-c-h1var-walls-v1`. ~**u299** nt_align/UUU **~0.147** nt_UUU **~0.046** ent **~1.36**. **ENT=0.05 did not unlock hold** (natural null vs B). Leave alone.
- **X1 handoff:** staged. Smoke deferred until H1 nt≳0.2 (B/C still ~0.05). Soft-land `VEL_COST_COEF=0.015` already in continue-b/c.
- **Diagnosis / actions this fire:** A hang still climbing (−0.059), healthy σ — leave alone. B σ deeper dead (−0.42) + nt flat ~0.05 + reward~0.49 — leave to natural end; ENT035 continue confirmed staged. C control unchanged. OOB=0. No mid-kills / no train starts / no code.
- **Next micro-task:** (1) Keep watching A through ~u80 for second NaN (if policy_loss explodes → mid-kill + further LR drop / grad clip). (2) Babysit B to stretch end (~u400 / ~05:14 CT): confirm ENT=0.035 continue; if H still ↓ + nt flat ~u80 post-restart → **RPO α≈0.01** or **ERA soft log_std** (not ENT≥0.05); if visit≠hold persists → Spong ENERGY_W / short ep / avg-reward **stay** only — **never AR-EAPO MaxEnt on H1** — before cold wipe / C-recipe promote (lr/noise, not C’s ENT). (3) A hang_align toward 0 / hang_at_goal↑ → when any H1 nt≳0.2 run `eval-handoff-uuu.sh`. (4) C keep as H1-var control. (5) H1 gate nt≳0.80 align≳0.90 → keep H1, then P1b void FT. Never void-FT before walls hold. Never mid-kill B/C.
- **Kill list:** no double/xonly; no void TQC; slots = **A S1 swing** / **B H1 hold** / **C H1-var hold**
- **Do not:** mid-kill B/C; relaunch wide TQC; stack GPU; run void during P1a; resume from NaN ckpt
- **NEED_USER_PING:** **yes** — standing think-out-loud every fire


## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (**DONE / exited**; basin **dead** @150k; reward-hack confirmed — do not relaunch). Hold mirror ladder leftovers (post-stretch only, **after** walls PPO-balance try): (a) M2 tighten `INIT_NOISE=0.05 HANG_FRAC=0 WIDE_FRAC=0`; (b) `ry_scale=1.0`; (c) **∫x obs**; (d) Baek VER flip; (e) optional fawraw M2 arch; (f) `n_steps=3`; (g) `use_sde=True sde_sample_freq=4`.
2. **Two-policy handoff** (**LIVE role split** A=S1 / B=H1 / C=H1-var + basin measured) — catcher order: **skip TQC zip** → LQR only tiny → **PPO-balance walls-on LIVE on B**. Enter \(\|\phi_i\|<0.1\) & \(\|\omega\|_\infty<1\); latch + exit 0.25 + dwell; LPF τ≈0.3. Ops leftovers: (i) `handoff_eval.py` smoke capture_tol=**0.1** (**STAGED**); (ii) `--init-noise` 0.05 (**LIVE on B**) + `VEL_COST_COEF=0.015` (**STAGED** next H1 restart only); (iii) **BaRC widen-after-mastery**: when B nt/at_goal/UUU ≳ **0.5** on noise=0.05, expand `INIT_NOISE` 0.05→0.10→0.15 (+ optional ω / off-centre x; hang=0) **before** soft-landing A (fawraw M4 / BaRC); (iv) if thin → LQI ∫x; (v) **P1a walls** (**DONE**); (vi) P1a flat-eval ladder if A flat past ~u150 (natural exit only) — **branch**: bar10→50 if rail-park, else ENERGY_W 0.2→0.35 if align↑/mid + nt flat (Spong), optional EPISODE_LEN 600–800 (Turcato); never mid-kill / hang / force60 mid-P1a; (vii) after A has hold signal: soft-land S1 (−w_ω / cart-centre / V_aug ẋ→0) so handoff ⊆ B RoA (fawraw M4 / 2606.28627); (viii) B entropy watch: ENT 0.02→0.035 **LIVE staged** (collapse H≈−0.29@u270; reward↑/hold≈0); if ENT035 fails → **RPO α≈0.01 μ-perturb at update** (CleanRL IDP; α=0.5 catastrophic on InvertedDoublePendulum — arXiv:2212.07536 Alg.1) or **ERA soft `log_std` floor** (H₀≈0.5–0.8; arXiv:2510.08549) first — **not** ENT≥0.05 (**C ENT=0.05 natural null**: H~1.47 but nt~0.04); if visit≠hold persists (align↑ / nt flat) → **non-MaxEnt stay pressure only** (Turcato short ep / Spong ENERGY_W / later average-reward soft bias — arXiv:2409.08938) before cold wipe / C-recipe promote (lr/noise knobs); **ban AR-EAPO MaxEnt / entropy-advantage on H1** — AI Olympics lessons (arXiv:2503.15290 §III-B) show MaxEnt *prevents staying upright*; hard clamp on global `log_std` zeros grad (rsl_rl). ASAP/CAPS thrash fix only after VEL_COST soft-land.
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU hold actually works
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
