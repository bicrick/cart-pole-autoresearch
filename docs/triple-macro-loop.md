# Triple pendulum — macro loop

Last updated: 2026-09-20 ~01:44 CT
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

**Live mapping (2026-09-20 ~01:44 CT):** **A = S1** walls-swing; **B = H1** walls-hold (tight `INIT_NOISE=0.05`); **C = H1-var** walls-hold (lr/ent/noise variant). Combo mush retired. X1 handoff_eval staged. `VEL_COST` staged for next H1 restart. Do not relaunch wide void TQC.

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
- **Live (2026-09-20 ~01:44 CT):** VM `cartpole-train-od` RUNNING us-east1-b (L4 ~99%/15.6GB). TB http://34.148.138.48:6006/ HTTP 200. Watchers continue-a/b/c. Commit `0eab823` stages `VEL_COST_COEF` for next natural H1 restart. **Roles confirmed healthy cold.**
  - **A = S1 walls-swing** — pid **145281** / watcher **146715** — `INIT_MODE=bottom`, hang_p=0.85, near_goal_p=0.10, warmup=0, progress_w=1, product, `TRACK_WALLS=1`, ENT=0.05 LR=3e-4. Marker `.triple-a-s1-walls-v1`. Run `…-walls-swing-s1-…`. ~**u16** reward~222 hang_align/UUU **-0.80→improving from -0.98** nt_UUU~0.035 OOB=0 ent~1.69↑. Healthy early.
  - **B = H1 walls-hold (tight)** — pid **145275** / watcher **146716** — `INIT_MODE=near_target`, near_goal_p=1.0, hang=0, warmup=0, **`INIT_NOISE=0.05`**, product, `TRACK_WALLS=1`, ENT=0.02 LR=1e-4. Marker `.triple-b-h1-noise05-v1`. Run `…-walls-hold-h1-…-in005-…`. ~**u18** reward~225 nt_UUU~0.034 nt_align/UUU **-0.03** (still cold) OOB=0 ent **1.45→1.12** (watch — ENT=0.02 intentional, not collapsed). Healthy early; **do not mid-kill**.
  - **C = H1-var walls-hold** — pid **145280** / watcher **146717** — same near-only as B but LR=5e-5 ENT=0.05 INIT_NOISE=0.08. Marker `.triple-c-h1var-walls-v1`. ~**u16** reward~221 nt_UUU~0.033 OOB=0 ent~1.74. Healthy early.
- **X1 handoff:** staged. Smoke deferred until H1 shows nt/align signal (ckpts exist but ~u15 cold — not informative). `VEL_COST_COEF=0.015` now wired in `next-train-triple.sh` + continue-b/c for **next natural restart only** (soft-land leftover from backlog).
- **Diagnosis / actions this fire:** Roles match S1/H1/H1-var (cmdline + markers + TB run names). Reward↑/hold≈0 is **cold-start product reward**, not yet diagnosed as hacking. No center-farm signal (OOB=0). No void/TQC. Staged vel-cost wire; left live stretches alone.
- **Next micro-task:** (1) Babysit to ~u50–100: B/C `eval/near_target/at_goal/UUU` + align; A `eval/hang/align*` climbing. (2) Watch B entropy floor — if <0.3 with nt still flat past ~u80, plan ENT 0.02→0.03–0.04 on natural exit only. (3) When B nt≳0.2 or align climbing clearly, run `eval-handoff-uuu.sh` smoke. (4) H1 gate nt≳0.80 align≳0.90 → keep H1, then P1b void FT. Never void-FT before walls hold works. Never mid-kill healthy stretch.
- **Kill list:** no double/xonly; no void TQC; slots = **A S1 swing** / **B H1 hold** / **C H1-var hold**
- **Do not:** mid-kill healthy H1/S1 stretches; relaunch wide TQC; stack GPU; treat TQC zip as catcher; run void during P1a
- **NEED_USER_PING:** **no** — early healthy cold; vel-cost staged for next restart


## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (**DONE / exited**; basin **dead** @150k; reward-hack confirmed — do not relaunch). Hold mirror ladder leftovers (post-stretch only, **after** walls PPO-balance try): (a) M2 tighten `INIT_NOISE=0.05 HANG_FRAC=0 WIDE_FRAC=0`; (b) `ry_scale=1.0`; (c) **∫x obs**; (d) Baek VER flip; (e) optional fawraw M2 arch; (f) `n_steps=3`; (g) `use_sde=True sde_sample_freq=4`.
2. **Two-policy handoff** (**CODE STAGED + basin measured**) — catcher order: **skip TQC zip** → LQR only tiny → **PPO-balance walls-on LIVE on B**. Enter \(\|\phi_i\|<0.1\) & \(\|\omega\|_\infty<1\); latch + exit 0.25 + dwell; LPF τ≈0.3. Ops leftovers: (i) `handoff_eval.py` smoke capture_tol=**0.1** (**STAGED** `scripts/handoff_eval.py` + `eval-handoff-uuu.sh`); (ii) PPO-balance `--init-noise` 0.05 (**LIVE on B**) + `--vel-cost-coef` 0.01–0.02 (**STAGED** `VEL_COST_COEF=0.015` in next-train + continue-b/c — applies on natural restart; live stretch undisturbed); (iii) if thin → LQI ∫x; (iv) **P1a: PPO-balance `TRACK_WALLS=1`** (**DONE**); (v) P1a flat-eval ladder if A stays flat past ~u150 (natural exit only) — **branch**: bar10→50 if rail-park (\|x\|@limit + flop), else ENERGY_W 0.2→0.35 if align↑/mid + nt flat (Spong visit≠hold), optional EPISODE_LEN 1200→600–800 third rung (Turcato); never mid-kill / hang / force60 mid-P1a; (vi) after A has hold signal: soft-land delivery (−w_ω / cart-centre / V_aug ẋ→0) so handoff lands in B RoA (fawraw M4 / 2606.28627).
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU hold actually works
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
