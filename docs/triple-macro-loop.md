# Triple pendulum — macro loop

Last updated: 2026-09-20 ~09:40 CT
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

**Live mapping (2026-09-20 ~09:40 CT):** **A = S1 anti-DDD** ~u20 (W_DOWN=0 UU_BIAS=2 ENERGY_W=0.6, marker `.triple-a-s1-antidd-v2`) hang_UUU −0.97→−0.71 hang_DDD 0.70→0.23; **B = H1 ATRPO+AVC ν=0.2** ~u76 (pid 172580) — H **0.517 hard-pin since u56**, nt **0.039→0.054**, nt_align −0.08→+0.178, avc_bias~−0.6..−1.1, nt_DDD 0.84→0.05, log_std≈−5; **C = H1-var** ~u60 (pid 173060). Combo mush retired. X1 staged. EMA-AVC **staged not armed**. Do not relaunch wide void TQC. **Do not** put AR-EAPO / ENT≥0.02 on RPO. **Do not** blind-resume floor-σ. **Do not** re-arm gSDE / ENERGY_W-only / plain ATRPO after AVC. **Stay order:** ENERGY_W (FAIL) → ATRPO-lite (**FAIL @u100**) → **ATRPO+APO AVC ν≈0.2 LIVE** → if FAIL @u80–100: **EMA-AVC harden (Alg.1, staged)** or **Naik ρ-center+keep-γ** → Turcato short-ep only as discounted fallback.

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
- **Live (2026-09-20 ~09:40 CT):** VM `cartpole-train-od` RUNNING us-east1-b L4 ~99%/15.5GB; TB http://34.148.138.48:6006/. Watchers continue-a/b/c. TRACK_WALLS=1 all slots.
  - **A = S1 anti-DDD** — pid **174599** ~**u20** (marker `.triple-a-s1-antidd-v2`). hang_align/UUU **−0.97→−0.71↑**, hang_at_goal/DDD **0.70→0.23↓**, nt_DDD **0.87→0.39**, ent~1.79 OOB=0 — anti-DDD early win; leave alone. If DDD farm returns by ~u80 → height-line bonus (Acrobot 2312.11311) — **not** W_DOWN=1.
  - **B = H1 ATRPO+AVC ν=0.2** — pid **172580** ~**u76**. nt_UUU **0.039→0.054 flat**, nt_align **−0.08→+0.178↑** (plateau u60–70), nt_rew 232→254→249, H **0.517 hard-pin since u56**, avc_bias **~−0.6..−1.1**, ρ~0.49, nt_DDD **0.84→0.05**, ckpt log_std **≈−5**, OOB=0. FAIL meters already (H≲0.55∧nt≲0.10∧align↑) but **babysit through u80–100** (ban mid-kill). FAIL even if |bias|≪1.
  - **C = H1-var** — pid **173060** ~**u60** nt_UUU **~0.046** ent **~0.72** OOB=0 — leave alone. Marker `.triple-c-h1var-walls-v1`.
- **X1 handoff:** staged. Smoke deferred until H1 nt≳0.2.
- **Diagnosis / actions this fire (overnight):** B still visit≠hold under healthy AVC-lite — leave alone until u80–100. A anti-DDD healthy early (DDD hang collapsing). No mid-kill / no marker arm.
- **Next micro-task:** (1) Next fire (~u95–110): if H≲0.55 ∧ nt≲0.10 ∧ (align↑∨rew↑) → touch `.triple-b-h1-atrpo-avc-ema-v1` + mid-kill into **EMA-AVC α=0.1** (staged) — **even if |avc_bias|≪1**. Else if γ-free family exhausted → **Naik ρ-center+keep-γ**. (2) Keep watching A hang_align/UUU + hang_at_goal/DDD. (3) C leave as H1-var control. (4) H1 nt≳0.2 → handoff smoke; gate nt≳0.80 align≳0.90 → P1b void FT.
- **Kill list:** no double/xonly; no void TQC; slots = **A S1 anti-DDD** / **B H1 ATRPO+AVC LIVE** / **C H1-var**
- **Do not:** relaunch ENT035; relaunch ENT=0.02-on-RPO; relaunch gSDE; relaunch ENERGY_W-only; relaunch plain ATRPO after AVC live; relaunch wide TQC; stack GPU; void during P1a; resume NaN ckpt; hard-clamp Listing-2 D=1; **blind-resume floor-σ without RESET_LOG_STD**; **stack ENT β ≥0.02 on RPO**; **re-arm gSDE after FAIL**; **shorten EPISODE_LEN before Naik**; **skip Naik and jump to short-ep**; **re-arm ENERGY_W-only**; **mid-kill AVC before u80–100 FAIL confirm**; **arm EMA-AVC marker while AVC-lite still live**; **relaunch A with W_DOWN=1 after anti-DDD**
- **NEED_USER_PING:** **yes** — think-out-loud (B pre-gate babysit; A anti-DDD early green)


## Ranked backlog (pull from top when a stretch ends)

1. **Lim TQC UUU specialist** (**DONE / exited**; basin **dead** @150k; reward-hack confirmed — do not relaunch). Hold mirror ladder leftovers (post-stretch only, **after** walls PPO-balance try): (a) M2 tighten `INIT_NOISE=0.05 HANG_FRAC=0 WIDE_FRAC=0`; (b) `ry_scale=1.0`; (c) **∫x obs**; (d) Baek VER flip; (e) optional fawraw M2 arch; (f) `n_steps=3`; (g) `use_sde=True sde_sample_freq=4` (**LIVE on B** `a7a764f`).
2. **Two-policy handoff** (**LIVE role split** A=S1 / B=H1 / C=H1-var + basin measured) — catcher order: **skip TQC zip** → LQR only tiny → **PPO-balance walls-on LIVE on B**. Enter \(\|\phi_i\|<0.1\) & \(\|\omega\|_\infty<1\); latch + exit 0.25 + dwell; LPF τ≈0.3. Ops leftovers: (i) `handoff_eval.py` smoke capture_tol=**0.1** (**STAGED**); (ii) `--init-noise` 0.05 (**LIVE on B**) + `VEL_COST_COEF=0.015` (**STAGED** next H1 restart only); (iii) **BaRC widen-after-mastery**: when B nt/at_goal/UUU ≳ **0.5** on noise=0.05, expand `INIT_NOISE` 0.05→0.10→0.15 (+ optional ω / off-centre x; hang=0) **before** soft-landing A (fawraw M4 / BaRC); (iv) if thin → LQI ∫x; (v) **P1a walls** (**DONE**); (vi) P1a flat-eval ladder if A flat past ~u150 (natural exit only) — **branch**: bar10→50 if rail-park, else ENERGY_W 0.2→0.35 if align↑/mid + nt flat (Spong), optional EPISODE_LEN 600–800 (Turcato); never mid-kill / hang / force60 mid-P1a; (vii) after A has hold signal: soft-land S1 (−w_ω / cart-centre / V_aug ẋ→0) so handoff ⊆ B RoA (fawraw M4 / 2606.28627); (viii) B entropy watch: ENT 0.02→0.035 **LIVE staged** (collapse H≈−0.29@u270; reward↑/hold≈0); if ENT035 fails → **RPO α≈0.01 + ERA H₀≈0.5 with ENT=0** (CleanRL `ent_coef=0`; α=0.5 catastrophic on IDP — arXiv:2212.07536 Alg.1 / §4.2.2; ERA soft floor — **ban Listing-2 D=1**; arXiv:2510.08549) — **ban ENT≥0.02 on RPO** (live ENT=0.02 mismatch 06:27); then **gSDE / Zoo-Pendulum `ent_coef=0 + use_sde sde_sample_freq=4`** (Raffin 2005.05719; relocate from TQC leftover g) — explore without ENT β-crank (**C ENT=0.05 natural null**: H~1.47 but nt~0.04); if visit≠hold persists (align↑ / nt flat) **or gSDE H diving by u30–40** → **non-MaxEnt stay pressure only** (Spong ENERGY_W 0.2→0.35 → **ATRPO-lite first**: ρ-center + γ-free GAE / bias critic, ENT=0, **keep EPISODE_LEN ≥1200** — Zhang–Ross 2106.07329 Alg.2/§6.3/App.G; optional ATRPO §6.2 reset-cost/continue-traj spirit on arm; if ATRPO-lite FAIL (H≲0.55@ERA **and** nt≲0.10 @u80–100) → **APO AVC** ν≈0.1–0.3 (**LIVE** ret-shift ν=0.2; research 08:33/09:00) → if AVC-lite FAIL same gate → **APO-faithful EMA-η̂ (α≈0.1) + EMA-V bias AVC** (Alg.1; research 09:00) **and/or** **Naik** ρ-center+**keep γ** — 2405.09999 / Wan 2501.06937 (twins; research 07:56); Turcato EPISODE_LEN 600–800 only as **discounted fallback** after those; EVAL-PPI 2501.09770 confirmation-only — **not** AR-EAPO 2409.08938 MaxEnt pair; research 07:36+07:56); optional **PPO-BR ε contract** (λ₂≈0.3, ε 0.2→0.1) when reward flat after σ tools (2505.17714) before cold wipe / C-recipe promote (lr/noise knobs); **ban AR-EAPO MaxEnt / entropy-advantage on H1** — AI Olympics lessons (arXiv:2503.15290 §III-B) show MaxEnt *prevents staying upright* (2503.15290 §III-B; 2506.05615 soft-Q bifurcation); hard clamp / Listing-2 D=1 pin on global `log_std` zeros grad (rsl_rl); after nt moves anneal ENT→0 (AdaEnt spirit). ASAP/CAPS thrash fix only after VEL_COST soft-land.
3. Energy-to-goal (true E→E_UUU) if product+progress plateaus
4. Force probe 40→60 only if OOB≈0 and plant feels underpowered
5. 8×TQC specialists (Lim full set) after UUU hold actually works
6. Progressive/adapters from double — deferred; fresh triple policy first

## 15-minute cadence (what "adapt" means)

Each fire: read this file → measure live TB/slots → decide whether the **next micro-task** above still holds → if evidence says jump (gate cleared or idea dead), **rewrite the Next micro-task line** and stage the matching continue script / code change → push box repo to `main` → keep VM+TB alive → ping user only on phase change, gate clear, or NEED_USER.
