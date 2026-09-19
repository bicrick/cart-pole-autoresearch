# Cart-triple-pendulum research notes

Last updated: 2026-09-19 ~10:10 CT.


## Implementation status (2026-09-19)

**Built on `main`:** cart-triple plant parallel to the double, same UVFA+PPO recipe (not Lim 8× TQC).

| Piece | Path | Notes |
|---|---|---|
| Physics | `train/physics_triple.py` | Batched torch, 4×4 mass solve, θ=0 upright, no track walls |
| Constants | `shared/constants-triple.json` | 3 equal links; `obsDim=25` |
| Goals | `train/goals_triple.py` | **8 eqs:** DDD, DDU, DUD, DUU, UDD, UDU, UUD, UUU; `OBS_DIM=25` |
| Train | `train/train_triple.py` | Reuses PPO; align/energy/center/oob; HER optional; `--transition-only` available but **not** the default |
| Smoke | `train/test_physics_triple.py` | Inverted unstable, hang restoring, nowalls |
| Launch | `scripts/next-train-triple.sh` | **Normal multi-eq curriculum** (hang / near-goal / UUU bias / energy) like early double |
| Watcher | `scripts/continue-triple.sh` | Loop forever like `continue-xonly.sh` |

**OBS_DIM = 25** = 11 state (`x,ẋ,sin/cos×3,ω×3`) + one-hot(8) + target sin/cos(6).

**Curriculum default:** hang_start / near-goal / wrong-eq / soft UUU bias — reach and hold the 8 equilibria first. Do **not** default to `--transition-only` (that remains the double xonly experiment). Transition-only over 56 pairs is a later phase once local capture works.

**L4 launch status:** Plant synced to `cartpole-train-od` for import/physics smoke only. **Do not** start triple GPU trains until the overnight gate / parent says go — leave double xonly alone.

**Launch (cold):**
```bash
NUM_ENVS=8192 bash scripts/next-train-triple.sh
# A/B tip: HANG_START_P=0.55 LR=1e-3 RUN_NAME=... CHECKPOINT=policies/checkpoint-triple-b.pt OUT=policies/policy-triple-b.json
```

Double **xonly** on GCP stays untouched.


## What "56" means

Not 56 equilibria. A planar cart-**triple**-pendulum has **8** discrete equilibria: each of 3 links is Up or Down → \(2^3 = 8\) (DDD … UUU). Directed transitions between distinct eqs: \(8 \times 7 =\) **56**. (Our double plant is the same idea: \(2^2 = 4\) eqs, \(4 \times 3 = 12\) directed transitions.)

Representation (same UVFA style as our double):
- State: cart \(x,\dot x\) + \((\sin\theta_i,\cos\theta_i,\dot\theta_i)\) for \(i=1,2,3\)
- Goal: one-hot(8) + target sin/cos for three angles (or 3-bit EP index)

## Classical control vs RL

| Approach | What it does | 56 transitions? |
|---|---|---|
| Glück et al. Automatica 2013 | Nonlinear feedforward (BVP) + time-varying Riccati; **experimental DDD→UUU** swing-up | Method applies in principle to other EPs; **not** all 56 demonstrated |
| Graichen et al. CDC 2005 | Constrained feedforward + LQR side-stepping of upright triple | Side-step / hold, not the full 56 graph |
| Lim / Ju / Lee KIEE 2025 | Sim-to-real **TQC**; custom high-fidelity plant; **all 56 on hardware** | Yes (reported + YouTube) |
| fawraw/triple-pendulum-sim2real | MuJoCo + TQC; aiming at 56 without feedforward | M4 still in progress (hand-off basin); as of 2026-06 not done |
| MDPI Machines 2025 (same lab lineage) | Double inverted pendulum Sim2Real; **4 EPs / 12 transitions** | Double cousin of our problem |

**Bottom line:** Classical feedforward covers swing-up / side-step with models. Getting *all* 56 without precomputed trajectories is the RL/sim2real benchmark — and Lim et al. already published a hardware demo (2025). PPO/TQC/SAC are viable; our transition-only PPO UVFA is one shape; Lim used a different one (below).

## Lim KIEE 2025 — how they actually did all 56 (read PDF)

Source: [KIEE2025_b.pdf](http://ecsl.inha.ac.kr/publication/KIEE2025_b.pdf); demo [YouTube](https://youtu.be/vVx3ffGo2mk).

| Choice | Lim et al. | Our double stack |
|---|---|---|
| Algo | **TQC** (3 critics, 25 atoms; policy 400→300; lr 3e-4) | PPO |
| Goal encoding | **8 separate policies** (one per EP); retarget by swapping the agent | Single UVFA + one-hot / goal sincos |
| Transition curriculum | **None explicit** — randomize start state over wide ranges; each policy attracts to its EP → transitions emerge | `--transition-only` samples A→B≠A every ep |
| Reward | **Product** of [0,1] terms: \(R_u R_y R_{\theta1} R_{\theta2} R_{\theta3} R_{\dot\theta1}\ldots\) with **cumulative absolute angles** \(\theta_1\), \(\theta_1+\theta_2\), \(\theta_1+\theta_2+\theta_3\) vs targets | Dense align / at_goal style |
| Reality gap | Domain randomization of ICs + purpose-built hardware (dual-rail, hollow-shaft joints, degreased bearings, direct-drive BLDC) | Sim-only for now |
| Action | Cart **acceleration** \(u\) (with \(\|u\|\) penalty) | Cart force (1D) |

Implication: "all 56" does **not** require training 56 specialists. Eight attractors + diverse starts can cover the graph. Our UVFA + transition-only is still a valid (and more parameter-efficient) alternative; Lim is evidence the plant is solvable with off-policy continuous control.

## Can we "append" the double policy?

Action stays 1D (cart force) — good. Observation and goal encodings **must grow**:
- Double obs_dim 16 → triple roughly +3 (sin/cos/ω for link 3) + expand one-hot 4→8 and goal sincos +2 → new obs_dim ~22–24.

Naïve "pad and continue" does **not** just work: weights for θ3 / new goal bits are undefined; dynamics change (coupling). Options:

1. **Input/output adapters** — freeze double hidden layers; new linear map; expand goal head; fine-tune with PPO (PPOPT-style).
2. **Progressive nets** — freeze double column; new column for triple with lateral adapters (Rusu et al. 2016). Keeps double policy intact.
3. **Fresh policy + curriculum** — train triple from scratch (Lim-style 8× TQC, or our UVFA PPO); use double only as teacher / reward shaping for the first two links.
4. **Lim-style 8 specialists** — skip UVFA entirely; train one TQC/PPO per EP with randomized starts.

**Clear answer on transfer:** No free append. Lim trained fresh. Adapters/PNN can reuse base-link swing instincts; third link + full graph still need substantial new learning. Prefer (3) or (4) as default; treat (1)/(2) as an A/B experiment once a plant exists.

## Implication for our stack

- Keep double **xonly** grinding 12 directed transitions on the shared L4; share headroom with triple multi-eq jobs (do not kill xonly).
- Triple plant is live (see Implementation status). Default recipe: **normal multi-eq PPO** (hang/near-goal/UUU bias) to reach/hold 8 eqs — same path as early double, **not** transition-only first.
- Later: optional `--transition-only` over 56 pairs, or Lim-style 8× TQC as A/B — not the overnight default.
- Optional: adapter/progressive init from double ckpt as a side experiment, not a requirement.

## Sources

- Glück, Eder, Kugi — Swing-up of a triple pendulum on a cart (Automatica 2013) — DDD→UUU
- Baek, Lee, Lee, Jeon, Han — TIP swing-up RL + VER (EAAI 128:107518, 2024); product-reward lineage for Lim
- Graichen, Treuer, Zeitz — Fast side-stepping of the triple inverted pendulum (CDC 2005)
- Lim, Ju, Lee — 56 transition control via sim-to-real RL / TQC (KIEE 2025); PDF + YouTube above
- MDPI Machines 13(3):186 (2025) — double inverted pendulum Sim2Real, 4 EPs / 12 transitions
- https://github.com/fawraw/triple-pendulum-sim2real (open TQC attempt; M4 not finished as of mid-2026)
- Progressive Neural Networks (Rusu et al. 2016); PPOPT-style adapters


## Drawing board — why our overnight triple is stuck (2026-09-19 ~09:30 CT)

Live: mean align ~0.18–0.22, **DDD ~0.45–0.55**, **UUU ~0.01–0.05**. Hang curriculum is teaching “stay down,” not swing up.

### What Lim / Ju / Lee (KIEE 2025) actually did (PDF)

- **Algo:** TQC (not PPO) — distributional critics, truncate top quantiles (Kuznetsov).
- **Architecture:** **8 separate policies**, one per EP (EP0=DDD … EP7=UUU). Not one UVFA. Transitions “for free” once each EP is reachable from random ICs.
- **Reward:** **product** of [0,1] terms (max 1/step; ep max 1000 over 10 s @ 10 ms):
  - \(R_u = \exp(-0.001 u^2)\), \(R_y = \exp(-0.3 |y|)\)
  - \(R_{\theta_1} = 0.5 + 0.5\cos(\theta_1 - \theta_1^*)\)
  - \(R_{\theta_2} = 0.5 + 0.5\cos(\theta_1+\theta_2 - \theta_2^*)\)  (world/cumulative abs angles)
  - \(R_{\theta_3} = 0.5 + 0.5\cos(\theta_1+\theta_2+\theta_3 - \theta_3^*)\)
  - \(R_{\dot\theta_1} = \exp(-0.015 |\dot\theta_1|)\), \(R_{\dot\theta_2} = \exp(-0.009 |\dot\theta_1+\dot\theta_2|)\), \(R_{\dot\theta_3} = \exp(-0.005 |\dot\theta_1+\dot\theta_2+\dot\theta_3|)\)
  - \(R = \prod R_\cdot\)  (all must be good — no compensating “hang forever” with cart motion)
- **ICs (exact):** \(y\sim U(-0.3,0.3)\), \(\dot y\sim U(-1.2,1.2)\); \(\theta_i\sim U(-\pi,\pi)\); \(\dot\theta_1\sim U(-10,10)\), \(\dot\theta_2\sim U(-20,20)\), \(\dot\theta_3\sim U(-30,30)\).
- **Episode:** 10 s @ 10 ms agent step (1000 steps); early stop if |y|>0.48 m or |a|>2.5 m/s².
- **Net:** critic 3×512, policy 400→300; lr 3e-4; γ 0.99; buffer 1e6; 25 atoms; minibatch 256.
- **EP targets (Table 3):** world angles — EP0 all −π; EP7 all 0; mixed EPs set each link’s world angle to 0 (up) or −π (down).

Demo: https://youtu.be/vVx3ffGo2mk

### What fawraw/triple-pendulum-sim2real does

- **Milestones:** M2 = stabilize **UUU first** → M3 = all 8 EPs (upweight hard EPs) → M4 = 56 transitions.
- **TQC** + MuJoCo; larger nets ([512,512] breakthrough vs [256,256] catastrophic forgetting on hard EPs).
- **M3 hard-EP recipe that worked (M3b-v6):** warm-start prior ckpt → phase1 `hard_ep_weight=20` (~46% each on EP4/EP6) lr=1e-4 × 600K → phase2 consolidate `hard_ep_weight=2.5` lr=5e-5 × 500K → 72.5% overall, all 8 EPs non-zero.
- Swing-up failure mode: **cart slide to rail** local optimum. Fixes: cart barrier \((x/\mathrm{limit})^8\) (coef≈50), progress shaping, higher cart cost.
- **M4 status (2026-06):** two-stage hand-off (swing-up → M3 stabilizer). Binding constraint = catcher basin ~0.1 rad / near-zero vel; swing-up delivers ~0.2 rad mid-swing. Next for them: wider-basin catcher + **soft landing** (arrive slow + cart-centred). See their `docs/m4_findings.md`.
- Probe single transitions (DDD→UDD easy, DDD→UUU hard) before full graph.

### Classical (Glück Automatica 2013)

Feedforward BVP + time-varying Riccati — works for DDD→UUU with a model; not our path unless we leave pure RL. (T≈3.5 s, accel limit ~22 m/s² in their setup; experimental validation.)

### Adjacent (not cart-56)

Cambridge Robotica 2026 (CSAC-QI): underactuated triple **balance** only (SAC + integral joint-error reward + curriculum). Useful for hold-precision ideas; not a 56-transition recipe.

### Vs our current plant

| Ours | Papers that work |
|---|---|
| One UVFA, hang_start 0.45–0.60 | Lim: 8 specialists; fawraw: UUU-first milestone |
| Additive align/energy/clip like double | Lim: **product** reward on absolute/world angles |
| Hang-biased resets → DDD sink | Wide random ICs (Lim ranges above) + explicit UUU stage |
| PPO on-policy | TQC/SAC off-policy (user still prefers PPO — keep PPO but steal reward+curriculum) |

### Recommended redesign (when user green-lights)

1. **UUU-only warmup** (fawraw M2) before multi-eq.
2. Port Lim-style **product reward** with the exact coeffs above (absolute/world θ) into `train_triple`.
3. Cut hang_start way down; adopt Lim IC ranges (or close).
4. Multi-eq stage: **hard-EP oversample** (fawraw M3 weight schedule) once UUU holds — tip-up-only / mid-up EPs starve under uniform 1/8.
5. Optional: **8 PPO heads / 8 runs** (one EP each) instead of one UVFA — matches Lim’s successful structure while keeping PPO.
6. Cart barrier \((x/\mathrm{limit})^8\) / stronger center so we don’t learn rail-slide.
7. Later (56 phase): soft-landing term + optionally swing-up→hold hand-off; do **not** start transition-only until local capture works.
8. Leave current A/B grinding only until redesign is coded — or pause one slot for experiments. Do not kill double xonly.

### Recipe lock-in (2026-09-19 ~09:37 CT research pass)

No direction change vs morning drawing board — only **copy-paste numbers** filled from Lim PDF + fawraw M3/M4 notes so a code pass can start without re-reading papers. Still waiting on user green-light to implement; overnight UVFA+hang stays parked as the failed recipe.


### Research pass (2026-09-19 ~10:10 CT) — new actionable diffs

Digged Baek EAAI 2024 (same Inha / tip-up lineage as Lim) + fawraw configs/env code. **Direction unchanged** (UUU-first → product/abs reward → hard-EP → later 56); three new levers to fold into the redesign when coded.

#### Baek et al. EAAI 2024 — product reward + VER (swing-up to UUU on hardware)

Scope: **DDD→UUU only** (not 56); trained **on hardware** (no sim2real). Off-policy actor-critic + structure-aware **Virtual Experience Replay**.

Dense **product** reward (Lim’s form is the multi-EP cousin of this):

\[
R(s,a)=f(a)\,g(x)\,h(\theta_1)\,h(\theta_2)\,h(\theta_3)\,\min\big[e(\dot\theta_1),e(\dot\theta_2),e(\dot\theta_3)\big]
\]

with floors so no term can zero the whole product:

- \(f(a)=\alpha_f+(1-\alpha_f)\max[1-(a/a_{\max})^2,0]\) — \(\alpha_f=0.80\)
- \(g(x)=\alpha_g+(1-\alpha_g)\exp(-c_g x^2)\) — \(\alpha_g=0.50\), \(c_g=0.57\)
- \(h(\theta)=\alpha_h+(1-\alpha_h)(1+\cos\theta)/2\) — \(\alpha_h=0.50\) (θ=0 upright)
- \(e(\dot\theta)=\alpha_e+(1-\alpha_e)\exp(-c_e\dot\theta^2)\) — \(\alpha_e=0.50\), \(c_e=0.09\)
- Velocity uses **min** of the three \(e\) (all links must be calm)

**VER:** reflect trajectories across the cart midline (left↔right symmetry) into the replay buffer → ~⅔ fewer trials/steps/wall time in their report. Free sample-efficiency for any off-policy run; for our on-policy PPO, the same symmetry is a **rollout augmenter** (mirror obs/action each batch) if we want it without a replay buffer.

Lim (KIEE 2025) is the same lab’s next step: product reward → 8 EP specialists → all 56 on hardware via sim2real.

#### fawraw knobs we hadn’t locked (from configs + `triple_pendulum_env.py`)

| Knob | Value that worked / mattered | Why it matters for us |
|---|---|---|
| Adaptive angle weights | **UP-targeted links ×5**, DOWN ×`w_down` (default 1) | Fixes inverted gradient: hang links stop dominating the cost when tip should be up — direct anti-DDD-sink under additive rewards |
| M2 / hold stage | `init_mode=near_target`, `init_noise=0.05`, 150K TQC, net [128,128] | UUU hold is **near-upright starts**, not hang_start |
| M3 hard EPs | EP4=**DDU** (tip-up only), EP6=**DUU**; `hard_ep_weight=20` → 72.5% but **EP7 80%→40%**; v7 tries **weight=10** | Prefer weight≈10 first if UUU regresses |
| M4 probe A (easy) | DDD→UDD; `init_mode=bottom`; `cart_barrier_coef=50`, `(x/limit)^8`; `cart_cost_coef=0.2`; `progress_reward_coef=1.0`; `cart_limit=1.10`; ep 2000 steps; lr 1e-4; 500K | Exact anti-rail-slide + swing-up gradient numbers |
| M4 probe B (hard) | DDD→UUU; same bottom init; `cart_cost_coef=0.5` (no barrier in that yaml yet) | Isolates 3-link swing-up difficulty |
| Catch basin (M3@UDD) | Reliable only at **≤0.1 rad & ~0 vel**; 0.2 rad / any vel ≈ 0 | Soft-landing + wider-basin catcher before 56 hand-off |
| Probe ladder | If A learns & B fails → **1-link → 2-link → 3-link** swing-up curriculum (warm-start each stage); if neither → add **energy-based** swing-up term | De-risk before a long GPU run |

fawraw reward itself is still **additive** (−weighted ang² − vel − cart − barrier − ctrl + progress), not Lim/Baek product — their breakthroughs for multi-EP were curriculum + weighting + barrier, not product. Steal both families.

#### Amend recommended redesign (additions only)

9. **Adaptive UP×5 / DOWN×1** angle costs in any additive stage (cheap; fights hang sink before product lands).
10. Prefer **Baek-style product with α floors** (or Lim’s pure product) — floors avoid total reward collapse when one link is wrong.
11. Optional **VER / mirror augment** on rollouts (cart left↔right).
12. After UUU holds: **probe ladder** DDD→UDD before DDD→UUU; then 1→2→3-link curriculum if needed.
13. Multi-eq hard-EP: start `hard_ep_weight≈10`, escalate to 20 only if tip-up EPs stay at 0; watch UUU regression.
14. Copy probe barrier pack when swing-up starts: `barrier_coef=50`, `cart_cost≈0.2–0.5`, `progress=1.0`.

Still **do not** redesign code or kill double xonly until user green-lights.

