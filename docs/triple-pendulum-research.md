# Cart-triple-pendulum research notes

Last updated: 2026-09-19 ~15:42 CT.


## Implementation status (2026-09-19 ~14:35 CT)

**Built on `main`:** cart-triple plant + Lim/fawraw product + **P0/P1 progress + flip-augment + curriculum eval**.

| Piece | Path | Notes |
|---|---|---|
| Physics | `train/physics_triple.py` | Batched torch, 4×4 mass solve, θ=0 upright, no track walls |
| Constants | `shared/constants-triple.json` | 3 equal links; `obsDim=25`; **`forceLimit=40.0`** |
| Goals | `train/goals_triple.py` | 8 eqs; product reward + **`progress_w` Δ cos-align**; Baek α floors; UP×5/DOWN×1; **`energy_w` = height/align proxy (not E→E_UUU)** |
| Train | `train/train_triple.py` | `--progress-w` (default 1.0 product); `--flip-augment` (Baek VER, default on); `--eval-curriculum` → `eval/near_target/*` + `eval/hang/*` |
| Launch | `scripts/next-train-triple.sh` | Passes `PROGRESS_W` / `FLIP_AUGMENT` |
| Slots | `scripts/continue-triple-{a,b,c}.sh` | **A** swing (hang+near, f50, progress+flip); **B** hold (near_target, f40); **C** combo (near+hang0.3, progress+flip). Markers `.triple-*-progress-flip-v5` / hold / combo |

**Symmetry (Baek VER / `--flip-augment`):** planar reflect across the vertical midline maps `(x,ẋ,θᵢ,θ̇ᵢ,F)→(−x,−ẋ,−θᵢ,−θ̇ᵢ,−F)`. Dynamics + product reward are equivariant; θ*∈{0,π} goal encodings invariant. After GAE, PPO batch is duplicated with flipped obs/raw and recomputed logπ.

**Eval meters:** keep harsh `eval/*` (random ICs). Also log `eval/near_target/at_goal/UUU`, `eval/near_target/align/UUU`, `eval/hang/*` — diagnose hold vs swing with the right meter.

**Slot policy:** **Kill double/xonly.** All 3 L4 slots on `cartpole-train-od` are triple-a/b/c only.

**Cold start:** new progress reward → wipe once via `.triple-a-progress-flip-v5`, `.triple-b-hold-progress-flip-v5`, `.triple-c-combo-progress-flip-v5`.

**Launch:**
```bash
NUM_ENVS=8192 FORCE_LIMIT=40 PROGRESS_W=1.0 FLIP_AUGMENT=1 bash scripts/next-train-triple.sh
# Specialists: bash scripts/next-train-triple-{a,b,c}.sh
# A/B/C on VM: continue-triple-{a,b,c}.sh
```


## Overnight fire — status (2026-09-19 ~15:32 CT)

**VM:** `cartpole-train-od` RUNNING us-east1-b L4 ~99%/15.7GB; TB http://34.148.138.48:6006/ up; **no double/xonly**. Uptime ~36.6h ≈ **~$26** @~$0.70–0.71/hr (≤$30; ~4–5h headroom to u400). continue-triple-a/b/c armed since ~14:36 CT. GPU healthy.

**Jobs (alive, progress+flip v5):**
| Slot | Recipe | ~u | rollout_r | entropy | goal_frac/UUU | near_target at_goal/UUU | near_target align/UUU | hang align/UUU |
|---|---|---|---|---|---|---|---|---|
| A | swing hang+near f50 bar10 e0.5 prog1+flip | ~100 | ~0.45 | **~−0.12** ↓ | ~0.93 | ~0.039 | ~+0.153 (was +0.18@u80) | ~−0.089 |
| B | hold near_target f40 bar10 e0.15 prog1+flip | ~100 | ~0.51 | ~+0.011 | ~0.92 | ~0.043 | ~+0.088↑ | ~−0.34↑ |
| C | combo hang0.3 f40 bar10 e0.35 prog1+flip | ~90–98 | ~0.50 | ~+0.072 | ~0.93 | ~0.045 | ~+0.135↑ | ~−0.22 |

**Diagnosis:** Mid v5 stretch. **A entropy crossed negative** (u50 +0.32 → u80 +0.04 → u101 **−0.12**) — same class of collapse that previously hit B on hot-lr; A is already lr3e-4 so not the same lever. near_target UUU align on A dipped slightly u80→u100. **B/C entropy still non-negative**; B/C near_target UUU align still climbing slowly. Harsh `eval/at_goal/UUU` still ~0 (expected). oob≈0. No NaNs / no plant patch / no restart this fire. Force 40–50 + barrier10 still OK vs Glück/Lim (no rail-slide). Stage-gate still UUU hold ≳0.80 before multi-eq.

**Watch next:** If A entropy stays ≤−0.1 past u150 with flat near_target at_goal, plan cooler/explore retune at u400 (not mid-run kill). u400 ETA ~18:20 CT (~$28). NEED_USER_PING no.

## Overnight fire — status (2026-09-19 ~15:06 CT)

**VM:** `cartpole-train-od` RUNNING us-east1-b L4 ~99%/15.7GB; TB http://34.148.138.48:6006/ up; **no double/xonly**. Uptime ~36.2h ≈ **~$25–26** @~$0.70–0.71/hr (≤$30; ~5–6h headroom to u400). continue-triple-a/b/c armed since ~14:36 CT. GPU healthy.

**Jobs (alive, progress+flip v5):**
| Slot | Recipe | ~u | rollout_r | entropy | goal_frac/UUU | near_target at_goal/UUU | near_target align/UUU | hang align/UUU |
|---|---|---|---|---|---|---|---|---|
| A | swing hang+near f50 bar10 e0.5 prog1+flip | ~53 | ~0.45↑ | ~0.28 | ~0.93 | ~0.038 | **~+0.171** | ~−0.09↑ |
| B | hold near_target f40 bar10 e0.15 prog1+flip | ~54 | ~0.50↑ | ~0.31 | ~0.93 | ~0.035 | ~+0.049 | ~−0.39↑ |
| C | combo hang0.3 f40 bar10 e0.35 prog1+flip | ~50 | ~0.47↑ | ~0.44 | ~0.92 | ~0.045 | **~+0.102** | ~−0.21↑ |

**Diagnosis:** Still early/mid v5 warmup — healthy, no action. Entropy cooling but **positive** (no B-style collapse). **A** still leads curriculum (near_target align/UUU +0.12→**+0.17**); hang align recovering toward zero on A/C. Harsh `eval/at_goal/UUU` still ~0 (expected). oob≈0. No NaNs / no plant patch / no restart. Stage-gate still UUU hold ≳0.80 before multi-eq. Force 40–50 + soft barrier still OK vs Glück/Lim notes — not underpowered yet (rail slide / flat UUU would trigger probe).

**Watch next:** near_target at_goal/UUU + hang align toward u80–150; entropy floor; u400 auto-continue ETA ~16:40–17:00 CT (~$27–28). NEED_USER_PING no.

## Overnight fire — status (2026-09-19 ~14:51 CT)

**VM:** `cartpole-train-od` RUNNING us-east1-b L4 ~99%/15.3GB; TB http://34.148.138.48:6006/ up; **no double/xonly**. Uptime ~36h ≈ **~$25–27** @~$0.70–0.75/hr (≤$30; ~4–5h headroom). continue-triple-a/b/c armed. Progress+flip v5 still cooking (cold-started ~14:36 CT).

**Jobs (alive):**
| Slot | Recipe | ~u | rollout_r | entropy | goal_frac/UUU | near_target at_goal/UUU | near_target align/UUU | hang align/UUU |
|---|---|---|---|---|---|---|---|---|
| A | swing hang+near f50 bar10 e0.5 prog1+flip | ~28 | ~0.39↑ | ~0.66 | ~0.93 | ~0.040 | **~+0.118** | ~−0.15↑ |
| B | hold near_target f40 bar10 e0.15 prog1+flip | ~30 | ~0.46↑ | ~0.59 | ~0.92 | ~0.035 | ~+0.001 | ~−0.60↑ |
| C | combo hang0.3 f40 bar10 e0.35 prog1+flip | ~26 | ~0.43↑ | ~0.78 | ~0.92 | ~0.036 | **~+0.087** | ~−0.30↑ |

**Diagnosis:** Early but healthy. Entropy still non-negative (no B-style collapse). **A** near_target align/UUU already ~+0.12 @u20 — best early curriculum signal vs prior UUU-only plateau. Hang align climbing on all slots (A/C fastest). Hold at_goal still ~0 (expected). No NaNs / no plant patch / no restart this fire. Stage-gate still UUU hold ≳0.80 before multi-eq.

**Watch next:** near_target at_goal/UUU toward u80–150; entropy floor; budget before u400 auto-continue (~ETA ~16:40 CT).

## Overnight fire — status (2026-09-19 ~14:40 CT)

**VM:** `cartpole-train-od` RUNNING us-east1-b L4 ~99%/13.6GB; TB http://34.148.138.48:6006/ up; **no double/xonly**. Uptime ~35.7h ≈ **~$25** @~$0.70/hr (≤$30; ~5–6h headroom). continue-triple-a/b/c armed. Box/VM HEAD `78d9c4f` hashes match.

**Jobs (alive, progress+flip v5 cold-started ~14:36 CT):**
| Slot | Recipe | ~u | rollout_r | entropy | goal_frac/UUU | near_target at_goal/UUU (u1) |
|---|---|---|---|---|---|---|
| A | swing hang+near f50 bar10 e0.5 prog1+flip | ~8 | ~0.17↑ | ~1.26 | ~0.92 | ~0.038 |
| B | hold near_target f40 bar10 e0.15 prog1+flip | ~7 | ~0.31↑ | ~1.30 | ~0.92 | ~0.034 |
| C | combo hang0.3 f40 bar10 e0.35 prog1+flip | ~7 | ~0.28↑ | ~1.30 | ~0.92 | ~0.036 |

**Diagnosis:** Fresh P0/P1 redeploy cooking. Entropy healthy (no B-style collapse). Curriculum meters live. Early UUU hold still ~0 (expected); watch `eval/near_target/at_goal/UUU` + hang meters toward u80–150 before retune. No NaNs / no plant patch / no restart this fire. Stage-gate still UUU hold ≳0.80 before multi-eq.

**Watch next:** progress reward + flip should lift near-target UUU vs prior plateau; if flat past u150, probe energy/force.

## Overnight fire — status (2026-09-19 ~14:20 CT)


**VM:** `cartpole-train-od` RUNNING L4 ~99%/5.4GB; TB http://34.148.138.48:6006/ up; **no double/xonly**. Uptime ~35.4h ≈ **~$25–26.5** @~$0.70–0.75/hr (≤$30; ~5–6h headroom). continue-triple-a/b/c alive.

**Jobs (alive):**
| Slot | Recipe | ~u | reward | align/UUU | at_goal/UUU | entropy | rollout_r |
|---|---|---|---|---|---|---|---|
| A | UUU-only bottom f40 bar10 e05 hang1.0 | ~329 | ~211 | **~+0.021** | ~0.004 | ~0.21 | ~0.48 |
| B | UUU-only wide f60 bar10 e035 lr5e-4 | ~324 | ~221 | ~−0.019 | ~0.002 | **~−0.41** | ~0.52 |
| C | UUU swing near_target hang0.3 f40 | ~370 | ~218 | ~−0.062 | ~0.002 | ~0.14 | ~0.52 |

**Diagnosis:** Near stretch end. **A** still healthiest (UUU align positive, entropy stable). **B** entropy still dead (~−0.41) — **cooler-lr retune decided**: rewrite continue-b to `LR=3e-4`, cold wipe via `.triple-b-uuu-wide-f60-lr3e4-v4`, restart watcher so auto-continue after u400 does not stack another hot-lr stretch. **C** UUU align dipped (−0.03→−0.06) near finish; leave recipe alone (auto-continue). No NaNs / no plant patch. Stage-gate still UUU hold ≳0.80 before multi-eq. ETA C ~14:27 CT, A/B ~14:34 CT.

**Paper skim (stuck UUU):** Existing notes already cover it — fawraw stage-gate ≥0.80, energy swing-up as Plan-B if probes fail, Glück ~22 m/s² vs our f40/f60. No new paper action this fire; force+barrier already addressed; B was lr pathology not plant.

**Watch next:** B finish → cold cooler-lr restart; A/C auto-continue; promote if any UUU hold appears.

## Overnight fire — status (2026-09-19 ~14:00 CT)

**VM:** `cartpole-train-od` RUNNING L4 ~99%/5.4GB; TB http://34.148.138.48:6006/ up; **no double/xonly**. Uptime ~35.1h ≈ **~$24.5–26.5** @~$0.70–0.75/hr (≤$30; ~5–7h headroom). continue-triple-a/b/c alive.

**Jobs (alive):**
| Slot | Recipe | ~u | reward | align/UUU | at_goal/UUU | entropy | rollout_r |
|---|---|---|---|---|---|---|---|
| A | UUU-only bottom f40 bar10 e05 hang1.0 | ~250 | ~214 | **~+0.030** | ~0.003 | ~0.23 | ~0.46 |
| B | UUU-only wide f60 bar10 e035 lr5e-4 | ~250 | ~222 | ~−0.066 | ~0.001 | **~−0.39** | ~0.51 |
| C | UUU swing near_target hang0.3 f40 | ~300 | ~222 | ~−0.034 | ~0.002 | ~0.11 | ~0.51 |

**Diagnosis:** Incremental vs 13:48 — still cooking to u400. **A** still best (align/UUU climbed slightly +0.015→+0.03). **B** entropy still collapsed (~−0.39) and UUU align worsened (−0.04→−0.07); no mid-run kill (prior plan). **C** ~u300/400, flat UUU. Eval at_goal/UUU still ~0 (harsh `random_states`; expected). No NaNs / no restarts / no plant patch. Stage-gate still UUU hold ≳0.80 before multi-eq. ETA A/B finish ~14:40 CT, C ~14:25 CT — then decide B cooler-lr retune vs continue.

**Watch next:** finishes + B entropy decision; budget before auto-continue stacks another 400.

## Overnight fire — status (2026-09-19 ~13:48 CT)

**VM:** `cartpole-train-od` RUNNING L4 ~99%/5.4GB; TB http://34.148.138.48:6006/ up; **no double/xonly**. Uptime ~35h ≈ **~$24–28** @~$0.70–0.75/hr (≤$30; ~3–5h headroom). All three continue-a/b/c watchers alive.

**Jobs (alive):**
| Slot | Recipe | ~u | reward | align/UUU | at_goal/UUU | entropy |
|---|---|---|---|---|---|---|
| A | UUU-only bottom f40 bar10 e05 hang1.0 | ~210 | ~216 | ~+0.015 | ~0.004 | ~0.25 |
| B | UUU-only wide f60 bar10 e035 lr5e-4 | ~210 | ~223 | ~−0.041 | ~0.001 | **~−0.39** |
| C | UUU swing near_target hang0.3 f40 | ~260 | ~220 | ~−0.045 | ~0.003 | ~0.14 |

**Diagnosis:** Still mid UUU-only stretch. `train/goal_frac/UUU`≈0.93; `train/rollout_reward` climbing (A~0.46 B/C~0.51). Eval `at_goal/UUU` still flat (expected — harsh `random_states`). **A** remains healthiest (align/UUU non-negative). **B** entropy collapsed negative (hot lr 5e-4) + UUU align slightly worse — watch at u400 finish; no mid-run kill. **C** entropy low but stable; UUU align noisy/negative. No NaNs. **No restart/patch this fire** — ETA ~u400 A/B ~14:40 CT, C ~14:25 CT. Stage-gate still UUU hold ≳0.80 before multi-eq.

**Watch next:** B finish → decide keep vs cooler-lr retune; budget before stacking more stretches; promote best of A/C if any hold appears.

## Overnight fire — status (2026-09-19 ~13:36 CT)

**VM:** `cartpole-train-od` RUNNING L4 ~99%/5.4GB; TB up; **no double/xonly**. Spend ~35h up ≈ **~$24–28** (≤$30; ~5h headroom @~$0.75/hr).

**Jobs (all alive + continue-a/b/c):**
| Slot | Recipe | ~u | reward | align/UUU | at_goal/UUU |
|---|---|---|---|---|---|
| A | UUU-only bottom f40 bar10 e05 hang1.0 | ~165 | ~214 | ~+0.015 | ~0.004 |
| B | UUU-only wide f60 bar10 e035 lr5e-4 | ~163 | ~220 | ~−0.030 | ~0.002 |
| C | UUU swing near_target hang0.3 f40 | ~214 | ~219 | ~−0.017 | ~0.001 |

**Diagnosis:** UUU-only cold starts (~12:53 CT) still early. `train/goal_frac/UUU`≈0.93 (curriculum OK). Eval `at_goal/UUU` flat is **expected**: `rollout_eval` uses `random_states(mild=False)`, not train init — so it understates near-basin progress. `train/rollout_reward` slowly climbing (~0.46–0.51). **No restart/patch this fire** — let A/B/C cook toward u400; stage-gate still UUU hold ≳0.80 before multi-eq.

**Watch next:** A align/UUU staying non-negative; C train reward vs eval gap; budget before u400 finishes.

## Overnight fire — A/B stage-gate restart (2026-09-19 ~12:50 CT)

**Diagnosis (~12:48 CT):** A (f40 bar50) ~u190–200 reward~215 at_goal/UUU≈0.0001 align/UUU≈−0.27 (DDD preferred). B (f40 bar10) ~u200 reward~226 at_goal/UUU≈0.0006 align/UUU≈−0.22. Both left UUU warmup at u80 with **zero hold** — violates fawraw stage gate ≥0.80. force40 alone did not unlock UUU; soft barrier on B insufficient. C (UUU swing retune) ~u40 at_goal/UUU≈0.004 — left alone.

**Fix:** rewrite A/B to **UUU-only forever** (WARMUP_UPDATES=100000, no goal switch/fold), cold wipe via new markers:
- **A** `.triple-a-uuu-bottom-v3`: `INIT_MODE=bottom`, hang=1.0, FORCE=40, bar=10, ENERGY_W=0.5, start_grace=40 → `ft-triple-a-…-uuu-bottom-f40-bar10-e05`
- **B** `.triple-b-uuu-wide-f60-v3`: `INIT_MODE=wide`, hang=0.3, FORCE=60, bar=10, ENERGY_W=0.35, NEAR_GOAL_P=0.25, LR=5e-4 → `ft-triple-b-…-uuu-wide-f60-bar10-e035-lr5e4`

Do not open multi-eq until UUU hold ≳0.80.

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

- **(Updated 2026-09-19)** Double **xonly killed** — all 3 L4 slots on triple-a/b/c while diagnosing UUU underpower / forceLimit.
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

**Green-lit 2026-09-19:** product/UUU-first/grace/barrier landed on main; **xonly killed** same day for forceLimit=40 triple A/B/C.


### Research pass (2026-09-19 ~10:31 CT) — new actionable diffs

Re-read Lim PDF end-to-end, fawraw README/CHANGELOG/`docs/m4_findings.md`, Glück post-print, and the rotary/double product-reward cousins. **Direction unchanged** (UUU-first → product/abs → hard-EP → later 56). Four new levers + cookbook fills to fold in when coding.

#### fawraw process knobs we had only half-locked

| Knob | Value | Why it matters |
|---|---|---|
| Soft fall grace | `fall_grace_steps≈20` | Swing-up with `start≠target` used to die at step 0/1 on the angle-fall check (−100). Grace lets the cart inject energy before fall-terminate. |
| Stage gates | M2 UUU success **≥0.80** before multi-eq; M3 overall **≥0.75** before 56 | Concrete stop rules so we do not open the next stage on a weak attractor. |
| Catch-basin measure | Sweep offset×vel on the stabilizer *before* hand-off | M3@UDD grid (fawraw): reliable only at **0.1 rad / 0 vel** (1.0); 0.2 rad @0 vel ≈0.6; **any vel ≥2 → 0**. Soft-landing must hit that basin. |
| Init for swing-up | `init_mode=bottom` (not `near_target`) when start≠target | Pairs with grace; near_target + wrong start is the step-1 death bug. |

#### Lim TQC cookbook fills (Table 1 + §4.2–4.3)

Already had product coeffs / ICs / 8 specialists. Newly locked:

- Optimizer ADAM; **γ=0.99**; target-smooth **β/τ=0.005**; target update every step; 1 grad step / 1 env step; ReLU.
- Specialist “done” diagnostic: ep return plateaus ~**700–800 / 1000** (not 1000) under wide random ICs — expect residual exploration noise.
- Wide random ICs can spawn **physically impossible** state combos → early-term noise; filter or clamp if adopting Lim ranges.
- Early-stop already locked: `|y|>0.48` m or `|a|>2.5` m/s²; ODE 1 ms / agent 10 ms / ep 10 s.

#### Product-reward lineage (double / rotary cousins → prefer Lim triple form)

| Source | \(R_u\) | Cart / vel notes |
|---|---|---|
| MDPI Machines 2025 (cart double, same Inha lab) | \(\exp(-0.015\|u\|)\) | \(R_y=\exp(-0.5\|y\|)\); per-link \(R_{\dot\theta}=\exp(-0.02\|\omega\|)\); 4 specialists → 12 transitions |
| em0sh/rdip (rotary double TQC reimpl) | \(\exp(-0.005\|u\|)\) | Same 10 s / 10 ms skeleton; product of angle+rate terms |
| Lim KIEE 2025 (cart triple) | \(\exp(-0.001 u^2)\) | Softer input; \(R_y=\exp(-0.3\|y\|)\); **cumulative** world angles + cumulative rates |

**Steal Lim’s triple form**, not the double’s harsher \(R_u\)/`R_y`. Confirms product + specialists transfer down the lab lineage; cumulative abs angles are the triple-specific upgrade.

#### Glück Automatica 2013 (classical — time/accel budget only)

Confirmed post-print numbers (already roughly noted): swing-up **T = 3.5 s** under box constraints \(|s|\le 0.7\) m, \(|\dot s|\le 3\) m/s, \(|\ddot s|\le 22\) m/s². Useful as a **horizon / actuator budget** check for our RL episodes, not a feedforward path. (fawraw’s README table attributing “Graichen Automatica 2013 / 56 trajectories” is a mis-cite — Glück is DDD→UUU only; Graichen CDC 2005 is side-step.)

#### Amend recommended redesign (additions only)

15. On any swing-up / transition episode: **`fall_grace_steps≈20`** + `init_mode=bottom` when start≠target.
16. **Stage gates:** do not leave UUU-only until hold success ≳0.80; do not open 56 until multi-eq overall ≳0.75.
17. Before any hand-off / transition-only phase: **measure catch basin** (offset×vel grid) on the hold policy; train wider-basin catcher and/or soft-landing until delivery lands inside it.
18. When porting Lim product+TQC (or PPO surrogate): copy **γ=0.99, τ=0.005**; judge EP specialists by ~700–800/1000 return plateau + hold metrics, not max return.
19. Prefer **Lim triple product** (cumulative abs angles) over MDPI-double coeffs if we A/B product forms.

**Green-lit 2026-09-19:** product/UUU-first/grace/barrier landed on main; still do not kill double xonly.


### Research pass (2026-09-19 ~11:06 CT) — new actionable diffs

Re-read fawraw `triple_pendulum_env.py` + CHANGELOG / M3b-v4–v7 configs, Glück post-print overshoot note, and BC→RL dead-ends. **Direction unchanged** (UUU-first → product/abs → hard-EP → later 56). Five env/process knobs we had not locked, plus one classical sensitivity fill.

#### fawraw per-link fall thresholds (CRITICAL for tip-up EPs)

Global angle-fall of **0.6 rad on every link** made EP4 (DDU) / EP6 (DUU) untrainable: cart recovery shakes hanging links past 0.6 → −100 FALL every attempt → 0% on tip-up EPs. Fix (audit 2026-05-10):

| Link target | Threshold | Why |
|---|---|---|
| UP (θ*≈0) | **`FALL_THRESHOLD_UP_RAD = 0.6`** (~34°) | Must stay near vertical |
| DOWN (θ*≈π) | **`FALL_THRESHOLD_DOWN_RAD = 1.5`** (~86°) | Hang links may swing during recovery |

`_fall_thresholds()` picks per link from current `target_ep`. Our overnight plant only oob-terminates on `|x|>track` (no angle-fall) — so this matters the moment we add hold-style fall checks for tip-up / multi-eq stages. Do **not** copy a global 0.6.

#### `vel_cost_coef`: default 0.05 over-damps UUU / tip-up

| Value | Where used | Effect |
|---|---|---|
| **0.05** | env default (and early bump from 0.01) | Suspected **EP7 (UUU) regression** — over-penalizes aggressive corrections upright configs need |
| **0.01** | v4H velfix; v5 phase1 hard-EP focus | EP7-friendly; tip-up needs cart motion |
| **0.02** | M3b-v6 cloud winner + v7 + most M4 probes | Compromise that shipped with 72.5% |

**Steal 0.01–0.02** for any additive ang²+vel² stage; never leave the env default 0.05 if UUU / tip-up are soft.

#### `start_grace_steps` ≠ `fall_grace_steps`

Already locked `fall_grace≈20` (consecutive-over-threshold before −100). Newly locked sibling:

- **`start_grace_steps`**: first N steps **immune** to angle-fall so the policy can orient (configs: **100** ≈2 s @50 Hz in v4C/F/H; 0 in strict eval).
- Eval always keeps both graces at 0 for fair scoring; train may use grace, eval must not inherit it.

#### `w_down` tradeoffs (refine item 9)

UP×5 is locked. DOWN weight is not free:

| `w_down` | Observed |
|---|---|
| 1.0 | Adaptive “correct gradient”; can **regress EP2 (UDD)** (lost base-stability prior) |
| 2.0 | Helps EP2; **regresses EP7** in A/B |
| **1.5** | v5 compromise between those two |

Prefer **1.0 first** with hard-EP oversample; bump to 1.5 only if base-down EPs collapse; avoid 2.0 unless UUU is already solid and EP2 is dead.

#### BC → multi-EP RL is a dead end (reinforces Lim specialists)

fawraw Plan B / Stage3A: BC-only [1024,1024] **12.5%**; BC-then-RL pollutes shared weights (covariate shift off LQR demos). EP4 **specialist** alone caused catastrophic forgetting of other EPs. **Dedicated tip-up probe** (EP4-fixed) did reach **~60% @200K** before mixing. Implication for us: do **not** BC-pretrain a shared UVFA from double/LQR demos; prefer Lim-style **8 heads / 8 runs**, or UUU-only → tip-up probe → weighted multi-eq. Matches recommended redesign items 1/5/12.

#### Glück Automatica 2013 — sensitivity fill

Already had T=3.5 s, |ÿ|≤22 m/s², |s|≤0.7 m, |ṡ|≤3 m/s. Post-print adds: even **tiny** angle/rate tracking errors → cart **~0.6 m overshoot**. Classical DDD→UUU is hypersensitive to residual error — another argument for **soft-landing + catch-basin** before 56 hand-off, and for sizing track / center cost so RL swing-ups have headroom comparable to that overshoot.

#### Amend recommended redesign (additions only)

20. If/when adding angle-fall termination: **per-link UP=0.6 / DOWN=1.5** — never a global 0.6 (kills tip-up EPs).
21. Additive vel penalty: **`vel_cost_coef≈0.01–0.02`** (not 0.05).
22. Pair `fall_grace≈20` with **`start_grace_steps≈100`** in train; keep eval strict (both 0).
23. `w_down`: start at **1.0**; try **1.5** only if EP2-class eqs die; avoid 2.0 early.
24. Skip BC-pretrain of a shared triple UVFA; use **fixed-EP tip-up probes** (~200K) before hard-EP mix; keep Lim-style specialists as the safe multi-eq structure.
25. Size track / soft-landing with Glück’s **~0.6 m** classical overshoot sensitivity in mind.

**Green-lit 2026-09-19:** product/UUU-first/grace/barrier landed on main; still do not kill double xonly.


### Research pass (2026-09-19 ~11:33 CT) — new actionable diffs

Re-checked fawraw main (commits through **2026-07-02**, last *code* **2026-06-25**), `sim/handoff.py` + `docs/m4_findings.md` + M3b-v6/v7 yamls, and adjacent 2025–2026 papers (arXiv:2606.22145 single cart-pole handoff; Cambridge Robotica 2026 already noted; no new cart-triple/56 paper beyond Lim). **Direction unchanged** (UUU-first → product/abs → hard-EP → later 56). Fills for the post-UUU UVFA hard-EP stage + catch-basin/56 plumbing; still no energy coeffs and no PPO+product cookbook.

#### Hard-EP oversample for a *shared* multi-EP policy (UVFA-shaped; fills gap 1)

fawraw's M3 breakthrough is **one** conditional TQC (not Lim's 8 specialists) with `target_mode=weighted`. Exact sampler (`triple_pendulum_env.py`):

```text
w = ones(8); w[4] = w[6] = hard_ep_weight; p = w / w.sum()
# EP4=DDU (tip-up), EP6=DUU only — never blanket-boost all non-UUU
```

| `hard_ep_weight` | P(EP4)=P(EP6) | P(each other EP) | Role |
|---|---|---|---|
| **10** | ≈38.5% | ≈3.8% | Prefer first if UUU regresses (v7 intent) |
| **20** | ≈43.5% | ≈2.2% | Cloud winner phase1 (~46% in their writeup) |
| **2.5** | ≈25% | ≈8.3% | **Required consolidation** after focus |

**UVFA PPO schedule to steal after UUU holds ≳0.80** (our `train_triple` still has no `hard_ep_weight` — next code lever, not overnight):

1. Warm-start the UUU/multi-eq product ckpt.
2. Phase1 focus: `hard_ep_weight=10` (escalate to 20 only if tip-up EPs stay ~0), `init_mode=near_target`, `init_noise=0.05`, lr≈1e-4 (or our PPO equivalent), ~600K-env-steps worth of updates.
3. Phase2 consolidate: **`hard_ep_weight→2.5`**, lr≈5e-5, ~500K — without this, UUU/easy EPs regress (v6: EP7 80%→40% under weight=20 alone).
4. Watch per-EP hold; do not open 56 until overall ≳0.75.

This is the closest published **shared-policy** hard-EP recipe to our UVFA; Lim's 8×TQC remains the fallback if UVFA+weight still starves tip-up.

#### Soft-landing / catch-basin — concrete next knobs, still no reward coefs (fills gap 2)

fawraw has **not** shipped soft-landing coefficients (plan only). What *is* newly lockable from `m4_findings.md` + `handoff.py`:

| Knob | Value | Why |
|---|---|---|
| Wider-basin catcher init (do this *before* inventing soft-landing coefs) | Raise `init_noise` ≫0.05; add **non-zero link velocities**; **off-centre cart** | M3@UDD catches only ≤0.1 rad / ~0 vel; swing-up delivers ~0.2 rad mid-swing |
| Handoff capture gate | Set `capture_tol_rad` ≤ **measured** basin (~**0.1**), **not** the code default **0.3** | Default 0.3 fires hand-off then stabilizer drops the delivery |
| Velocity gate | Optional `capture_vel_rad_s` (near 0) | Basin grid: any |ω|≳2 → catch rate 0 |
| Latch | `latch=True` | Prevents flip-flop back to swing-up on a near-miss |
| Transition success (56 eval) | **0.2 rad** tol × **200** consecutive steps; sparse `transition_bonus=200` | fawraw M4 env defaults — hold, not touch-and-go |
| Soft-landing reward | Still unspecified ("arrive slow + cart-centred") | No numbers yet; prefer wider catcher + progress/barrier first |

#### 8 specialists vs one UVFA (gap 3) — still no PPO head-to-head

- Lim: **8×TQC** → all 56 on hardware.
- fawraw: **1 shared** conditional TQC + hard-EP → 72.5% on 8 holds (not 56).
- No new 2025–2026 evidence that 8 PPO specialists beat one UVFA (or vice versa) on cart-triple. Keep UVFA+hard-EP as default; specialists if tip-up stays dead after weight=20 + consolidate.

#### Gaps that remain empty

- **Energy-based swing-up coeffs for cart-triple:** still none (fawraw lists it as Plan-B if probes fail; H-EARS / Xin–Spong are wrong plant or abstract-only).
- **PPO+product hypers** (n_envs / rollout / clip / lr schedule): nothing published on underactuated multi-link with product reward — keep our double PPO defaults until A/B.
- **fawraw since mid-2026:** last *code* **2026-06-25** (handoff + barrier + catch-basin tool); **2026-07-02** docs-only. No M4 soft-landing implementation, no new configs, M4 still gated.

#### Adjacent paper (later 56 / sim2real only)

arXiv:2606.22145 (2026) — *single* cart-pole swing-up↔stabilize zero-shot: separate policies + handoff; discrete action LPF **τ=0.3**; sensitivity-guided DR + linear CL. Steal the **LPF / bumpless switch** idea for a future 56 hand-off, not for the current UUU/product stage.

#### Amend recommended redesign (additions only)

26. After UUU holds: implement **weighted EP sampling** (boost **EP4+EP6 only**) with the exact `w/sum(w)` formula; run **focus (w=10→20) then consolidate (w=2.5, lower lr)** — do not stay at weight=20.
27. Before soft-landing coef hunting: **widen the catcher** (larger `init_noise`, nonzero ω, off-centre x) and measure basin again.
28. Handoff gate ≤ measured basin (~0.1 rad) + optional vel gate + **latch**; never ship the 0.3 default against a 0.1 basin.
29. 56 success metric: **0.2 rad × 200 steps** (+ optional bonus 200); require hold after arrival.
30. Optional later: action LPF **τ≈0.3** on swing-up (arXiv:2606.22145) for smoother hand-off.

**xonly killed 2026-09-19 ~11:45 CT** so all 3 L4 slots run triple (force40 A/B/C). Next code lever after UUU holds is hard-EP weighting + consolidate, not a redesign.


## Force / actuation diagnosis (2026-09-19 ~11:45 CT)

| Quantity | Value | Implication |
|---|---|---|
| `constants-triple.json` `forceLimit` (old) | **20.0 N** | Hard `tanh` action cap |
| `cartMass` | 1.0 kg | Peak cart accel ≈ force/mass ≈ **20 m/s²** |
| Glück Automatica 2013 swing-up | ~**22 m/s²** cart accel | Classical DDD→UUU already at/above our old cap |
| Multi-arm / lab cart-triple sizing | ~110 N continuous / ~20 m/s² on heavier carts | Same accel ballpark; our 20 N @ 1 kg matches accel but **friction 0.08** + 3×0.5 m links eat budget |
| Product `R_u` | Soft `exp(-k u²)` | Prefer gentle force; does **not** raise the hard ceiling |
| Cart barrier | `(x/track)^8 * coef` | coef=50 fights rail-run-up needed for energy pump |

**Decision:** bump default `forceLimit` → **40.0** (optional `--force-limit` / `FORCE_LIMIT` / `constants-triple-force40.json`). Keep barrier=50 on **triple-a**; drop to **10** on **triple-b** so the cart can swing. **triple-c** = UUU-only specialist (`warmup_updates=100000`, `hang_start_p=0.05`, `near_goal_p=0.3`).


## Strategy research — what to implement next (2026-09-19 ~14:30 CT)

Live symptom: even UUU-only + force40/60, `at_goal/UUU≈0` and align/UUU often ≤0. Eval uses random ICs (harsh); train may still not be pumping energy.

### Ranked strategies that worked elsewhere

1. **Baek et al. EAAI 2024 (hardware TIP swing-up)** — **SAC/off-policy** + **product reward**  
   `R = f(a)·g(x)·h(θ1)·h(θ2)·h(θ3)·min(e(ω))` with α floors so one term can’t zero the product.  
   **VER (virtual experience replay):** mirror trajectories left↔right using geometric symmetry → ~⅔ fewer samples. Ports to PPO as a **rollout augmenter** (duplicate flipped obs/actions/rewards).

2. **Two-policy handoff (cart-pole 2026 arXiv + fawraw M4)** — train **swing-up** and **stabilize** separately; switch when state enters catch basin (~0.1 rad, near-zero ω). Our single PPO tries both and fails both. Highest-ROI architecture change if UUU-only stays flat.

3. **Energy-based swing-up (Xin double-cart analysis)** — pump total energy toward E(UUU), then local capture. Almost-global for double; for triple, classical uses **feedforward trajectory + TV-LQR** (Glück), not pure energy. Still: add an **explicit energy-to-UUU term** stronger than our current 0.35 additive bonus inside product.

4. **Progress shaping + cart barrier (fawraw)** — dense Δ(angle-error) reward; barrier stops rail-slide local optima. We have barrier; we may lack strong **progress** (derivative of |θ−θ*|).

5. **Curriculum SAC (Cambridge Robotica 2026 UTPR)** — adaptive CL easy→hard + quadratic+integral angle error for hold. Good for **stabilize** phase after swing-up.

6. **LQR-trees / funnel capture (double)** — covers many ICs with trajectory tree. Heavy; only if we leave pure end-to-end RL.

### Recommended implementation order (keep PPO for now)

| Priority | Change | Why |
|---|---|---|
| P0 | **Train-eval fix**: log `at_goal` from *near_target / hang* starts, not only random ICs | We’re diagnosing with the wrong meter |
| P0 | **Progress reward** Δ cos-align toward UUU | Baek/fawraw dense swing signal |
| P1 | **Left–right VER / flip augment** in PPO rollouts | Baek’s big sample-efficiency win |
| P1 | **Split swing vs hold** (two nets or two heads) + handoff | Matches every successful hardware story |
| P2 | **True mechanical E→E_UUU** (not crank height-proxy `energy_w`); SAC/TQC cookbook; demote force 80–100 unless rail-slide | Energy gap filled below; force already ≥ Glück accel |
| P2 | Off-policy **SAC/TQC** side experiment (one slot) | Papers that hit hardware used SAC/TQC not PPO |
| P3 | Multi-eq only after UUU hold ≥0.8 | Stage gate |

### Not the bottleneck (already tried)

- force 20→40 alone (didn’t unlock UUU)
- Soft vs hard barrier alone
- Hang-heavy multi-eq UVFA (DDD sink)


### Research pass (2026-09-19 ~15:14 CT) — energy / force / SAC cookbook fills

Digged Spong/MIT energy shaping, Xin parallel-pendulum energy papers, Baek EAAI force bound, Lim/Baek SAC hypers, IROS'24 SAC punishment reward (arXiv:2410.20096), SAC→LQR handoff (arXiv:2312.11311), and re-checked fawraw M4 (still 2026-06-25). **Direction unchanged.** P0 progress + P1 flip are **already live** on v5; top *unimplemented* lever remains **two-policy swing↔hold handoff**. No user ping (refines existing P1/P2; no new #1 lever).

#### `energy_w` ≠ E→E_UUU (fills long-empty gap)

Under product mode, `--energy-w` adds **mean cos-align / height proxy**, not mechanical energy error. Cranking it does **not** implement classical energy swing-up.

**Implementable proxy for our plant** (`θ=0` upright, equal links \(m=0.1\), \(L=0.5\), \(l_c=L/2\), \(g=9.81\), cart \(m_c=1\)):

- World angles (Lim): \(\phi_1=\theta_1\), \(\phi_2=\theta_1+\theta_2\), \(\phi_3=\theta_1+\theta_2+\theta_3\)
- Potential: \(U = \sum_i m_i g l_{c,i}\cos\phi_i\) → \(U_{\mathrm{UUU}} \approx +0.736\,\mathrm{J}\), \(U_{\mathrm{DDD}} \approx -0.736\,\mathrm{J}\), \(\Delta U \approx 1.47\,\mathrm{J}\)
- Kinetic \(T\): cart \(\tfrac12 m_c\dot x^2\) + link terms from `physics_triple` mass matrix (or reuse existing batched energy if exposed)
- Target: \(E_{\mathrm{UUU}} = U_{\mathrm{UUU}}\) at \(\omega=\dot x=0\)
- RL shaping candidates: \(r_E = -w_E |E - E_{\mathrm{UUU}}|\) or progress \(\Delta\) toward \(E_{\mathrm{UUU}}\); near upright add soft-landing \(-w_\omega\|\omega\|\)

Classical Spong/MIT single cart-pole (reference only): after collocated PFL, \(E=\tfrac12\dot\theta^2-\cos\theta\), \(E^d=1\), \(u = k_E\dot\theta\cos\theta\,\tilde E - k_p x - k_d\dot x\). **Xin CDC 2009** energy control is for **parallel** n-pendulums on a cart — still **no published serial cart-triple energy coefficients**. Glück remains feedforward+TV-LQR, not energy RL.

#### Force sizing — demote 80–100 N probe

| Source | Actuation | Vs our plant |
|---|---|---|
| Glück Automatica 2013 | \(\|\ddot s\|\le 22\,\mathrm{m/s}^2\) | Needs ~**22 N** on our \(m_c=1\) |
| Baek EAAI 2024 (hardware TIP) | \(F\in[-10,10]\,\mathrm{N}\) | Smaller force on their hardware; still swung up |
| Ours | `forceLimit` **40–50** → peak ~40–50 m/s² | **Already above Glück**; not underpowered |

Keep force probe gated on **rail-slide** or **flat `near_target/align/UUU` past ~u150**. Prefer energy / handoff before more Newtons.

#### Baek / Lim SAC·TQC cookbook (for optional P2 slot)

Same skeleton both Inha-line papers used successfully:

| Knob | Value |
|---|---|
| Algo | SAC (Baek) / TQC (Lim): 3 critics, 25 atoms |
| lr / γ / τ | **3e-4** / **0.99** / **0.005** |
| Buffer / minibatch | **1e6** / **256** |
| Policy net | **400→300** ReLU |
| Update | 1 grad / 1 env step |
| VER | Mirror each transition; Baek doubles mb 256→512 effective |

#### Off-policy reward polarity (arXiv:2410.20096)

Positive dense rewards that spike after first successful swing-up **destabilize Q** (policy degradation). Prefer **cost-style** reward (**0 at optimum**) for any SAC/TQC trial. Our Lim/Baek product is already bounded ∈[0,1]/milder — but if adding sparse UUU bonuses off-policy, frame as punishments not jackpots.

#### Soft-landing / handoff fill (arXiv:2312.11311 SAC→LQR)

Three-stage reward: (1) quadratic swing, (2) line/height toward upright with **−r_vel** if too fast, (3) large bonus inside LQR RoA. Same story as fawraw catch-basin: deliver **slow + near-upright**. Steal **−r_vel near upright** as the soft-landing coef we still lack from fawraw.

#### fawraw status

Unchanged since mid-2026: no soft-landing numbers, no energy Plan-B coeffs, last code **2026-06-25**.

#### Amend recommended redesign (additions only)

31. When coding P2 energy: implement **true \(|E-E_{\mathrm{UUU}}|\)** (formula above) — do **not** treat cranking height-proxy `energy_w` as energy swing-up.
32. **Demote force 80–100** until rail-slide / flat near-align evidence; Glück+Baek say 40–50 is enough on this mass scale.
33. Optional SAC/TQC slot: copy Baek/Lim hypers table; use **cost-style** reward polarity; VER minibatch doubling.
34. Soft-landing starter: **−r_vel** when \(\|\omega\|\) high near upright (2312.11311) before inventing fawraw-unknown coefs.
35. Still: after v5 cook, **split swing vs hold + ~0.1 rad handoff** remains highest-ROI unimplemented architecture change.

**No code this fire** (overnight owns train; wait for green-light / overnight ask). NEED_USER_PING no.

### Research pass (2026-09-19 ~15:42 CT) — progress PBRS + entropy floor + handoff hysteresis

Digged Ng PBRS (progress gaming), AE-PPO adaptive entropy (Symmetry 2026), arXiv:2608.24488 (latent vs executed entropy under tanh), and arXiv:2606.22145 end-to-end (handoff + LPF discrete form + hysteresis). Re-fetched fawraw `sim/handoff.py` + `m4_findings.md` (still mid-2026; no new M4 soft-landing). **Direction unchanged.** P0 progress + P1 flip still live on v5; top *unimplemented* lever remains **two-policy swing↔hold**. No user ping (refines existing P0/P1/P2; no new #1).

Live context (overnight ~15:32): A entropy already **negative (~−0.12 @u100)** under fixed `--ent=0.01` — same collapse class as prior B hot-lr; cooler-lr alone may not be enough next stretch.

#### Progress shaping: raw Δ vs potential-based (PBRS)

Our live term is:

```text
rew += progress_w * (align_now − align_prev)   # γ missing
```

Ng/Harada/Russell **potential-based reward shaping** preserves optimal policies only for \(F=\gamma\Phi(s')-\Phi(s)\). Raw undamped Δ can be farmed by oscillating align (pump progress without net swing-up). With γ=0.99 and Φ = mean cos-align:

| Form | Formula | Status |
|---|---|---|
| Live (fawraw-style Δ) | `progress_w * (Φ_now − Φ_prev)` | Shipping on v5 |
| **PBRS-correct** | `progress_w * (γ·Φ_now − Φ_prev)` | **Not coded** — one-line fix when green-lit |
| Optional Φ | mean cos-align **or** −‖θ−θ*‖ / −\|E−E_UUU\| | Prefer align first (matches current meter) |

Does **not** displace two-policy handoff; it de-risks the progress signal already in A/B/C.

#### Entropy collapse under product+progress (addresses A @u100)

Our actor: unbounded `Normal(mean, std)` on **raw** → `tanh(raw)*forceLimit` executed. Entropy bonus uses **latent** `dist.entropy()` (`train/ppo.py`). Fixed `--ent` default **0.01**.

| Lever | Recipe | Why |
|---|---|---|
| **AE-PPO adaptive β** (Symmetry 2026) | Linear β **0.05→0.005** + raise β when measured H below target, cut when above | Direct anti-collapse; MuJoCo continuous control, not pendulum-specific but matches our symptom |
| **Executed-action entropy H(a)** (arXiv:2608.24488) | Entropy on post-tanh action, not latent u | Latent H has **zero mean gradient** + constant variance push → bound saturation; H(a) Jacobian pulls means inward |
| Fixed floor (cheap A/B) | `--ent` **0.02–0.05** on swing slot A until near_target at_goal moves | Retune-at-u400 candidate if A stays ≤−0.1 |

Prefer adaptive β or H(a) over cranking force. Do **not** mid-run kill; fold into next cool retune / handoff stretch.

#### Two-policy handoff cookbook fills (arXiv:2606.22145 + fawraw)

Already had: catch basin ~0.1 rad / ~0 ω; fawraw latch; LPF τ≈0.3. Newly locked for when we code split nets:

| Knob | Value | Source |
|---|---|---|
| Swing algo (paper) | **TD3** + action LPF in-env | 2606.22145 (we can keep PPO swing) |
| Hold algo (paper) | REINFORCE / any stabilizer | Separate from swing |
| Discrete LPF | `y_{t+1}=y_t+(h/τ)(x_t-y_t)`, **τ=0.3**, h=dt | Prevents bang-bang that broke their hardware |
| Switch hysteresis | **Slower exit than enter**; larger exit bounds (dwell without hard min-time) | Beyond fawraw latch — stops chatter at basin edge |
| Reach / handoff region (single-pole paper) | `|α|<π/12` (~15°), `|α̇|<5π/6`, `|x|<0.4`, `|ẋ|<3` | Scale to triple: use measured basin, not these numbers blindly |
| fawraw API | `HandoffController(swing, stab, target, capture_tol_rad=0.3, capture_vel_rad_s=None, latch=True)` | **Ship tol≤0.1** (measured), not code default 0.3 |
| DR | Sensitivity-guided **small** param set ≫ blanket 10% | Only matters at sim2real; skip for sim-only UUU |

#### Still empty / unchanged

- Serial cart-triple **classical energy coeffs**: still none (Xin=double/parallel; Glück=feedforward). Keep P2 mech \|E−E_UUU\| from 15:14 pass.
- fawraw M4 soft-landing coefs: still unspecified; last code **2026-06-25**.
- Force 80–100: still demoted.

#### Amend recommended redesign (additions only)

36. When touching progress: switch to **PBRS** `γ·Φ_now − Φ_prev` (Φ=mean cos-align); keep `progress_w≈1`.
37. On A entropy ≤−0.1 past ~u150: next stretch try **`--ent` 0.02–0.05** or AE-PPO β schedule; optional entropy on **tanh action**.
38. Handoff impl: copy fawraw latch + **hysteresis exit**; LPF τ=0.3 on swing actions; capture_tol from **measured** basin (≤0.1), never 0.3 default.
39. Rank unchanged: after v5 cook, **split swing vs hold** still highest-ROI unimplemented; PBRS + ent floor are cheap co-travelers.

**No code this fire** (overnight owns train). NEED_USER_PING no.

