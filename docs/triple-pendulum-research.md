# Cart-triple-pendulum research notes

Last updated: 2026-09-19 ~00:00 CT (research only — no triple training yet).

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

- Keep double **xonly** grinding 12 directed transitions. Do **not** start triple training or spin extra GPUs.
- Triple is a **new plant** (physics + 8 goals), not a drop-in.
- When we *do* build it, two credible recipes:
  1. **Copy our loop:** 8-way UVFA + `--transition-only` over A→B≠A; evaluate all 56 pairs (closer to our double).
  2. **Copy Lim:** TQC (or SAC) × 8 EP policies; product reward with cumulative angles; wide IC randomization; no explicit 56-pair sampler.
- Optional: adapter/progressive init from double ckpt as a side experiment, not a requirement.

## Sources

- Glück, Eder, Kugi — Swing-up of a triple pendulum on a cart (Automatica 2013) — DDD→UUU
- Graichen, Treuer, Zeitz — Fast side-stepping of the triple inverted pendulum (CDC 2005)
- Lim, Ju, Lee — 56 transition control via sim-to-real RL / TQC (KIEE 2025); PDF + YouTube above
- MDPI Machines 13(3):186 (2025) — double inverted pendulum Sim2Real, 4 EPs / 12 transitions
- https://github.com/fawraw/triple-pendulum-sim2real (open TQC attempt; M4 not finished as of mid-2026)
- Progressive Neural Networks (Rusu et al. 2016); PPOPT-style adapters
