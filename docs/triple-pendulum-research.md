# Cart-triple-pendulum research notes

Last updated: 2026-09-18 ~23:55 CT (research only — no triple training yet).

## What "56" means

Not 56 equilibria. A planar cart-**triple**-pendulum has **8** discrete equilibria: each of 3 links is Up or Down → \(2^3 = 8\) (DDD … UUU). Directed transitions between distinct eqs: \(8 \times 7 =\) **56**. (Our double plant is the same idea: \(2^2 = 4\) eqs, \(4 \times 3 = 12\) directed transitions.)

Representation (same UVFA style as our double):
- State: cart \(x,\dot x\) + \((\sin\theta_i,\cos\theta_i,\dot\theta_i)\) for \(i=1,2,3\)
- Goal: one-hot(8) + target sin/cos for three angles (or 3-bit EP index)

## Classical control vs RL

| Approach | What it does | 56 transitions? |
|---|---|---|
| Glück et al. Automatica 2013 | Nonlinear feedforward (BVP) + time-varying Riccati feedback; experimental swing-up DDD→UUU | Designed for one (or a few) transitions; other setpoints in principle, not all 56 demonstrated |
| Graichen et al. | Precomputed trajectories + feedback for many transitions | Classical; many trajectories hand-designed |
| Lim / Ju / Lee KIEE 2025 | Sim-to-real RL; claims **all 56** on hardware | Yes (reported) |
| fawraw/triple-pendulum-sim2real | MuJoCo + TQC; goal = all 56 without feedforward | In progress; 8-EP stabilize ~70%+; 56 transitions staged |

**Bottom line:** Explicit control can swing-up / stabilize with models + trajectories. Getting *all* 56 is exactly the modern RL/sim2real benchmark. PPO/TQC/SAC are viable; our transition-only PPO recipe is the right *shape* of objective.

## Can we "append" the double policy?

Action stays 1D (cart force) — good. Observation and goal encodings **must grow**:
- Double obs_dim 16 → triple roughly +3 (sin/cos/ω for link 3) + expand one-hot 4→8 and goal sincos +2 → new obs_dim ~22–24.

Naïve "pad and continue" does **not** just work: weights for θ3 / new goal bits are undefined; dynamics change (coupling). Options that *are* researched:

1. **Input/output adapters** — freeze double hidden layers; new linear map; expand goal head; fine-tune with PPO (PPOPT-style).
2. **Progressive nets** — freeze double column; new column for triple with lateral adapters (Rusu et al. 2016). Keeps double policy intact.
3. **Fresh policy + curriculum** — train triple from scratch; use double only as teacher / reward shaping for the first two links (weaker transfer).

Honest expectation: (1) or (2) can reuse swing/centering instincts for the base links; the third link and the 56-transition graph still need substantial new learning. Not a free append.

## Implication for our stack

- Keep double **xonly** grinding 12 directed transitions.
- Triple is a **new plant** (physics + 8 goals + transition-only over 56 pairs), not a drop-in.
- When we *do* build it: copy double train loop → 8 goals → `--transition-only` over A→B≠A; evaluate all 56 pairs; consider adapter/progressive init from double ckpt as an experiment, not a requirement.

## Sources

- Glück, Eder, Kugi — Swing-up of a triple pendulum on a cart (Automatica 2013)
- Botha & Qi — Analysis of the triple pendulum (8 equilibria)
- Lim, Ju, Lee — 56 transition control via sim-to-real RL (KIEE 2025)
- https://github.com/fawraw/triple-pendulum-sim2real
- Progressive Neural Networks (Rusu et al. 2016); PPOPT pretrain sandwich (2025)
