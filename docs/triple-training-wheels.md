# Triple pendulum — training wheels plan

Last updated: 2026-09-21 (DDD 150k died before zip; vendor M2 ep7=0.50). This is the high-level project plan. Findings: `docs/triple-uuu-hold-findings.md`. Operational fire logs stay in `docs/triple-macro-loop.md`. Method notes stay in `docs/working-impls-reverse-eng.md`.

## Destination

An interactive cart-triple demo that can:

1. Hold **UUU** (all three links upright)
2. Recover after a shove or a drag
3. Hold all **8** equilibria (DDD … UUU)
4. Switch among them — eventually all **56** directed transitions

Ship plant is **no walls** (void / respawn). Training starts with **walls on**. Walls come off only after hold already works.

## Locked method

Research (Glück 2013, Baek 2024, Lim 2025, fawraw) picked the stack. We copy it. We do not invent another PPO zoo.

| Decision | Locked choice | Why |
|---|---|---|
| Algorithm | **TQC** (SAC only as a later cousin) | Hardware winners. PPO/ATRPO/HER did not. |
| Action | **Continuous** cart force / accel | Discrete / bang-bang shakes the chain and falls. |
| Policies | **One specialist per equilibrium** (8 nets) | Lim: 8 specialists, not 56 policies, not one UVFA. |
| Reward | **Product** of `[0, 1]` terms | Additive `x²` / spin / align fights energy and farms DDD. |
| Hold vs swing | **Separate roles, then hand off** | Glück 2-DOF. fawraw M2 → M3 → M4. |
| Transitions | **Switch specialists** after each EP can be reached from wide ICs | Lim: 56 “for free.” Do not train 56 nets. |
| Local capture oracle | **LQR** near UUU | Already PASS. Proves the plant is holdable. |

**Do not revive:** PPO / ATRPO / AVC / gSDE / ENT ladders, one-net eight-goal UVFA, HER on discrete equilibria, bang-bang actions, additive unbounded rewards, 56-transition training before UUU hold exists.

## Rules of the road

1. **One wheel at a time.** Clear the gate before starting the next wheel.
2. **Training wheels first.** Walls on, quiet basin, one goal, tiny local run — then widen.
3. **Prove stay, not visit.** Reward up + hold ≈ 0 is a fail, not progress.
4. **Local before cloud.** Mac Silicon smoke must run before any GCP stretch.
5. **No mid-run redesign.** If a wheel fails its gate, stop, quarantine, drop to the simpler wheel. Do not stack paper knobs.
6. **One change per retry** when a wheel fails.
7. **Cite a file or paper.** Every train job names a source (`m2_upright_tqc.yaml`, Lim 8 specialists, Glück 2-DOF, fawraw `m4_findings.md`). If the knob is not in that source, we do not add it. Do not port fawraw M3 one-net UVFA. Do not reopen W2 0.04 grinding.

## Where we are now

| Asset | Status |
|---|---|
| Quiet-basin env contract (`train/envs/triple_gym.py`) | Done |
| LQR UUU oracle (`scripts/lqr-oracle-uuu.sh`) | **PASS** (survival 1.0 @ noise 0.01 / 0.02) |
| BC from LQR demos (`scripts/bc-lqr-uuu.sh`) | **PASS** (survival 1.0 / at_goal 1.0 / align 1.0 @ 0.02) |
| Browser demo | Triple defaults to UUU; loads BC hold (`train/export_hold.py` → `web/public/policy-triple.json`) |
| TQC frozen-actor smoke (`policies/tqc-m2-uuu-hold-mac-smoke.zip`) | **W1 GATE PASS** (N=50) — actor never unfroze |
| TQC freeze-then-unfreeze (no BC-reg) | Fail: survival 1.0 → 0 within ~4k steps after unfreeze |
| TQC + BC-anchor (`policies/tqc-m2-uuu-hold-mac.zip`) | **W1 GATE PASS** (N=50, noise 0.02, after actor unfreeze). Also **W2@0.03 PASS** (0.80 / 0.80 / 0.98) |
| W2 mix ICs `[0.02,0.04]` (`*-n004-mix*.zip`) | Fail: mix1 0.72@0.04, mix2 0.64@0.04. Do not repeat. |
| W2 0.03-LQR-BC stay (`policies/tqc-m2-uuu-hold-w2-n004-lqrbc.zip`) | **W2@0.03 PASS** 0.90 / 0.90 / 0.99. W1 quiet 1.0. **0.04 FAIL** 0.72 / 0.72 / 0.98 |
| W2 stay + VER (`*-n004-ver.zip`) | Same 0.72@0.04 plateau. 0.03 0.88. Do not repeat. |
| W2 hard-IC oversample (`*-n004-hard.zip`) | Fail: 0.68@0.04, **0.03 slipped 0.90→0.74**. Do not repeat. |
| W2 looser BC-reg 1.0 (`*-n004-loose.zip`) | Same **0.72@0.04** LQR ceiling. 0.03 0.80 (slipped from 0.90). Quiet 1.0. |
| W2 0.035 LQR dump (`policies/lqr-uuu-demos-n035.npz`) | 138/160 survivors (oracle 0.80, bar 0.95). Job stopped; no more 0.035 TQC. |
| W2 0.035 LQR-BC (`policies/bc-lqr-uuu-n035.pt`) | 0.78@0.035 (shy of 0.80). 0.84@0.03. 1.0@0.02. |
| PPO / ATRPO / HER zoo | Retired. Do not continue those checkpoints. |

Current wheel: **W3 UUU swing from the bottom.** Quiet holds for all 8 are gated at noise 0.03. Demo loads all eight. The product-reward swing (progress on cos-align, barrier 50) stayed alive and never entered the catch box: closest angle 0.111 rad at 25k, then 1.87 rad at 75k, enter-rate 0.000. That job is stopped. The replacement uses the published M4 swing cost (weighted error² + vel 0.02 + cart 0.2 + barrier 50 + one-shot bonus 200 after 100 steps inside 0.3 rad). Enter-gate stays `|θ|≤0.03`, `|ω|≤0.01`. Findings: `docs/triple-uuu-hold-findings.md`.

---

## The wheels

Each wheel is a smaller bike. We only take the next set of wheels off when the gate is green.

```text
W0 smoke  →  W1 quiet hold  →  W2 widen hold  →  W3 swing
    →  W4 handoff  →  W5 void  →  W6 eight EPs  →  W7 switch (56)
```

### W0 — Local smoke (Mac Silicon)

**Goal.** Prove the new stack *runs* on this machine. Not a good policy. A green process.

**Plant.** Walls on. UUU only. Quiet basin.

**Recipe.**
- Device: Apple Silicon (`mps` if TQC/sb3 supports it cleanly, else `cpu`)
- Tiny steps (smoke flag), 1 env, product reward, `progress_w=0`
- Existing scripts: `SMOKE=1 bash scripts/next-train-triple-m2-hold.sh`

**Gate to leave.**
- Import + env reset + a few TQC updates without crash
- Checkpoint writes
- Eval loop prints `survival_success` / `at_goal` (values may be ~0)

**Fail.** Install / MPS / gym contract bugs. Fix those. Do not “just train longer.”

---

### W1 — Quiet-basin UUU hold (training wheels)

**Goal.** A specialist that *stays* upright if it starts almost upright.

**Plant.** Walls on. Near-target ICs only. Hang = 0. Wide = 0.

**Recipe (fawraw M2 / Lim product).**
- TQC, lr `3e-4`, net `[128, 128]`, 3 critics
- Quiet rates **fixed ±0.01** (do not scale rates by `init_noise`)
- Fall-kill `|θ_err| > 0.6` on UP links
- `progress_w=0`
- Optional warm start from `policies/bc-lqr-uuu.pt`
- Primary meter: **survival** (`ep_len ≥ 0.8 * max_steps`), plus `at_goal`

**Local first.** Short CPU/MPS run to see survival leave 0. Then, only if that trend is real, a longer stretch (cloud L4 is allowed *after* W0 is green and this local run is healthy).

**Gate to leave.**
- Survival ≳ **0.80**
- `near_target/at_goal/UUU` ≳ **0.80**
- Align ≳ **0.90**
- Entropy / action not bang-banging the chain

**Fail.** Reward up, hold flat → stop. Next simpler retry is env contract (rates, fall-kill, walls), not a new algorithm.

---

### W2 — Widen the hold basin

**Goal.** Same UUU specialist, larger catch region. Still hold-only.

**Plant.** Walls on. Still no hang.

**Recipe.** After W1 gate: measure the hold basin. On this plant the lock is **0.03** (n004-lqrbc 0.90). fawraw’s YAML `init_noise: 0.05` is their near-upright spawn on **their** MuJoCo plant — not a number we must grind to here. Do not add mix / VER / hard-IC / BC-reg ladders that are not in their M2 file.

**Gate to leave.** Quiet hold (W1) plus a measured catch box the specialist actually owns. We have that at 0.03. Stop widening.

**Fail.** Do not add swing ICs to “help” a dead hold. Do not invent knobs to beat LQR at 0.04.

---

### W3 — Swing specialist (separate net)

**Goal.** Get *near* UUU from hanging / bottom. Not required to hold forever.

**Plant.** Walls on. Hang-heavy ICs. Product + progress term allowed here only.

**Recipe.** Copy a published swing, do not design one. Glück energy/BVP feedforward, or fawraw M4 swing (`init_mode: bottom`, their progress + cart barrier). Delivers into the measured W2 catch box. Judge on enter-gate rate, not `at_goal` hold.

**Gate to leave.** Hang rollouts reach the handoff box often enough to be useful (start looking around enter-rate ≳ 0.3, then chase higher).

**Fail.** A swing net that parks at DDD. Do not fold swing into the hold net to “save capacity.”

---

### W4 — Handoff (glue)

**Goal.** Swing delivers, hold latches, stay upright.

**Plant.** Walls on.

**Recipe.** Copy Glück / fawraw M4 two-DOF, not a homemade third net: swing policy → enter the catch box this hold net **owns** → latch. On this plant that box is `|θ| ≲ 0.03` and `|ω| ≲ 0.01` (quiet-rate contract). fawraw measured ~0.1 rad / near-zero rate on **their** plant; do not paste 0.1 onto ours. Dwell. Exit if we leave the box. LQR remains the backup catcher. Glue already stubbed in `train/handoff.py`.

**Gate to leave.** Hang → hold success rate useful (≳ 0.3, then chase 0.7+). Hold meters from W1 must not collapse.

**Fail.** If handoff misses because swing never arrives, stay on the published swing recipe. If it misses because delivery is outside the measured box, teach swing to hit that box (fawraw M4 “soft delivery”). Do not reopen W2 grinding. Do not train a third “do everything” net.

---

### W5 — Take the walls off

**Goal.** Same UUU path on the **void** ship plant.

**Plant.** Walls off. Hard OOB terminal + large oob penalty (after reward clip).

**Recipe.** Fine-tune W1 hold first, then W3+W4. Never void-FT a policy that cannot hold on walls.

**Gate to leave.** W1 / W4 meters on the void plant. No mid-track flop-to-avoid-void farming.

---

### W6 — Eight equilibrium specialists

**Goal.** Lim’s 8 policies. One target EP each. Clone the W1 (then W2) recipe.

**Plant.** Walls on until each EP holds, then void FT per specialist.

**Recipe.**
- Cite: fawraw `m2_upright_tqc.yaml` + Lim 8 specialists
- Same TQC + product reward + quiet basin. Change **only** the target angles for that EP
- First after UUU: **DDD** (easiest hang EP in fawraw’s M3 table)
- Train until return plateaus in the Lim band (~700–800 / 1000) *or* our survival / `at_goal` gates
- Do **not** port fawraw M3 one-net / one-hot / `target_mode: random`
- Do **not** reopen W2 0.04 grinding on a new EP

**Gate to leave.** Each of the 8 EPs clears a hold gate (start at `at_goal` ≳ 0.55, align ≳ 0.75; UUU stays at the stricter W1 bar). DDD first uses the W1 bar (survival ≥ 0.80, at_goal ≥ 0.80, align ≥ 0.90) at noise 0.03.

**Fail.** One shared UVFA for all eight. That is the retired approach.

---

### W7 — Switch specialists (56)

**Goal.** Interactive transitions. Not 56 trained trajectories.

**Plant.** Void (ship).

**Recipe.** At runtime, pick the specialist for the *requested* goal. Lim’s claim: if each specialist can reach its EP from wide ICs, the 56 directed pairs come from switching, not from a 56-way curriculum. If a pair is still weak (the fawraw M4 lesson: deliver into a basin the catcher actually owns), add a **soft delivery** for that pair only.

**Gate to leave.** Directed A→B eval over 56 pairs good enough for the demo. Drag / shove recovery still works.

**Fail.** Do not stand up a 56-policy farm or a HER multi-goal PPO to “finish faster.”

---

## Local Mac Silicon pass (do this before any new cloud job)

This is W0, spelled out.

1. Confirm `scripts/lqr-oracle-uuu.sh` still PASSes on this machine (CPU).
2. Confirm `scripts/bc-lqr-uuu.sh` eval still PASSes.
3. `SMOKE=1` the TQC M2 trainer with MPS-or-CPU.
4. If smoke is green, a short local hold run (minutes, not overnight) with the W1 contract.
5. Only then: longer W1 on the L4, from a clean TQC/BC init — not from retired PPO checkpoints.

Optimize later (vectorized envs, MPS, thread counts). First we need a trainer that takes a step.

## Success meters (read these, not mean reward)

| Wheel | Primary meter | Ignore as “winning” |
|---|---|---|
| W0 | Process lives; logs exist | Reward number |
| W1–W2 | Survival + `at_goal/UUU` + align | Reward if hold is flat |
| W3 | Hang enter-rate / hang align | `at_goal` hold |
| W4 | Hang → hold success | Swing reward alone |
| W5 | Same as W1/W4 on void | Center-only / not-dying |
| W6 | Per-EP `at_goal` (min over 8) | Mean over easy EPs |
| W7 | 56-pair success + recovery | A pretty UUU clip |

## Pointers

| Thing | Where |
|---|---|
| UUU hold findings (paused) | `docs/triple-uuu-hold-findings.md` |
| Research brief | Canvas `triple-pendulum-methods.canvas.tsx` |
| What already worked elsewhere | `docs/working-impls-reverse-eng.md` |
| LQR oracle | `scripts/lqr-oracle-uuu.sh` |
| BC | `scripts/bc-lqr-uuu.sh` |
| TQC M2 hold | `scripts/next-train-triple-m2-hold.sh` |
| Gym contract | `train/envs/triple_gym.py` |
| Overnight ops (not this plan) | `docs/triple-macro-loop.md` |
