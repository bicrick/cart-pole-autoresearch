# Triple pendulum — training wheels plan

Last updated: 2026-09-21 (DDD 150k died before zip; vendor M2 ep7=0.50). This is the high-level project plan. Findings: `docs/triple-uuu-hold-findings.md`. Operational fire logs stay in `docs/triple-macro-loop.md`. Method notes stay in `docs/working-impls-reverse-eng.md`.

## Destination

An interactive cart-triple demo that can:

1. Hold **UUU** (all three links upright)
2. Recover after a shove or a drag
3. Hold all **8** equilibria (DDD … UUU)
4. Switch among them — eventually all **56** directed transitions

Ship plant is **no walls** (void / respawn). Training starts with **walls on**. Walls come off only after hold already works.

## MPPI teacher (supersedes the stack below)

As of 2026-09-24 the working controller is a sampling MPC teacher, not RL. Code: `train/mppi/`.

- **Teacher.** MPPI on the exact batched plant (`train/mppi/dynamics_fast.py`, parity with `physics_triple` to 1e-15). Each sample is `clip(gate(s) * LQR_target(s) + residual)`. The target's LQR is faded in near the goal and out far from it, so swing-up plans are scored on whether LQR can catch the arrival. Pure open-loop MPPI cannot work here (upright modes grow at ~11/s), and multi-start iLQR could not beat LQR near upright.
- **Config.** `train/mppi/configs/uuu.json` (4096 samples, 45 knots x 4 steps = 1.5 s). The same config works for every goal via `--goal`.
- **Gates** (`scripts/mppi-teacher-gates.sh`). G1 compares against the target's LQR on the same starts: independent +-0.10 rad per-link starts are mostly unrecoverable on this plant, so a fixed 0.95 bar there is out of reach. UUU at N=50: quiet 0.98 (LQR 0.86), 0.05 rad 0.84 (0.72), 0.10 rad 0.40 (0.24), shoves equal to LQR. G2 swing from a hang, UUU N=50: 49/50, median entry 5.5 s. The other 7 goals at N=10: 10/10 each (DDD starts near UUU). Results: `policies/mppi/`.
- **Web demo.** The teacher runs in the browser (`web/src/mppi/`). A JS port of the numba rollout, parity ~1e-16, is spread across Web Workers. The 4096-sample teacher replans in 16–24 ms on an M2 Pro, so it runs in real time with no server. The page holds physics until each plan returns, so it behaves exactly like the Python loop. `train/mppi/test_web_mppi.py` checks it: 8/8 UUU swing-ups from a hang in node. The Python-backed sim (`scripts/mppi-server.sh`, `/?sim#triple`) is kept for comparison.
- **Speed cuts (UUU G2, N=20, `mppi/speed_sweep.py`).** Kept: hold mode (128 samples while quiet at the target) and early exit for samples past `x_dead`, both free. Rejected: 60 Hz rollouts (`rollout_sub=2`, 0-5% swing-up; semi-implicit Euler drifts off the 120 Hz plant). Also rejected: a 0.5 s horizon, which gave 20% swing-up with the analytic terminal and 0% with a learned terminal value (`mppi/value.py`, fit to teacher cost-to-go, ~1.8x log-space error).
- **Distillation (negative).** A state-only MLP student reached 0% swing-ups. The teacher's force depends on its current 45-knot plan: the same state gives labels ~15 N apart under different plans, against ~2 N seed noise from a fixed plan. A plan-conditioned net (`mppi/plannet.py`, `mppi/dagger_plan.py`) fits teacher-driven data to noise level. Once the net drives, one MPPI pass from its plans gives ~22 N seed-to-seed labels. Labeling from the teacher's own shadow plan makes the label depend on a plan the net never sees (fit error rose to ~27 N RMS). 0% swing-ups after 3 DAgger iterations.
- **Distillation** (`scripts/mppi-dagger.sh`, anchored student). Parked: the student holds like LQR but has 0 swing-ups. Larger nets fit the labels better without swinging, which points to noisy MPPI labels.

## Quad pendulum (n-link MPPI, 2026-09-25)

The same recipe on a general n-link plant (`train/mppi/nlink/`), swept on a GCP L4 with a numba.cuda f32 kernel (about 1.7 billion plant steps/s at 131k samples). Results: `policies/mppi/quad/`.

- **All 16 goals** (32k samples, 1.5 s horizon, 40 N, 10 episodes each, from a hang): 148/160. Twelve goals at 10/10. UUUD 9/10, DUUU and UDUU 8/10, UUUU 3/10. No episode that reached its goal fell out of it: every failure is a timeout. How hard a goal is tracks its fastest open-loop divergence rate (UUUU and UUUD at about 13.5/s).
- **UUUU** (20 episodes): a 1.25 s horizon beats 1.0 s and 1.5 s (40% vs 20% and 25% at 32k). Two MPPI passes per replan barely help (45%), and neither does more force (60 N 15%, 80 N 30%). More samples do: 65% at 131k with a 40 s window, median 12 s.
- **Fewer samples** (1.25 s horizon): 9 of 16 goals at 10/10 at 4k, 10 at 8k, 11 at 16k. The goals with three or four links up are the ones that need 32k.
- **Browser.** `web/src/mppi/rollout-nlink.js` and the generated WGSL (`gpu-kernel-nlink.js`) run it with no server. On an M2 Pro, WebGPU replans 32k samples in about 13 ms, inside the 33 ms knot budget. Live on the page, UUUU swung up from a hang in 42 s and held, at full sample count and 120 fps. The single pendulum is the same code with n = 1.

## Locked method (superseded, kept for history)

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

Current wheel: **swings into the other seven equilibria.** Void quiet-holds, noise 0.03, N=50, walls off: DDD, DDU, DUD, DUU, UDD, UDU, and UUD all pass at survival 1.000 and at_goal 1.000 (align 0.994–1.000). UUU hang-to-upright is gated (W3 enter 1.0, W4 hold 1.0, W5 void quiet hold 0.90). Quiet holds for all 8 are gated at noise 0.03 with the walls on, and the same seven stay up with the walls off. Demo loads all eight. Product-reward swing never entered (closest 0.111 rad at 25k, then 1.87). M4 quadratic cost reaches the neighborhood and then slips: corrected 25k median closest 0.121 rad at 17.9 rad/s (10/10 inside 0.3 rad), 50k slipped to 0.334 rad at 18.3 rad/s (4/10 inside 0.3). Enter-rate still 0.000. The earlier 0.220 and 0.620 figures were distance to the goal index. That slip is the fast pump being charged the whole way up. One 25k rollout does pass through 0.089 rad at step 1112, spinning at 17.7 rad/s, then falls back to a hang. The UUU hold catches about 0.1 rad/s at 0.03 rad and falls by 0.5 rad/s, so that delivery cannot be handed off. Discounting spin everywhere made the whip about 15k return better and only moved the near-top cost by about 300, so that run was stopped. The soft-landing run that zeroed spin cost far from the target was stopped at 25k (exit 143). Bottom-spawn enter-rate 0. Closest point: 10k median 0.862 rad at 4.8 rad/s (10/10 inside 1 rad); 25k median 1.167 rad at 2.1 rad/s (0/10 inside 1 rad). Started at 0.05 rad with 5 rad/s, it came back no closer than 0.45 rad still near 5 rad/s. That spin drop is a weaker pump: the hold only catches about 0.1 rad/s, so a 5 rad/s arrival is outside any savable set, and discounting the pump removed the cost that had pushed the quadratic run to 0.22 rad. The replacement keeps the published vel cost everywhere and spends half its episodes near upright with a shared rate only up to 0.3 rad/s. LQR from θ=0.05 catches 0.10, 0.20, and 0.30 rad/s (final ω about 0.02) and falls at 0.50, so 0.5 was outside the controllable set. The hold itself only catches 0.10. At 25k this run, from a hang, median closest is 0.177 rad at 27.1 rad/s (7/10 inside 0.3 rad, enter-rate 0). Started at 0.05 rad with 0.3 rad/s, the closest later point is 0.127 rad at 39 rad/s, so the near-upright starts were pumped, not bled. That job was stopped. Under the same reward, LQR from that start returns about +37 and pays the arrival bonus; the swing returns about -209k. The replacement keeps the published vel cost and the 0.3 rad/s arrivals, and adds an LQR anchor on the actor only while `|θ|<0.35` and `|ω|<0.40`. That anchor at 25k still did not brake: hang closest 0.512 rad at 17 rad/s, and a 0.3 rad/s start came back at 16 rad/s. The actor matched LQR only on the first full-force shove, because the rest of the brake was not in the replay. The live run writes 32 of those LQR arrivals into the buffer first (mean return about +162). At 25k that still did not enter: from a hang the closest point was 2.45 rad at 1.0 rad/s, and a 0.3 rad/s start came back at 0.11 rad spinning at 25 rad/s, so the job was stopped. The replacement is the published one-link probe, DDD to UDD, with the same M4 cost and no arrival mix and no LQR anchor. The enter-rate script was subtracting the goal index from the angles; UUU hid that because its index wrapped near upright. Corrected: UDD 25k median closest is 1.32 rad at 18 rad/s (2/10 inside 1 rad), and 50k is 2.37 rad at 1.2 rad/s, so that job was stopped. The same fix on the three-link quadratic 25k says it really does reach 0.12 rad at 18 rad/s, all 10 episodes inside 0.3 rad, enter-rate still 0. The live swing is that same cost on a 1.10 m rail. At 25k, from a hang on that rail, median closest is 0.051 rad at 42.0 rad/s (10/10 inside 0.3 rad, enter-rate 0). The shorter rail got the angle closer and drove the links into the 50 rad/s speed limit, with the cart on the bumper for hundreds of steps, so that job was stopped. On the 2.4 m flyby the spin cost inside 0.5 rad is only about 450, and a 13× near-top multiplier adds about 3000 to a −120k return. Gain 700 at a 0.3 rad scale added about 166k on that flyby, including 24k while still more than 1 rad from upright. At 25k the pump was gone: median closest 2.80 rad at 0.5 rad/s, 0/10 inside 1 rad, so that job was stopped. The replacement keeps gain 700 and shortens the scale to 0.10 rad. On the same flyby that adds about 16k, and almost nothing past 1 rad. That job ran to 225k and was then stopped. From a hang: 25k median closest 0.096 rad at 8.2 rad/s (10/10 inside 0.3 rad), 225k slipped to 0.545 rad at 8.8 rad/s (0/10 inside 0.3, 9/10 inside 1). Enter-rate still 0. The 25k zip is the best delivery on disk. On that 25k flyby, raising the peak from 700 to 2000 would add about 75k. That job was stopped before 50k. A classical energy pump (force aligned with cart velocity, saturated at 40 N, reversed at the bumper) was measured from a hang. Counting cart kinetic energy, median closest was 1.46 rad at 2.7 rad/s. Leaving cart kinetic energy out of the target, median closest is 1.02 rad at 6.5 rad/s. Both have lqr-region rate 0 and quiet enter-rate 0, so it was not latched to LQR or the hold. Handing the 25k swing to that pump once `|θ|≤0.5` reaches 0.022 rad but at 9.6 rad/s. Handing it to LQR at the same point stays at 0.10 rad and 11 rad/s. Neither enters the LQR region. From the first step already inside 0.3 rad, the better saturated force (40 N) cuts the spin while still inside that box from about 8 rad/s to 2.8 rad/s, and the poles leave at about 3 rad/s. A short search over saturated forces from 0.5 rad also misses the box (closest about 0.12 rad at 5.6 rad/s). Raising the force on that same pass barely moves the floor: 40 N reaches 2.8 rad/s inside 0.3 rad, 80 N reaches 2.6, 120 N reaches 2.3, and 200 N reaches 2.1. None enter `|θ|≤0.05` and `|ω|≤0.30`. An 8-segment bang-bang search from the hang (256 sequences at 40 N) also misses that box: the closest angle is 0.028 rad, at 31 rad/s. Reshaping the last 0.5 s of the 25k swing that already reaches 0.084 rad only moves the arrival cost from 9.5 to 9.1, still a fast pass. On that episode the spin is already 15 rad/s when the poles are still 2 rad from upright, 5 rad/s at 1 rad, and 9.6 rad/s at the closest point. Editing the 2 s before the top only got the arrival cost from 8.3 to 5.5 before the search stopped. A 16-segment search of the whole rise (448 evaluations) comes back to the same fast pass: 0.12 rad at 8.6 rad/s. Enter-gate stays `|θ|≤0.03`, `|ω|≤0.01`. Findings: `docs/triple-uuu-hold-findings.md`. Warm-starting that 25k swing with a 1.5 charge on pole energy above the upright energy made the pass worse: at 25k, median closest 0.199 rad at 21.9 rad/s, 9/10 inside 0.3 rad, never inside 0.05 rad, lqr_region_rate 0, enter-rate 0. The run was stopped there. The 25k near-top swing at 0.096 rad and 8.2 rad/s remains the best delivery. A 4-second program ended at 0.88 rad and 12.9 rad/s. An 8-second search with the rest condition weighted (180 tries, 12 segments) ended at 0.91 rad and 3.9 rad/s with the cart held on the 2.4 m bumper. The same search on a 6 m rail ended at 1.12 rad and 4.3 rad/s with the cart at 2.86 m, not on the bumper. A deeper 2.4 m search (528 tries, 16 segments) ended at 0.93 rad and 7.4 rad/s, again on the bumper. Still outside the LQR box. A constant 40 N shove from the straight hang separates the links within 0.7 s (inner link at 2 rad/s, outer link nearly stopped). Scaling the best swing's force by 0.5 still reaches 0.06 rad, at 30 rad/s. Pumping only the outer link's energy, the one a constant shove leaves behind, gets closer: median closest 0.27 rad at 9.3 rad/s, and 2/10 episodes pass 0.05 rad at about 10.8 rad/s. The first time |θ| drops through 0.5 rad the spin is already 3.7–18 rad/s, so there is no slow state to hand to LQR. Gradients of an 8-second force program explode (norm about 1e9); a step along that gradient hits the 50 rad/s speed clip. At full gain the outer pump either matches upright energy with the tip still at 3.6 rad/s (inner rates already inside 0.3) or overshoots to about 16 J and crosses the top at 7–12 rad/s with the brake saturated. Quarter gain never builds that energy and stays in the lower half. Multiple shooting on 0.25 s pieces starts with the pieces joined and, after 80 steps with the clip loosened, still has a knot jump of 0.50. The open-loop replay ends at 2.77 rad and 12.8 rad/s. Pumping only the cart-most link (k=40.8, sized to saturate the one-link gap) does not leave the other two hanging: median closest to UDD is 0.65 rad at 10.5 rad/s, best 0.24 rad at 9.2 rad/s. The nearrad 25k policy passes 1 rad at about 5 rad/s. Handing that state to LQR whips it to 15–30 rad/s. Handing it to the pole pump either does the same or stalls near 0.7 rad still at about 4 rad/s. Staying with the policy reaches about 0.1 rad at 8–9 rad/s. At that 1 rad moment the cart is already on the 2.4 m bumper. Replaying the same policy on a 6 m rail still rides the bumper and arrives worse, median 0.14 rad at 10.1 rad/s. A 4-second collocation from hang to rest does converge, with peak force 14 N and the cart inside 2.2 m, but the path spins at 17.5 rad/s. Played through the demo's Euler step it ends at 0.22 rad and 18 rad/s; one step drifts from that curve by 0.20. Newton on that same stepper joins 0.25 s pieces to about 1e-4, last piece included, at about 14 N. One-shot playback from the hang leaves the curve about 2 s in and ends near 2.7 rad and 6–8 rad/s. Time-varying Riccati along that same curve, using the upright LQR weights, ends at 3e-5 rad and 8e-5 rad/s from the exact hang, and on 5 noisy hangs (angles ±0.05, rates ±0.01) the worst arrival is 0.0003 rad and 0.003 rad/s. Quiet-box rate 1.0 on that sample. Force never saturated. Latching the UUU hold zip for 1000 steps from those arrivals, including the exact hang, gives survival 1.0, at_goal 1.0, align 1.0 on that sample of 6. With the demo bumpers on, that same tracker still finishes in the quiet box (max |x| 2.07 m, 3e-5 rad, 8e-5 rad/s). The triple page plays that file from a hang, then the UUU hold. On 50 bottom starts (noise 0.05, walls on) enter-rate is 1.000, worst arrival 0.0005 rad and 0.005 rad/s, cart inside 2.13 m, force never saturated, and the UUU hold then stays at survival 1.000, at_goal 1.000, align 1.000. The same 50 with the walls off match that arrival and the hold still stays at 1.000. The quiet-basin UUU hold on the void plant, noise 0.03, N=50, is survival 0.900, at_goal 0.900, align 0.993. Demo bumpers stay on. Switching the hold that owns the destination, walls off, quiet start at the source, N=5: the 8 same-goal pairs stay (at_goal 1.00). All 48 other pairs die on the first step (at_goal 0.00). The catcher does not own a foreign equilibrium, so those pairs need their own delivery. DDD to UUU is the one delivery that already enters. The same 4-second collocation aimed at DDU does not converge: the mesh hits its node cap with the tip's rate still 0.17 rad/s, the path spins at 160 rad/s, and the peak force is 81 N. An 8-second try makes the Jacobian singular. Aiming the same collocation at UDD is singular too. Discrete Newton from a straight angle line, with the step allowed to be eight times longer, brings the shooting residual down to 0.20. Open-loop playback still ends 3.04 rad and 11.8 rad/s away, and time-varying Riccati along that curve saturates on 60% of steps, runs the cart to 14 m, and finishes 2.78 rad and 11.7 rad/s from UDD. Not a delivery. A +40 N shove added the way the demo's a/d key is added, walls off, N=5: 0.05 s is survived by every hold except UUU (UUU survival 0.20). At 0.10 s, DDD, DDU, DUD, and UDD still survive and DUU, UUD, and UUU fall. At 0.25 s every hold loses survival.

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
