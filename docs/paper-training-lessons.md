# Paper training lessons → next cart-double-pendulum runs

Audit of `papers/*.pdf` against `train/goals.py` + `train/train.py` (goal-conditioned UU/UD/DU/DD, θ=0 upright).

**Status 2026-09-18 ~00:30 CT:** Collapsed HER@0.8 run finished at update 400 with `eval/reward≈−8916`, `align≈0` (discard). Fresh paper-recipe run `20260918-052042` is live on `cartpole-train` (commit `20a3ae4` knobs): at **update ~210/400**, `eval/reward≈+253`, `eval/align≈0.21` (rising; no −10k collapse). TB: http://34.148.138.48:6006/. **Do not kill / redesign mid-run** while reward and align keep climbing.

---

## (a) What each paper did

### Xin / IFAC 2008 — *Energy-based swing-up for cart double pendulum*
- Plant matches ours: four equilibria **UU / UD / DU / DD**; θ=0 upright.
- Control is **hybrid**: energy pump toward `E → E_uu`, then **LQR capture** near upright.
- Lyapunov: `V = (E − E_uu)² + k_D ẋ² + k_P x²`. Cart displacement is controlled; energy converges to UU for almost all ICs; other equilibria are unstable attractors of the energy controller if `E* ≠ E_uu`.
- **Lesson:** swing-up ≠ balance. Need energy shaping (or a curriculum that separates them). Soft `x` regulation matters, but unbounded `x` is fatal.

### Spong 1995 — *Acrobot swing-up*
- Two algorithms (partial feedback linearization + **energy pumping**), then switch to **LQR balance**.
- Energy is *increased* during swing-up (high |ω| is expected). Capture region of the local controller is small → must arrive near the upright manifold with the right energy.
- **Lesson:** heavy spin penalties fight the swing-up phase. Hybrid / staged learning beats one dense reward for everything.

### Gustafsson 2016 — *RL on inverted double pendulum (CS229)*
- **Balance** worked with linear Q-learning: episode fail if |θ₁| > π/2 or cart OOB; γ=0.99; max **300** steps; init near upright.
- **Swing-up failed** entirely with the same linear FA (init hanging, 1000 steps). Author: swing-up needs nonlinearity / deep RL / better features.
- Costs: failed-episode cost + continuous angle-from-multiple-of-2π penalty.
- **Lesson:** learn **local balance first** (near-goal resets, single goal). Full swing-up from hanging is a harder second stage.

### Andrychowicz et al. 2017 — *HER*
- UVFA + **sparse** binary rewards; relabel failed trajectories with achieved goals (`final` / best: **`future` k=4**).
- Shaped dense rewards alone often fail or are brittle; HER shines when the only signal is “did we hit g?”.
- **Lesson:** our reward is already dense cos-alignment, so HER@0.8 is secondary. Prefer lower HER during a dense curriculum; raise HER if we switch toward sparse at-goal bonuses.

### Schaul et al. 2015 — *UVFA*
- One value/policy over `(s, g)`; goal embedding / concat architectures; generalizes across goals.
- **Lesson:** our 16-D obs (state + one-hot + target sin/cos) is the right UVFA shape. Multi-goal from day one is OK *after* a single-goal warmup so the shared torso sees success.

### Turcato et al. 2024 — *MB-RL underactuated double pendulum*
- MC-PILCO; cost = **saturated distance to goal angles** `1 − exp(−‖q−q_G‖²_Σ)` (ℓ_c=3); **no velocity in cost**.
- Short policy horizon (2–3 s) forces fast swing-up; later hand-off ideas to LQR.
- **Lesson:** prefer bounded / saturated distance-to-goal over unbounded quadratic state penalties.

### Duan et al. 2016 — *rllab continuous control*
- Double inverted pendulum **balance** reward: tip height + soft tip-x + tiny spin; **terminate when tip too low** (`y_tip ≤ 1`). Cart-pole swing-up: `r = cos(θ)`, **terminate `|x| > 3` with −100**.
- TRPO/TNPG strong on these tasks; DDPG needs reward rescaling (~0.1).
- **Lesson:** **hard track / fall termination** + bounded shaping. Unbounded live episodes with quadratic `x²` poison returns.

---

## (b) Why our reward likely collapsed

`goal_reward` (pre-fix) was roughly:

```text
r = (cosΔθ1 + cosΔθ2) − 0.02(x² + 0.1 ẋ²) − 0.002(ω1² + ω2²) − effort + sparse
```

1. **Unbounded cart `x` (primary).** Physics clamps velocities, **not** `x`. A failed policy drifts the cart; `0.02 x²` grows without bound. At `|x|≈25`, `r≈−14.5`/step → **~−5800** over 400 eval steps; `|x|≈30–40` lands in the observed **−8000…−11000** band. Matches TensorBoard collapse with `align≈0`.
2. **No track termination.** Duan/Gustafsson end the episode when `|x|` exceeds a limit. We time-cap only (`episode_len=1200`), so bad policies keep accruing quadratic penalties and corrupt advantages/value targets.
3. **Spin term fights energy pumping.** Spong/Xin *increase* kinetic energy during swing-up; `0.002 ω²` at large ω (allowed up to 50 rad/s) can dominate the ±2 align signal.
4. **All four goals + wild IC + HER@0.8 from step 0.** Gustafsson could not even solve single-goal hanging→upright with weaker FAs. Uniform multi-goal + HER on dense rewards dilutes early credit before any equilibrium is locally stable.
5. **Secondary:** mild resets are only ~20% near upright and not goal-conditioned, so most mass starts far from the requested equilibrium.

`align≈0` means the policy is not tracking goals — likely thrashing / drifting — while the **metric** `eval/reward` is dominated by cart quadratic blow-up, not by alignment failure alone.

---

## (c) Ranked next experiments

| Rank | Experiment | Concrete change | Why |
| --- | --- | --- | --- |
| **1** | Bound penalties + track fail | Soft-clip `x`/`ω` in reward; `r.clamp(±clip)`; done when `|x|>track_limit` | Stops −10k collapse; Duan/Gustafsson |
| **2** | Energy + denser align | `align_w`, light energy-to-goal (align − soft kinetic); lower spin weight | Xin/Spong swing-up; saturated distance (2024) |
| **3** | UU warmup curriculum | First `warmup_updates` only sample goal UU; then all four | Gustafsson balance-first; UVFA transfer |
| **4** | Near-goal resets | `--near-goal-p 0.5`: reset angles near requested goal | Local capture region (Spong LQR) |
| **5** | Lower HER early | `--her-ratio 0.3` (or 0 until warmup ends) | HER is for sparse; dense first |
| **6** | Longer local balance | After UU works: add UD/DU/DD one-by-one | Multi-equilibrium Xin |
| **7** | Sparse-heavy + HER future | If dense stalls: `r≈−1` until `at_goal`, HER future-k | HER paper |
| **8** | Specialist nets | Separate policy per goal if UVFA plateaus | README fallback |

**Implemented in this commit (ranks 1–4 as flags + defaults):** reward clip / soft penalties / energy term / track done / UU warmup / near-goal reset. See CLI in `train/train.py`.

### Exact next-run command (after current GCP job ends)

```bash
# On the training VM (or via ./train/gcp/train.sh after syncing this commit):
python3 train/train.py \
  --num-envs 4096 \
  --updates 400 \
  --rollout 128 \
  --episode-len 800 \
  --track-limit 4.0 \
  --reward-clip 8.0 \
  --align-w 1.5 \
  --energy-w 0.15 \
  --spin-w 0.0003 \
  --warmup-updates 80 \
  --warmup-goal UU \
  --near-goal-p 0.5 \
  --her-ratio 0.3 \
  --impulse-p 0.005 \
  --logdir runs \
  --checkpoint policies/checkpoint.pt \
  --out policies/policy.json
```

Helper: `scripts/next-train.sh` wraps the same knobs for GCP `train.sh` env overrides.

**Success gates (TensorBoard):** `eval/align` rising above ~0.5 within ~100 updates; `eval/reward` staying in roughly `[-2000, +2000]` (not −10k); `eval/at_goal/UU` nonzero before enabling full multi-goal pressure (warmup already does UU-first).

**Do not** start a second GPU train while the live paper-recipe job is still running (quota `GPUS_ALL_REGIONS=1`).


---

## (d) Live-run lesson: soft goal curriculum (2026-09-18)

Paper-recipe run (`warmup=80×UU`, then uniform multi-goal) fixed the cart-quadratic collapse, but TB shows a **UU regression at the hard warmup cut**:

- During UU warmup, `eval/align/UU` peaked ~0.19 (update 60) with `eval/reward/UU` ~+216.
- Right after warmup ended (update 80→100), UU goal fraction fell ~0.86→0.27 and `align/UU` dropped ~0.13→0.06; by update 200 `align/UU≈0.09` while hanging/partial goals lead (`at_goal/DD≈0.12`, `UD≈0.09`, `UU≈0.01`).

**Paper link:** Gustafsson — learn local balance on one equilibrium before full swing-up/multi-goal. Spong/Xin — hybrid swing-up then *local capture*; other equilibria are unstable attractors under a UU energy target. Abrupt 4-way mixing dilutes the capture basin before UU is stable.

**Next experiment (only if this run plateaus before a usable UU):** replace the hard warmup cut with an **annealed goal mix** — e.g. keep `P(UU)≥0.5` (or raise `--warmup-updates` to 150–200) until `eval/at_goal/UU` clears a gate, then unlock UD/DU/DD one-by-one (rank 6). Do **not** change knobs while `eval/reward` and `eval/align` are still rising.

---

## Track walls (2026-09-18)

User ask + Duan-style rail: physics now has elastic walls at `|x|=trackLimit` (default **2.4**, restitution **0.3**) in `train/physics.py` and `web/src/physics.js` via `shared/constants.json`. Cart no longer runs to infinity; demo can stay no-reset. Next GCP restart after the current paper-recipe run should pick this up via sync + `scripts/next-train.sh` (`--track-limit 2.4`).

---

## (e) Soft UU anneal after paper-recipe finish (2026-09-18)

Paper-recipe run `20260918-052042` **finished healthy** (no cart-quadratic collapse):

| Metric | Final (update 400) | Notes |
| --- | --- | --- |
| `eval/reward` | **~416** | Was −8900 on legacy run |
| `eval/align` | **~0.32** | Rising through run (best at end) |
| `align/UU` | ~0.15 (peak ~0.19 @60) | Hard cut at warmup=80 diluted UU |
| `at_goal/DD` / `UD` | ~0.18 / ~0.16 | Hanging/partial lead; UU ~0.01 |

**Paper link:** Gustafsson — protect local balance mass; Spong/Xin — hybrid swing-up then capture. Abrupt 4-way mix after 80 updates lets DD/UD dominate.

**Change implemented:** `--uu-bias` + `--anneal-updates` soft curriculum. After hard `--warmup-updates`, sample with `P(UU)=uu_bias` (rest split) for the anneal window (`anneal_updates=0` ⇒ rest of run). `scripts/next-train.sh` defaults: walls `track_limit=2.4`, fine-tune checkpoint, `warmup=40`, `uu_bias=0.55`.

**Next restart:** sync walled physics + this curriculum, **fine-tune** from VM `policies/checkpoint.pt` (do not cold-start; do not archive — this checkpoint is a keeper).

---

## (f) Soft UU bias on walls fine-tune (2026-09-18)

Live run `20260918-055004` (fine-tune from paper-recipe u400, walls `|x|=2.4`, `warmup=40×UU`, `uu_bias=0.55`, `anneal=0`):

| Metric | Paper-recipe end (u400) | Walls FT ~u320 |
| --- | --- | --- |
| `eval/reward` | ~416 | **~730** |
| `eval/align` | ~0.32 | **~0.50** |
| `align/UU` | ~0.15 | **~0.54** |
| `at_goal/UU` | ~0.01 | **~0.28** |
| `at_goal` UD/DU/DD | ~0.16/—/0.18 | **~0.21 / 0.41 / 0.22** |

**Paper link:** Gustafsson — keep balance-mass on the hard equilibrium; Spong/Xin — hybrid swing-up then capture. Sustained `P(UU)≥0.55` for the whole fine-tune (no hard 4-way cut) recovered UU without starving hanging/partial goals.

**Throughput note:** same recipe leaves T4 ~35–40% util / ~0.6 GB VRAM at 4096×128 (~3 s/update). Next iteration: **8192 envs × rollout 256** (already `scripts/next-train.sh` defaults) once this run finishes — do not kill while reward/align still rising.

---

## Perf TensorBoard scalars (next runs)

`train/train.py` logs under `perf/`: `sec_per_update`, `updates_per_sec`, `env_steps_per_sec`, `samples_per_update`, `num_envs`, `rollout`, and every 5 updates `gpu_util_percent`, `gpu_mem_used_mb`, `gpu_mem_total_mb`, `torch_cuda_allocated_mb`, `cpu_percent`. Use these to decide 8192→16384 bumps.


---

## (g) FT warmup dip is transient; Turcato short-horizon note (2026-09-18)

Live walls soft-UU fine-tune `20260918-061318` (8192×256 from archived healthy u400): during pure-UU warmup, logged `eval/reward` dipped toward ~0 / slightly negative while `align/UU` stayed ~0.6 — then by **u80–u100** recovered to **reward~740–792, align~0.50–0.53**, with `at_goal` UU/UD/DU/DD ~**0.36/0.22/0.39/0.26** (ahead of prior finished FT). Do **not** treat the mid-warmup reward dip as collapse.

**Turcato et al. 2024 (MC-PILCO / AI Olympics):** cost is saturated distance to goal angles with **no velocity in the cost**; they emphasize a **short policy horizon (2–3 s)** to force fast swing-up before local stabilization. If later `at_goal` plateaus while `align` is already high, prefer shortening `episode_len` (denser credit / faster capture pressure) over raising `spin_w`.

---

## (h) Soft-UU walls FT mid-run: all-four at_goal gate clearing (2026-09-18 ~01:35 CT)

Run `20260918-061318` at **u180/400** (8192×256, soft `uu_bias=0.55`): `eval/reward~804`, `align~0.53`, per-goal align UU/UD/DU/DD **0.54/0.50/0.57/0.49**, `at_goal` **0.41/0.29/0.46/0.35** — all four clearly nonzero and roughly balanced (UD still weakest). Soft bias after short UU warmup recovered UU *and* let hanging/partial goals keep rising (contrast hard-cut paper-recipe where UU collapsed to ~0.01).

**Paper link:** Gustafsson local-balance mass + Xin/Spong hybrid capture — sustained P(UU)≥0.55 protects the hard upright without starving other equilibria. **Throughput:** still ~7.6 s/update, GPU ~36% / 2.1 GB of 15 GB → after finish, sync `perf/*` train.py and FT again at **NUM_ENVS=16384** × rollout 256–384, `WARMUP_UPDATES=20`.

---

## Center / no rail-parking (2026-09-18)

Live demo @ ~u230: UU `at goal` with **x≈+2.4** (on the wall). Old `center=0.02 x²` was too weak vs align, so parking at the rail was free. Fix: `center_w=0.06` always + `center_hold_w=0.18` scaled by how aligned the links are (Xin-style cart regulation once upright). Next FT should use these defaults; do not cold-start.

---

## (i) Late walls FT ~u300: demo-range align, UD still the gate (2026-09-18 ~01:50 CT)

Run `20260918-061318` still live at **u300–u320/400** (8192×256, soft `uu_bias=0.55`, **no** stronger center knobs yet — those land on next FT): peak **`eval/reward~941`, `align~0.613`** at u300 (UU/UD/DU/DD align **0.62/0.53/0.67/0.63**); at u310 `at_goal` **0.43/0.26/0.44/0.40** with UD the clear weak link (align also dipped hardest on UD). By u320 recovered to reward~902 / align~0.59 with UD align back ~0.57.

**Insight:** soft-UU walls FT is past the “all-four nonzero” gate and into **interactive-demo-adjacent** align (≥0.55 sustained, peaks ~0.61) without collapsing UU. Remaining gap to the “damn good” bar (align ≳0.85 / at_goal ≳0.7) is **hold quality**, especially **UD capture** — not missing equilibria. This run’s TB has **no `perf/*` scalars** (older `train.py` on VM); after finish, sync main (perf logging + `center_w`/`center_hold_w`) then FT from this checkpoint at **16384×256**, `WARMUP_UPDATES=20`. If `at_goal` plateaus while align stays mid-0.5s, prefer Turcato-style shorter `episode_len` over more spin penalty.

---

## Walls are cart-only / inelastic (2026-09-18)

User: elastic rail bounce was letting UU prop itself at the wall (demo x≈+2.4 at goal). Walls must **not** act like a pole brace. Change: `wallRestitution=0` — clamp cart `x`, kill cart `ẋ`, leave pole states untouched (no pole–wall collision exists; bounce coupling was the cheat). Stronger center reward still pulls back to mid.


## (j) Finished walls soft-UU FT u400 → Phase A still open (2026-09-18 ~02:25 CT)

Run `20260918-061318` completed at **u400**: `eval/reward≈890`, `align≈0.575`, `at_goal` UU/UD/DU/DD **0.45 / 0.30 / 0.48 / 0.47**. All four nonzero; **UD remains the gate**. Still short of Phase A bars (align ≳0.85, at_goal ≳0.7 all four + mid-track return).

**Next:** keep Phase A — FT from this checkpoint with hard endstops + `center_w=0.06` / `center_hold_w=0.18` at **16384×256** (run `20260918-072313` after spot restart). No Phase B mid-episode switches yet. GPU ~35% / ~2.3 GB — bump `NUM_ENVS` further only on a later restart if util stays soft after warmup.


## (k) 32768 L4 FT: same warmup dip, soft GPU util (2026-09-18 ~03:25 CT)

Run `20260918-080608_ft-e32768-r256-hardwalls-center-uub055` (hard walls, center_w/hold, soft uu_bias=0.55, warmup=20) on `cartpole-train-od` L4: through **u20** reward~−154 / align~0 while **align/UU stayed ~0.68** (lesson g pattern at larger batch); by **u40** recovered to reward~**248**, align~**0.31**, at_goal UU/UD/DU/DD **0.38/0.10/0.35/0.13** — UD still the gate. **Do not redesign mid-run.**

**Throughput:** ~22 s/update, ~382k env-steps/s, GPU util ~**16%**, VRAM **7.8/23 GB** (~34%). Headroom remains after this 400-update stretch; bump envs/rollout only on a *natural* restart if Phase A still open — never kill an improving FT just to fill the L4.

**Xin 2008 reminder:** if energy settles off UU, the plant parks at unstable UD/DU/DD attractors — keep soft `uu_bias≥0.55` until Phase A bars clear.


## (l) Phase A gate = per-goal at_goal, not 1-step mean align (2026-09-18 ~03:30 CT)

Live L4 FT `20260918-080608_ft-e32768-r256-hardwalls-center-uub055` at **u60**: mean `eval/reward`/`align` dipped u50→u60 (**469→442**, **0.456→0.425**) while **`at_goal` UU/UD/DU/DD rose to ~0.41/0.21/0.40/0.25** (UD **0.10→0.21**). Do not treat a single-update mean-align dip as collapse while per-goal at_goal (esp. the UD gate) is climbing.

**Xin 2008:** if energy converges to a value ≠ `E_uu`, the plant remains at the **UD / DU / DD** equilibria (unstable attractors of the energy controller). That is why UD stays the hardest Phase A gate under soft multi-goal — keep `uu_bias≥0.55` and judge progress on **`eval/at_goal/{UU,UD,DU,DD}`**, not noisy mean align. No recipe change mid-run.


## (m) UD is the nearest wrong energy sink to UU (2026-09-18 ~03:50 CT)

Live L4 FT `20260918-080608_ft-e32768-r256-hardwalls-center-uub055` at **u110**: reward~**702**, align~**0.58**, `at_goal` UU/UD/DU/DD **~0.48/0.41/0.47/0.40** — big climb from u60 (reward~442 / align~0.43 / UD~0.21). UD remains weakest but **doubled again** (0.21→0.41). Still Phase A; leave recipe alone.

**Xin 2008 potential:** \(P=\beta_1\cos\theta_1+\beta_2\cos\theta_2\) with \(\beta_1>\beta_2\) (outer mass on longer lever). Ordering of equilibria energies is **UU > UD > DU > DD**. Under energy control targeting \(E_{uu}\), undershoot parks first at **UD** — the nearest incorrect unstable attractor — before DU/DD. That is a sharper reason UD is the standing Phase A gate than “one of three wrong attractors” alone (lesson l): protect UU mass (`uu_bias≥0.55`) and keep judging on per-goal `at_goal`, especially UD. No mid-run knob change.


## (n) HER can reinforce the UD energy sink under UU-biased mix (2026-09-18 ~04:07 CT)

Live L4 FT `20260918-080608_ft-e32768-r256-hardwalls-center-uub055` at **u160**: reward~**700**, align~**0.57**, `at_goal` UU/UD/DU/DD **~0.50/0.36/0.50/0.39** — oscillating near the u110 plateau (reward~702 / align~0.58 / UD~0.41), not collapsing. Soft `uu_bias=0.55` + `her_ratio=0.3` still on; leave knobs alone mid-run.

**HER (Andrychowicz 2017) × Xin 2008:** hindsight relabel replaces the intended goal with a goal actually reached in the episode. Under Xin potential ordering **UU > UD > DU > DD**, failed UU rollouts that undershoot energy most often land in **UD** (lesson m). Relabeling those as UD successes turns the nearest wrong attractor into a *dense* HER target — so a nonzero `her_ratio` can amplify the UD gate even while `uu_bias` protects on-policy sampling. That is distinct from “judge on per-goal at_goal” (l) or “UD is nearest sink” (m): it is a **replay** mechanism that can fight the curriculum.

**Next restart only (if UD plateaus while UU/DU/DD keep rising):** drop `her_ratio` (e.g. 0.3→0.1) or skip UU→UD / UU→DU relabels; do **not** raise `uu_bias` further first. No mid-run change while reward/align still oscillate in a rising band.


## (o) Visit≠hold: Spong capture region vs sparse@1.0 (2026-09-18 ~04:20 CT)

Live L4 FT `20260918-080608_ft-e32768-r256-hardwalls-center-uub055` at **u190**: reward~**691**, align~**0.56**, `at_goal` UU/UD/DU/DD **~0.52/0.35/0.53/0.41** — still oscillating in the u110–u180 band (peak reward~738 / align~0.59 @u180; UD align dipped 0.62→0.46 on this step). Phase A open; leave knobs alone mid-run.

**Spong 1995:** swing-up then switch to a *local* LQR whose capture region is small — arriving near the upright manifold is not the same as staying there. Our dense `align_w=1.5` pays for cos-proximity every step, while `sparse_bonus=1.0` only fires on `at_goal` (cos≥0.95). That asymmetry explains a sustained **align≳0.55 with at_goal stuck ~0.35–0.52**: the policy learns to *visit* goal neighborhoods but is under-paid for *holding* inside the capture ball — exactly Spong's hybrid gap, not a missing equilibrium.

**HER shape note (Andrychowicz 2017):** our `her_relabel_inplace` uses **end-of-rollout nearest achieved equilibrium**, not paper-`future` k=4. Under Xin UU>UD>DU>DD (lesson m/n), final-state HER preferentially stamps **UD** on failed UU episodes. Dropping `her_ratio` on the next natural restart (if UD still gates) remains preferred over switching to unfiltered `future` relabels.

**Next restart only (if still short of Phase A at u400):** raise hold pressure (`sparse_bonus` or `center_hold_w`) and/or drop `her_ratio` 0.3→0.1 — do **not** kill this FT early; GPU still ~16% / 7.8 GB (throughput bump only after finish).


## (p) Multi-goal interference: UU climb can steal UD hold (2026-09-18 ~04:31 CT)

Live L4 FT `20260918-080608_ft-e32768-r256-hardwalls-center-uub055` at **u220**: reward~**783**, align~**0.632** — broke the u110–u190 plateau (peak was ~738/0.59 @u180). Per-goal align UU/UD/DU/DD **~0.70/0.59/0.60/0.64**; `at_goal` **~0.54/0.38/0.50/0.47**. UU `at_goal` rose (0.52→0.54) while **UD `at_goal` dipped** (0.45@u200 → 0.38@u220) even as mean align/reward hit run highs. Phase A still open; leave knobs alone mid-run.

**UVFA (Schaul 2015) × Gustafsson local-balance:** a shared goal-conditioned policy has finite capacity; when soft `uu_bias=0.55` plus on-policy success pushes more mass into UU, gradient steps that improve UU capture can *regress* the UD hold manifold without collapsing mean align (align pays for visiting neighborhoods; `at_goal` pays for hold). Treat mean-align breakouts that coincide with UD `at_goal` dips as **interference**, not as Phase A progress — keep judging the gate on `eval/at_goal/UD`.

**Next restart only (unchanged from n/o):** if UD still gates at u400 while UU/DU/DD are ahead, drop `her_ratio` 0.3→0.1 and/or raise hold pressure (`sparse_bonus` / `center_hold_w`). Do not kill this FT; GPU still ~14% / 7.8 GB.


## (q) Post-breakout regression ≠ collapse; Spong capture still unstable (2026-09-18 ~04:47 CT)

Live L4 FT `20260918-080608_ft-e32768-r256-hardwalls-center-uub055` at **u260**: reward~**698**, align~**0.572**, `at_goal` UU/UD/DU/DD **~0.48/0.37/0.48/0.42**. Run high-water remains **u220** (reward~783 / align~0.632); then **u230–u250** correlated multi-goal dip (reward trough ~624 @u250, UU `at_goal` 0.54→0.42, DD 0.47→0.33) with partial recovery by u260. **UD `at_goal` stuck ~0.37** (peak since u200 was 0.45 @u200) — still the Phase A gate. Leave knobs alone mid-run.

**Spong 1995 × UVFA interference (p):** a shared policy that briefly enters a better capture region can leave it again — swing-up proximity ≠ stable local LQR hold. Mean-align/reward breakouts under soft multi-goal are **noisy high-water marks**, not new floors; judging Phase A on the post-peak trough would falsely call collapse while per-goal `at_goal` stays in the same band (complement to lesson l, which warned against treating dips *during* climbs as collapse).

**Next restart only (unchanged from n/o/p):** if UD still gates at u400, drop `her_ratio` 0.3→0.1 and/or raise hold pressure (`sparse_bonus` / `center_hold_w`). Do not kill this FT; GPU ~16% / 7.8 GB.


## (r) Intermediate-equilibrium competition: DU climb can steal UD (2026-09-18 ~05:04 CT)

Live L4 FT `20260918-080608_ft-e32768-r256-hardwalls-center-uub055` at **u310**: reward~**637**, align~**0.514**, `at_goal` UU/UD/DU/DD **~0.52/0.39/0.46/0.33**. Since lesson (q) @u260 the run recovered toward **u270–u300** (reward~735–730 / align~0.58–0.59; UU `at_goal` back ~0.55) then dipped again @u310. At **u300** DU `at_goal` spiked to **~0.56** while **UD crashed to ~0.33** (from ~0.42 @u290) and **DD stayed weak ~0.33–0.40**. Run high-water still **u220** (783 / 0.632). Phase A open; leave knobs alone mid-run.

**Xin 2008 × UVFA (Schaul 2015) × Turcato 2024:** Xin potential orders **UU > UD > DU > DD**, so UD and DU are *adjacent* intermediate attractors. Lesson (p) noted UU mass can steal UD hold; the u300 snapshot shows the same capacity fight **between the two intermediates** — a shared UVFA that improves DU capture can regress UD without collapsing mean align. DD (`at_goal` floor ~0.33) is now co-gating with UD. Turcato/MC-PILCO only swing-up+stabilize *one* unstable equilibrium per task; our four-way UVFA pays an interference tax they avoid by specialization.

**Phase A judge:** use **`min(at_goal/{UU,UD,DU,DD})`**, not mean align or the strongest goal. **Next restart only (unchanged from n/o/p/q):** if min still ≪0.7 at u400, drop `her_ratio` 0.3→0.1 and/or raise hold pressure; optionally consider goal-specialist heads if interference persists. Do not kill this FT; GPU ~17% / 7.8 GB; ~90 updates left (~22 s/u).


## (s) Late FT: HER redistributes among wrong attractors; mean recovery ≠ min-gate close (2026-09-18 ~05:21 CT)

Live L4 FT `20260918-080608_ft-e32768-r256-hardwalls-center-uub055` at **u360**: reward~**750**, align~**0.590**, `at_goal` UU/UD/DU/DD **~0.54/0.39/0.50/0.44**. Recovered from the u310 dip (636/0.514); DD `at_goal` climbed **0.33→0.44** while **UD stayed the min-gate ~0.39–0.43**. Run high-water still **u220** (783 / 0.632). ~40 updates left (~22 s/u). Phase A open; **do not kill** — leave knobs alone mid-run.

**Andrychowicz 2017 × Xin 2008 (new angle on n/o):** classic HER assumes achieved goals form a *useful continuum* toward the intended goal. Our goals are **four discrete unstable equilibria** with Xin energy order UU>UD>DU>DD, so “achieved” states on failed UU/DD episodes are usually **other wrong attractors**, not waypoints on a path to the requested hold. Late in a soft multi-goal FT, nonzero `her_ratio` therefore mostly **reshuffles credit among UD/DU/DD** (mean reward/align and the non-min goals bounce back) while **`min(at_goal)` stays stuck** — exactly the u310→u360 pattern. That is stronger than “HER amplifies UD” (n): even when UD is not the HER stamp, discrete-equilibrium HER still fails to close the Phase A min-gate.

**Spong visit≠hold still binds:** align~0.59 vs min `at_goal`~0.39.

**Next natural restart (at/after u400 if Phase A still open):** execute prior plan — drop `her_ratio` 0.3→0.1 and/or raise hold pressure (`sparse_bonus` / `center_hold_w`); keep `uu_bias≥0.55`. Do **not** start another identical 400-u stretch with the same HER mix. GPU still ~13% / 7.8 GB — throughput bump only after this FT finishes.


## (t) Finish stretch: DD re-co-gates; Phase A will miss u400 (2026-09-18 ~05:34 CT)

Live L4 FT `20260918-080608_ft-e32768-r256-hardwalls-center-uub055` at **u390/400**: reward~**678**, align~**0.548**, `at_goal` UU/UD/DU/DD **~0.56/0.38/0.46/0.34**. UU late high (~0.56) while **min-gate flipped to DD ~0.34** (was UD-led ~0.39 @u360; DD had briefly climbed to 0.44 then relapsed). UD still ~0.38 — neither clears 0.7. Run high-water still **u220** (783 / 0.632). ~10 updates left; **do not kill** — leave knobs alone mid-run.

**Confirms (s)/(r):** discrete-equilibrium HER keeps reshuffling credit among wrong attractors into the finish; mean reward/align bounce (u360–u380 ~715–736 / ~0.57–0.59) does not lift `min(at_goal)`. Phase A bars (align ≳0.85, at_goal ≳0.7 all four) will not clear at u400.

**Next natural restart (after u400):** drop `her_ratio` 0.3→**0.1** (scripts/next-train.sh), keep `uu_bias≥0.55`, optionally raise hold pressure later if min-gate still stuck. GPU ~13% / 7.8 GB.
