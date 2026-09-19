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

## (u) Lower HER preserves checkpoint manifolds through UU warmup (2026-09-18 ~06:00 CT)

New FT `20260918-104353_ft-e32768-r256-hardwalls-center-uub055-her01` (`her_ratio=0.1` from prior u400 ckpt) at **u40**: reward~**880**, align~**0.677**, `at_goal` UU/UD/DU/DD **~0.59/0.40/0.59/0.50**. Already above the parent FT's u400 finish (reward~786 / align~0.622 / at_goal ~0.58/0.39/0.53/0.42) and far above that run's own post-warmup u40 (reward~248 / align~0.31 / UD~0.10). Warmup dip still exists (u20 reward~25 / align~0.11) but rebound is sharper and all-four `at_goal` recover together. Phase A still open (bars align≳0.85 / at_goal≳0.7); leave knobs alone mid-run.

**Andrychowicz 2017 × Xin 2008 (n/s/t confirmed early):** discrete-eq HER at 0.3 was not only late-run reshuffling among wrong attractors — it also **fought warm-start transfer**. During UU-only warmup, failed non-UU episodes still get end-of-rollout HER stamps onto UD/DU/DD; at `her_ratio=0.3` that dilutes the checkpoint's already-learned hold manifolds before multi-goal training resumes. Dropping to **0.1** lets the ckpt ride through warmup with less attractor reshuffle, so u40 already sits at prior high-water rather than rebuilding from a thrashed UD gate.

**Spong 1995:** a warm-started local capture region is fragile; heavy hindsight relabel during a single-goal warmup is the opposite of protecting it. Prefer low/zero HER across UU warmup when fine-tuning from a multi-goal ckpt; raise HER only if later switching toward sparse@goal.

**Do not redesign mid-run.** GPU still ~16% / ~4.3 GB used (headroom); throughput bump only on a later natural restart if Phase A still open after this 400-u stretch.

## (v) Low HER lets min-gate climb with the mean; UVFA interference is bidirectional (2026-09-18 ~06:20 CT)

Live L4 FT `20260918-104353_ft-e32768-r256-hardwalls-center-uub055-her01` (`her_ratio=0.1`) at **u90**: reward~**905**, align~**0.691**, `at_goal` UU/UD/DU/DD **~0.59/0.51/0.57/0.51**. Vs prior fire @u40 (880 / 0.677 / 0.59/0.40/0.59/0.50): **UD min-gate +0.11** and DD co-lifted; run high-water ~**u70–u80** (reward~926–928 / align~0.71). Phase A still open (bars align≳0.85 / at_goal≳0.7); **leave knobs alone** mid-run.

**Andrychowicz 2017 × Xin 2008 (extends n/s/t/u past warmup):** parent FT at `her_ratio=0.3` kept `min(at_goal)` stuck ~0.34–0.39 while mean reward/align bounced. At 0.1, **min-gate rises in lockstep with the mean** through the first ~90 multi-goal updates — so discrete-eq HER dilution was the dominant Phase A blocker, not only a warmup-transfer issue (u). Spong visit≠hold (o) still binds (align~0.69 vs min `at_goal`~0.51) but the gap is narrowing together rather than mean-only thrash.

**UVFA (Schaul 2015) bidirectional blip:** @u80 UU `at_goal` dipped **0.59→0.51** while UD/DU/DD briefly peaked (~0.49/0.61/0.55); @u90 UU recovered and the four re-balanced near ~0.51–0.59. Complements lesson (p) (UU stealing UD): under a softer HER mix the capacity fight runs **both ways**. Treat single-goal dips during a rising band as interference noise, not collapse (q).

**Next restart only (if Phase A still open at/after u400):** raise hold pressure (`sparse_bonus` / `center_hold_w`) before raising HER again; keep `her_ratio≤0.1` and `uu_bias≥0.55`. GPU ~16% / ~4.3 GB — throughput bump only after this FT finishes.


## (w) Post-peak trough: low HER still allows DU↔UD interference (2026-09-18 ~06:30 CT)

Live L4 FT `20260918-104353_ft-e32768-r256-hardwalls-center-uub055-her01` (`her_ratio=0.1`) at **u120**: reward~**885**, align~**0.682**, `at_goal` UU/UD/DU/DD **~0.578/0.418/0.606/0.487**. Vs prior fire @u100 (908 / 0.693 / 0.545/0.477/0.593/0.507): soft multi-goal dip (also u110 trough reward~852 / align~0.664). Run high-water still **u70–u80** (reward~926–928 / align~0.71). **UD min-gate** 0.51@u90 → 0.42@u120 while **DU hit run-high `at_goal`~0.61**. Phase A open; **leave knobs alone** mid-run.

**Xin 2008 × UVFA (extends r/v):** dropping HER to 0.1 fixed the *systematic* min-gate freeze (v) but does **not** erase adjacent-intermediate capacity fights. The first post-peak trough under low HER re-shows lesson (r)'s DU↔UD steal — mean reward/align stay in the mid-880 / ~0.68 band while `min(at_goal)` slides. Treat this as **interference noise on a rising band**, not HER failure and not collapse (q).

**Spong visit≠hold (o) still binds:** align~0.68 vs min `at_goal`~0.42. **Next restart only (unchanged from v):** raise hold pressure (`sparse_bonus` / `center_hold_w`) before raising HER; keep `her_ratio≤0.1` and `uu_bias≥0.55`. GPU ~16% / ~4.3 GB — no throughput bump mid-run.



## (x) Post-trough new high-water under low HER (2026-09-18 ~06:49 CT)

Live L4 FT `20260918-104353_ft-e32768-r256-hardwalls-center-uub055-her01` (`her_ratio=0.1`) at **u170**: reward~**962**, align~**0.725**, `at_goal` UU/UD/DU/DD **~0.632/0.524/0.592/0.503**. Broke prior run high-water **u70–u80** (~926–928 / ~0.71). Recovered from the u110–u150 trough (w: UD min-gate had slid to ~0.42 @u120). **All four `at_goal` ≥0.50** for the first sustained snapshot this FT; min-gate still **UD ~0.524**. Phase A open (bars align≳0.85 / at_goal≳0.7); **leave knobs alone** mid-run.

**Spong 1995 × Andrychowicz 2017 × Xin 2008 (extends u/v/w):** under `her_ratio=0.1`, a DU↔UD interference trough is not a ceiling — the subsequent climb set a **new** mean high-water *and* lifted all four holds together. Contrast the parent FT at `her_ratio=0.3`, where late mean bounces never sustainably beat the u220 peak while `min(at_goal)` stayed stuck ~0.34–0.39 (s/t). Low discrete-eq HER therefore does three related things: preserves ckpt manifolds through UU warmup (u), lets min-gate climb with the mean (v), and lets capture basins **expand past earlier peaks** after an interference blip because recovery credit is not reshuffled onto wrong Xin attractors.

**Spong visit≠hold (o) still binds:** align~0.725 vs min `at_goal`~0.503 (gap narrowing vs 0.68 vs 0.42 @u120). **Next restart only (unchanged from v/w):** raise hold pressure (`sparse_bonus` / `center_hold_w`) before raising HER; keep `her_ratio≤0.1` and `uu_bias≥0.55`. GPU ~16% / ~4.3 GB — no throughput bump mid-run (~230 updates left @ ~22 s/u).


## (y) Long dense episodes buy late visits; Turcato short-horizon forces early capture (2026-09-18 ~07:06 CT)

Live L4 FT `20260918-104353_ft-e32768-r256-hardwalls-center-uub055-her01` (`her_ratio=0.1`) at **u210**: reward~**938**, align~**0.706**, `at_goal` UU/UD/DU/DD **~0.579/0.498/0.602/0.527**. Vs prior fire @u170 (962 / 0.725 / 0.632/0.524/0.592/0.503): soft post-peak oscillation (u180–u190 dip then u200 rebound ~962/0.722). Run high-water still **u170/u200** (~962 / ~0.72–0.725). **UD min-gate ~0.50**. Phase A open; **leave knobs alone** mid-run.

**Turcato / MC-PILCO 2024 × Spong visit≠hold (o):** competition eval is T=10 s, but they deliberately optimize the swing-up policy on a **much shorter horizon (T=2–3 s)** so the optimizer must reach the unstable equilibrium *early*, then hand off to LQR inside the capture region. Our `episode_len=800` at `dt≈1/120` ≈ **6.7 s** of dense `align_w` credit — long enough that a policy can score mid/high align by **late-episode neighborhood visits** without early hold (align~0.71 vs min `at_goal`~0.50). That is the same Spong gap, now with a concrete horizon lever: short optimization horizons force capture timing; long dense episodes subsidize visit≠hold.

**Also (Turcato saturated cost):** their `1−exp(−‖q−q_G‖²_Σ)` (ℓ_c=3) has **no velocity term** yet still encourages zero-velocity arrival — our spin_w is already tiny (0.0003); next-restart hold pressure should come from `sparse_bonus` / `center_hold_w` (or optional shorter `episode_len`), not from re-raising spin.

**Next restart only (unchanged from v/w/x + optional horizon):** raise hold pressure before raising HER; keep `her_ratio≤0.1` and `uu_bias≥0.55`; consider shorter `episode_len` only if min-gate still stalls after hold-pressure bump. GPU ~16% / ~4.3 GB — no throughput bump mid-run (~190 updates left @ ~22 s/u).


## (z) Low HER raises the interference-crash floor, not the crash rate (2026-09-18 ~07:25 CT)

Live L4 FT `20260918-104353_ft-e32768-r256-hardwalls-center-uub055-her01` (`her_ratio=0.1`) at **u260**: reward~**900**, align~**0.676**, `at_goal` UU/UD/DU/DD **~0.560/0.515/0.540/0.523**. Soft post-peak band continues (u200/u230 near high-water ~961–962 / ~0.72; u240–u250 UD crash to ~0.46 then recover). Run high-water still **u170** (reward~962 / align~0.725). **UD min-gate ~0.515** (best this stretch **0.528 @u230**). Phase A open; **leave knobs alone** mid-run.

**Xin 2008 × UVFA (Schaul 2015) × Andrychowicz 2017 (extends w/x/y):** dropping HER to 0.1 did not erase ~20-update DU↔UD interference crashes (u240–u250 UD 0.53→0.46 while DU/UU held). What changed vs parent `her_ratio=0.3` is the **crash floor**: UD bottoms ~0.46 and recovers within ~10 updates instead of freezing at ~0.34–0.39 for the rest of the FT (s/t). Low discrete-eq HER therefore raises the *floor of interference crashes*, not their occurrence — judge Phase A on whether successive crash troughs trend up, not on whether troughs vanish.

**Spong 1995 visit≠hold still binds:** align~0.676 vs min `at_goal`~0.515; full-episode `at_goal` averages capture-enter/leave cycles over `episode_len=800` (complements Turcato short-horizon lever in y).

**Next restart only (unchanged from v–y):** raise hold pressure (`sparse_bonus` / `center_hold_w`) before raising HER; keep `her_ratio≤0.1` and `uu_bias≥0.55`; optional shorter `episode_len` only if min-gate stall persists after hold bump. GPU ~24% / ~4.3 GB — no throughput bump mid-run (~140 updates left @ ~22 s/u).


## (aa) Reward high-water can lead min-gate by a full interference cycle (2026-09-18 ~07:44 CT)

Live L4 FT `20260918-104353_ft-e32768-r256-hardwalls-center-uub055-her01` (`her_ratio=0.1`) at **u310**: reward~**973**, align~**0.719**, `at_goal` UU/UD/DU/DD **~0.591/0.523/0.619/0.551**. **New run high-water on reward** (prior ~962 @u170/u200); align nearly ties the prior peak (~0.725). UD min-gate just recovered from the u290 trough (~0.458) to **~0.523** — same crash-floor band as lesson (z), not a Phase A clear. Phase A open; **leave knobs alone** mid-run.

**Spong 1995 × Turcato 2024 × Xin/UVFA (extends x/y/z):** under low discrete-eq HER, mean **reward can set a new high-water while `min(at_goal)` is still only climbing out of an interference trough**. Episode return is dominated by dense `align_w` (and center terms) over `episode_len=800`, so a policy that visits neighborhoods more often / with better late-episode align prints a bigger reward before hold fractions catch up. Treat reward/align HWs as **leading** signals; Phase A still judges on **`min(at_goal/{UU,UD,DU,DD})`** and whether successive UD crash floors trend up (z), not on reward alone.

**Practical:** do not interpret a mid-run reward spike as reason to raise HER or kill the FT — and do not declare Phase A progress from reward HW if the min-gate has not moved. **Next restart only (unchanged from v–z):** raise hold pressure (`sparse_bonus` / `center_hold_w`) before raising HER; keep `her_ratio≤0.1` and `uu_bias≥0.55`; optional shorter `episode_len` only if min-gate stalls after hold bump. GPU ~15% / ~4.3 GB — no throughput bump mid-run (~90 updates left @ ~22 s/u).


## (ab) Finish-stretch: non-min goals can peak while min-gate stays flat (2026-09-18 ~08:08 CT)

Live L4 FT `20260918-104353_ft-e32768-r256-hardwalls-center-uub055-her01` (`her_ratio=0.1`) at **u380** (~20 updates left): reward~**953**, align~**0.714** (new align HW **0.729 @u360**, nearly tying reward HW **973 @u310/u360**). `at_goal` UU/UD/DU/DD **~0.581/0.523/0.593/0.556** — **DD set a run peak @u380**; UU spiked **0.633 @u370**; **UD min-gate still oscillates 0.46–0.55** (troughs @u350/~0.467, @u370/~0.464 — same crash floor as lesson z). Phase A bars (align≳0.85 / at_goal≳0.7 all four) will miss u400. **Leave knobs alone** for the last ~20 updates.

**Xin 2008 × Spong 1995 × UVFA (extends aa/z):** late in a soft multi-goal FT, **non-min equilibria can print new `at_goal` highs in the finish stretch while `min(at_goal)` is frozen**. DD (Xin lowest potential) and brief UU spikes are easier capture expansions than holding the mixed UD intermediate; mean align/reward HWs (aa) therefore co-occur with *asymmetric* per-goal peaks that do **not** move the Phase A gate. Judging “almost there” from DD/UU late peaks would false-positive Phase A progress.

**Confirms next natural restart plan (v–aa):** after u400, raise hold pressure — bump `center_hold_w` **0.18→0.30** in `scripts/next-train.sh` (align-ramped mid-track term; Spong capture needs stay-near-goal, not more HER). Keep `her_ratio≤0.1`, `uu_bias≥0.55`. Optional shorter `episode_len` only if min-gate still stalls after the hold bump. GPU ~16% / ~4.3 GB — no throughput bump mid-run; consider env bump only on a later restart if still short after hold-pressure FT.



## (ac) her01 finish spike still misses Phase A; hold-pressure FT started (2026-09-18 ~08:44 CT)

Prior L4 FT `20260918-104353_ft-e32768-r256-hardwalls-center-uub055-her01` (`her_ratio=0.1`, `center_hold_w=0.18`) **finished u400**: reward~**1015**, align~**0.744** (new run HWs; prior ~973 / ~0.729 @u310–u360). `at_goal` UU/UD/DU/DD **~0.618/0.553/0.636/0.551** — **UD still min-gate ~0.55**; Phase A bars (align ≳0.85, at_goal ≳0.7 all four) **missed**. Finish stretch confirms (ab): mean/align can print a late spike while min-gate stays in the 0.46–0.55 crash-floor band (z).

**Spong 1995 visit≠hold × Xin 2008:** align~0.74 vs min `at_goal`~0.55 is still a capture-enter/leave gap, not a near-Phase-A miss. Low HER let the mean climb (v–x) but did not alone buy sustained hold.

**Action taken (natural restart):** started `20260918-134605_ft-e32768-r256-hardwalls-center-ch030-uub055-her01` from the u400 ckpt with `center_hold_w` **0.18→0.30**, keep `her_ratio=0.1`, `uu_bias=0.55`, NUM_ENVS=32768. Leave knobs alone mid-run; judge on whether UD crash floors trend up and whether min `at_goal` closes on 0.7. Throughput bump only if this FT finishes still short with VRAM headroom.


## (ad) Early ch030 hold FT: UD floor recovers faster post-warmup (2026-09-18 ~09:11 CT)

Live L4 FT `20260918-134605_ft-e32768-r256-hardwalls-center-ch030-uub055-her01` (`center_hold_w=0.30`, `her_ratio=0.1`) at **u60**: reward~**873**, align~**0.696** (early peak **u50** ~943 / ~0.737); `at_goal` UU/UD/DU/DD **~0.557/0.487/0.551/0.552**. Warmup trough @u20 (reward~398 / align~0.41 / UD~0.21) then rebound; **UD min-gate ~0.487** already above parent her01's early multi-goal UD (~0.40 @u40) and near that run's mid-band floor (~0.46–0.55). Mean still below parent u400 finish (1015 / 0.744). Phase A open; **leave knobs alone** mid-run.

**Spong 1995 × Xin 2008 (extends v/ac):** raising align-ramped `center_hold_w` 0.18→0.30 is a **capture-stay** lever (keep cart near mid-track once aligned), orthogonal to discrete-eq HER. Early signal: after UU warmup, the UD intermediate recovers its hold floor faster without thrashing mean reward/align transfer from the ckpt. That matches Spong's local capture region needing *stay* pressure, not more hindsight relabel. Xin potential ordering still makes UD the nearest wrong sink — judge the bump on whether successive UD crash floors trend up toward 0.7 over the rest of this 400-u FT, not on the u50 reward spike (aa).

**Do not redesign mid-run.** GPU ~15% / ~4.3 GB — throughput bump only after this FT finishes still short. Keep `her_ratio≤0.1`, `uu_bias≥0.55`; optional shorter `episode_len` only if min-gate still stalls after this hold-pressure stretch.

## (ae) Edge-case curriculum is the new deal (2026-09-18 ~09:42 CT)

Mean align ~0.71–0.74 under near-goal-heavy FT is mostly **local capture** diluted by long hang→goal stretches — Spong/Xin swing-up was never the training distribution. New Phase ("edge"):

- **`hang_start_p≈0.45`**: reset near hanging (θ≈π) so the policy must inject energy toward the assigned goal (DD→UU and hang→any).
- **`wrong_eq_p≈0.25`**: start at a *different* discrete equilibrium than the goal (anywhere↔anywhere without mid-flight flip).
- **`goal_switch_p≈0.002`/step**: mid-episode goal change without reset (interactive picker).
- Drop **`near_goal_p` to ~0.15**, raise **`energy_w≈0.35`**, **`episode_len≈1200`**, slightly harder impulses — keep `her_ratio≤0.1`, `center_hold_w≥0.30`, `uu_bias≥0.55`.

Apply on natural restart only; leave improving mid-run alone.


## (af) Mid ch030: mean rebounds to early HW; min-gate still her01 crash floor (2026-09-18 ~09:46 CT)

Live L4 FT `20260918-134605_ft-e32768-r256-hardwalls-center-ch030-uub055-her01` (`center_hold_w=0.30`, `her_ratio=0.1`, near-goal-heavy) at **u160**: reward~**920**, align~**0.725** (early HW still **u50** ~943 / ~0.737); `at_goal` UU/UD/DU/DD **~0.596/0.491/0.591/0.471**. Recovered from u120–u140 trough (align~0.67–0.69). **Min-gate flipped to DD ~0.471** (UD ~0.491) — same 0.44–0.55 crash-floor band as parent her01 (z/ab), not a Phase A lift. Phase A open; **leave knobs alone** mid-run (~240 updates left @ ~22 s/u).

**Spong 1995 × Xin 2008 × (ae):** raising `center_hold_w` 0.18→0.30 under `near_goal_p=0.5` bought faster post-warmup UD floor recovery (ad) but by mid-FT the mean is again **local-capture saturated** (align~0.72–0.74 band) while `min(at_goal)` stays in the her01 interference floor. Hold pressure alone does not force hang→goal energy injection or wrong-eq transitions — those ICs are still rare under near-goal-heavy sampling. Confirms edge curriculum (ae) as the next natural-restart lever, not another mid-run hold bump.

**Staged for next natural restart:** VM `scripts/next-train.sh` synced to edge defaults (`hang_start_p=0.45`, `wrong_eq_p=0.25`, `goal_switch_p=0.002`, `near_goal_p=0.15`, `energy_w=0.35`, `episode_len=1200`, `her_ratio=0.1`, `center_hold_w=0.30`). Do **not** kill this improving ch030 FT. GPU ~16% / ~4.3 GB.


## (ag) Multi-strategy mid-run: edge holds, swing dips, hot steals UU (2026-09-18 ~10:52 CT)

Live L4 triple FT (16k envs each, GPU ~99%/5.6GB):
- **edge** `…edge-hang-xeq-gsw-ch030-her01` @**u160**: reward~**1082**, align~**0.767**; `at_goal` UU/UD/DU/DD ~**0.550/0.508/0.644/0.534** (min **UD**). Flat/slightly up vs ~u60 (1085/0.759 / min UD~0.517).
- **swing** `…swing-hang070-ew045-uub070-her01` @**u160**: reward~**948**, align~**0.659**; `at_goal` ~**0.602/0.447/0.483/0.488** (min **UD**). **Regressed** vs ~u60 (1014/0.692 / UD~0.427) — mean down, DU down hard; UU hold still the swing specialty (~0.60).
- **hot** `…hot-lr1e3-clip03-edge` @**u260** (rollout 128, faster clock): reward~**1046**, align~**0.750**; `at_goal` ~**0.438/0.554/0.643/0.467** (min flipped to **UU**). vs ~u70 (1032/0.752 / min DD~0.488): UD/DU up, **UU collapsed ~0.52→0.44**.

**Turcato specialization × UVFA interference (q/p):** three ckpts on one GPU are doing what a single shared UVFA cannot — edge keeps the balanced min-gate, swing protects UU under hang-heavy energy, hot's aggressive PPO (lr=1e-3, clip=0.3) is **re-allocating** capacity toward UD/DU at UU's expense. Treat swing mean dip and hot UU collapse as **strategy divergence**, not a reason to kill either mid-run (lesson l/o: mid-run troughs ≠ collapse). Promote later by `min(at_goal)` when a stretch finishes; do **not** revive ch030 brute.

**Action:** leave all three alone through their 400-u stretches. GPU saturated — no fourth train.py. Phase A still open (align≪0.85, min at_goal≪0.7).

## (ah) Hot UU recovers; strategy min-gates diverge further (2026-09-18 ~11:16 CT)

Live L4 triple FT (16k envs each, GPU ~99%/5.6GB; ~8.4h VM up):
- **edge** `…edge-hang-xeq-gsw-ch030-her01` @**u220**: reward~**1070**, align~**0.759**; `at_goal` UU/UD/DU/DD ~**0.595/0.499/0.643/0.478** (min flipped to **DD**). vs ~u160 (1082/0.767 / min UD~0.508): mean flat; UU up; DD dipped into min-gate.
- **swing** `…swing-hang070-ew045-uub070-her01` @**u220**: reward~**1013**, align~**0.699**; `at_goal` ~**0.622/0.476/0.598/0.434** (min **DD**). **Recovered** vs ~u160 trough (948/0.659 / DU~0.483) — UU specialty ~0.62; DU back; DD now weakest.
- **hot** `…hot-lr1e3-clip03-edge` @**u370** (~30 u left, ~5 min): reward~**1050**, align~**0.761**; `at_goal` ~**0.536/0.475/0.597/0.537** (min **UD**). **UU recovered** from collapse ~0.438@u260 → **0.536** (extends ag: mid-run trough ≠ permanent loss under hot PPO).

**Turcato specialization × lesson (ag)/(l):** swing mean dip and hot UU crash both reversed without knob changes — confirms leave-alone through 400-u. Best `min(at_goal)` still **edge ~0.478** (swing 0.434, hot 0.475). Phase A still open.

**Action:** leave edge/swing alone (~65 min left). Armed one-shot hot watcher to restart from `checkpoint-hot.pt` on natural u400 finish (no idle under budget). Do **not** revive ch030 brute. Promote best min-gate → `checkpoint.pt` when stretches finish.


## (ai) No walls / void respawn (2026-09-18 ~11:55 CT)

Hard endstops let policies prop UU / pump against the rail. Demo goal: leave the track → fall into the void → respawn. Training matches: **no clamp walls** in `physics.step`; `|x| > trackLimit` still ends the episode (train reset / web `offTrack` respawn). Fine-tune from the best edge ckpt on this plant.

## (aj) Catastrophic void-death (2026-09-18 ~12:14 CT)

Without walls the policy ran off-track cheaply under a −2 OOB hitch (clipped away by `reward_clip=8`). Fix: `--oob-penalty` default **20**, applied **after** reward clip so terminal void-death stays ~−20, plus episode end. Log `train/oob_rate`. Restart nowalls FT with this.

## (ak) Edge+hot finish → restart on oob20 nowalls (2026-09-18 ~12:40 CT)

Edge stretch finished **u400**: reward~**1077**, align~**0.750**, `at_goal` UU/UD/DU/DD ~**0.70/0.78/0.76/0.77** (best min_at_goal ~**0.74** @u90 — keeper). Hot stretch2 also finished **u400**: reward~**1073**, align~**0.769**, `at_goal` ~**0.71/0.80/0.78/0.79** (best min ~**0.75** @u10). Swing left dead (do not revive).

**Nowalls** FT `…nowalls-oob20-edge` @**u70**: reward~**977**, align~**0.638**, `at_goal` ~**0.55/0.60/0.67/0.73** (min **UU**) — climbing from cold transfer; leave alone.

**Action:** promoted finished edge → `checkpoint.pt` / demo `policy.json`. Synced `scripts/next-train.sh` (oob20). Restarted **edge + hot** from their ckpts with `--oob-penalty 20` on the nowalls plant (u1 transfer dip reward~627–700 / align~0.44–0.49 — expected; same recovery curve as nowalls). Triple alive (GPU ~99%/6GB). Prefer promote from nowalls once its min_at_goal beats edge.

## (al) DU↔DD fold bias (2026-09-18 ~21:00 CT)

Demo failure: settled in **DU** (`th1=π, th2=0`) and never folds outer link into **DD**. Align stays decent; sparse hold on the new goal fails. Fix: `--fold-pair-p` (default recipe **0.55**) biases wrong-eq starts and mid-episode goal flips toward outer-link partners **DU↔DD** and **UU↔UD**. Also bump `--goal-switch-p` 0.002→0.004 so interactive toggles show up more often.

## (am) Transition-only curriculum (2026-09-18 ~23:50 CT)

Plateau on mixed edge/fold FT: align~0.70 / min_at_goal stuck ~0.45–0.65. Demo money is **eq→eq switches** (esp. DU→DD). New mode `--transition-only`: every reset spawns near discrete A with goal B≠A (uniform over 12 directed pairs); mid-episode flips always change goal (`goal_switch_p≈0.01`). No hang/near-goal/fold mix. Script: `scripts/next-train-transitions.sh` + `continue-xonly.sh`.


## (an) Triple plant: multi-eq first, not transition-only (2026-09-19 ~00:10 CT)

Cart-**triple** is a new plant (`physics_triple` / `goals_triple` / `train_triple`, `OBS_DIM=25`, 8 goals DDD…UUU). Overnight plan: share the L4 with double **xonly** (leave xonly alone) and run **normal** multi-eq PPO on triple — hang starts, energy, near-goal, soft UUU bias — the same curriculum that taught the double its 4 equilibria **before** xonly. Do **not** default triple to `--transition-only` (56 pairs); that is a later phase after local capture. Scripts: `next-train-triple.sh` / `continue-triple.sh`. Two parallel triple jobs (vary `lr` or `hang_start_p`) OK if `nvidia-smi` shows headroom alongside xonly.

## (ao) Triple redesign: product + UUU-first (2026-09-19 ~11:30 CT)

Overnight UVFA+hang (`hang_start_p≈0.45–0.60`) parked at mean align~0.2 with **DDD~0.5 / UUU~0.01** — hang curriculum taught “stay down.” Papers that work (Lim KIEE 2025 product on world angles; fawraw M2 UUU-first + cart barrier + fall_grace≈20) keep **PPO** but steal reward+curriculum.

**Shipped:** `--reward-mode product` (Lim soft coeffs + Baek α_th=0.5 floors + UP×5/DOWN×1 geo weights), `--cart-barrier-coef 50`, `--fall-grace-steps 20`, `--init-mode`, `--warmup-hang-start-p 0`, post-warmup `hang_start_p≈0.10`. Scripts `next-train-triple.sh` / `next-train-triple-uuu.sh` / continue-a (defaults) / continue-b (W_UP=8, lr=5e-4). **Cold-start** triple A/B — product scale ≠ additive ckpt. Leave double **xonly** alone.

---

## Triple P0/P1 (2026-09-19 ~14:35 CT) — progress + VER + right meter

**Stuck symptom:** UUU-only + force40/60 still `at_goal/UUU≈0` on harsh random-IC eval; train may be pumping but we were reading the wrong meter.

**Shipped (keep PPO):**
1. **`--progress-w`** (default 1.0 in product): dense `Δ mean cos-align` toward goal inside `product_reward` (fawraw M4 / Baek dense swing signal).
2. **`--flip-augment`** (default on): Baek VER left–right duplicate of PPO batch after GAE; flip `x,ẋ,θ,θ̇,F`; recompute logπ. Documented planar symmetry.
3. **`--eval-curriculum`**: TensorBoard `eval/near_target/at_goal/UUU` + `eval/hang/...` separate from random-IC `eval/*`.
4. **Slot recipes:** A=swing (hang+near, f50), B=hold (near_target), C=combo — all progress+flip; cold markers `*-progress-flip-v5`.

**Watch:** `eval/near_target/at_goal/UUU` for hold progress; `eval/hang/align/UUU` for swing. Stage-gate still UUU hold ≳0.80 before multi-eq.

