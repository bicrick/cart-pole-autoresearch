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
