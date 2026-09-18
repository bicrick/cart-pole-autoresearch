# Paper training lessons → next cart-double-pendulum runs

Audit of `papers/*.pdf` against `train/goals.py` + `train/train.py` (goal-conditioned UU/UD/DU/DD, θ=0 upright).  
Current GCP run (`--num-envs 4096 --updates 400`, HER 0.8) is **not** a keeper: eval reward ~−130 → ~−8000…−11000 after updates ~50–100, `align≈0`. Do not kill that job; restart with the config below after it finishes.

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

**Do not** start a second GPU train while updates ~370/400 of the current job are still running.
