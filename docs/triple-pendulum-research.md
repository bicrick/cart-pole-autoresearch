# Cart-triple-pendulum research notes

Last updated: 2026-09-20 ~04:59 CT.

## Research pass (2026-09-20 ~04:59 CT) — H1: **EVAL** non-MaxEnt avg-reward (2501.09770) + **PPO-BR** clip-contract under visit≠hold

**Sources checked (this fire):** live TB fresher ~04:58 CT (parent steer); overnight/macro @04:55; prior research 04:37 (ERA D=1 pin / 2506 AdaEnt) + 04:02 (MaxEnt ban / C ENT null) + 03:28 (RPO α≈0.01); **EVAL** arXiv:2501.09770 (EigenVector Average-reward Learning + Posterior Policy Iteration — avg-reward *without* entropy); **PPO-BR** arXiv:2505.17714 (dual-signal ε adapt: entropy expand / reward-plateau contract); skimming ATRPO Zhang–Ross arXiv:2106.07329 (on-policy avg-reward, no MaxEnt) as cousin. Confirmed box `scripts/continue-triple-b.sh`: **ENT=0.035** + `VEL_COST_COEF=0.015` + `INIT_NOISE=0.05` + RUN_NAME `…-ent035-…` staged for natural restart. Overnight owns slots — no train start / no mid-kill.

**Phase focus:** P1a walls-on role split. Fresh meters (~04:58 CT):

| Slot | ~u | entropy | nt_at_goal/UUU | nt_align/UUU | other |
|---|---|---|---|---|---|
| **A S1** (LR1e-4) | ~128 | **~1.87** | ~0.042 | ~0.206 | hang_align/UUU **−0.05→−0.03** (was −0.97); hang_at_goal~0.003; policy_loss ~0.005 (was +0.019@u122) — **past u120 clean; entering prior NaN window ~u150–228** |
| **B H1** | 366–370 | **−0.649→−0.665↓↓** | **flat 0.041→0.052** | ~0.15–0.17 | rollout_reward≈**0.50** flat; policy_loss~+0.02; OOB=0; ~30 upd left → ENT035 ~05:07 CT |
| **C H1-var** | ~360 | **~1.22** (healthy) | **~0.049** flat | ~0.15 | ENT=0.05 **null reconfirmed**; policy_loss spike 0.45@u360 (watch one-off) |

### Q1 — NEW paper: EVAL (2501.09770) fills the **non-MaxEnt stay** hole

Prior fail-stack said “avg-reward soft bias / AR-EAPO half without MaxEnt” but cited only AR-EAPO (2409.08938), which **couples** average-reward **with** MaxEnt. EVAL is the missing cookbook:

- Learns entropy-regularized average-reward rate θ + differential value via tilted-eigenvector TD (off-policy DQN-style).
- **Posterior Policy Iteration (PPI / Alg.2):** iteratively replace prior π₀ ← current soft-optimal π; Rawlik theorem → recovers **greedy average-reward optimum as β→∞**, i.e. **avg-reward without entropy regularization**.
- Classic-control suite includes **CartPole continuing balance**: after 5k train steps, EVAL+PPI holds ≥1e5 (claimed ≥1e10) steps while Soft Q-Learning rarely matches — direct *stay upright forever* evidence.
- Acrobot-v1 also in suite (underactuated cousin). Discrete-action / value-based today; continuous actor port is future work in the paper — still steals the **objective**, not the DQN nets.

**Steal for H1 (post-ENT035 / RPO / ERA, visit≠hold branch — never lead):**

1. Prefer **EVAL-PPI spirit** over AR-EAPO: differential / average-reward advantage **with ENT annealed→0** (or no entropy term), not MaxEnt+avg-reward.
2. Cheap on-policy cousins already closer to our PPO stack: **ATRPO** (Zhang–Ross arXiv:2106.07329) / APO trust-region avg-reward — same “continuing stay” objective, no MaxEnt. Port later if short-ep / ENERGY_W stall.
3. Immediate proxies unchanged and still first: Turcato `EPISODE_LEN` 600–800; Spong `ENERGY_W` 0.2→0.35; optional ρ / differential-advantage soft bias once coded.
4. **Ban** shipping EVAL’s *ERAR* (entropy-regularized) half or ASAC-style avg-reward+MaxEnt on H1 — same MaxEnt-vs-hold footgun (2503 / 2506).

### Q2 — NEW recipe: PPO-BR (2505.17714) clip-**contract** when reward flat under σ death

Live B = textbook dual failure: entropy **dead** (H≈−0.65) **and** reward **plateau** (~0.50) with nt flat — visit≠hold. PPO-BR adapts the PPO clip ε from **both** signals:

```text
ε_t = ε₀ · [1 + λ₁·tanh(φ(H_t)) − λ₂·tanh(ψ(ΔR_t))]
ε_t ← clip(ε_t, ε_min, ε_max)
```

Paper defaults: ε₀=**0.2**, λ₁=**0.5**, λ₂=**0.3**, reward window k=**10**. Ablation: entropy drives ~70% of *early* gains; **reward-guided contraction** dominates late stability / variance cut.

**Steal for H1 (orthogonal to RPO μ-perturb + ERA log_std floor):**

1. After σ is restored (ENT035 / RPO / ERA) but **reward still flat + nt flat**: **contract clip** ε 0.2→**0.1** (or apply PPO-BR contraction term only — λ₂>0, do **not** expand on low H). Prevents large ratio steps from locking a “visit once then flop” mean while σ is still recovering.
2. Do **not** use PPO-BR entropy-*expansion* while H is negative/dead on H1 — that fights the catcher (same class as ENT≥0.05 / MaxEnt). Expansion is for S1 swing if ever needed.
3. Caveat: single-author TNNLS-submission claims are aggressive; treat numbers as a **cookbook sketch**, not gospel. Still the only open recipe that jointly says “reward plateau → tighten trust region” for our exact B meters.
4. Tiny patch (clip scalar only) — after RPO α≈0.01 / ERA soft floor in the ladder, before cold wipe.

### Q3 — Locked stack reconfirm (nothing displaces babysit)

| Rank | Lever | Status this fire |
|---|---|---|
| 0 | Babysit B → **ENT=0.035** + VEL_COST=0.015 natural (~05:07) | **Confirmed staged** in continue-b; do not mid-kill |
| 1 | RPO α≈0.01 (ladder 0.01→0.05→0.1; never 0.5) | Unchanged (03:28 / 04:37) |
| 2 | ERA soft log_std floor H₀≈0.5–0.8 softplus/detached — **not** Listing-2 D=1 pin | Unchanged (04:37) |
| 3 | Non-MaxEnt stay: ENERGY_W / short ep / **EVAL-PPI or ATRPO spirit** (not AR-EAPO MaxEnt) | **Sharpened** — EVAL fills citation hole |
| 3b | Optional **PPO-BR ε contract** (λ₂) if reward flat after σ tools | **New** orthogonal clip lever |
| 4 | ENT anneal→0 once nt moves (AdaEnt / 2506) | Unchanged |
| — | AR-EAPO MaxEnt / ENT≥0.05 / hard log_std clamp / Listing-2 D=1 | **Banned** — C null + 2503/2506 |

C ENT=0.05 remains a **natural null** for nt (H healthy ~1.22, nt~0.049≈B). A hang_align still climbing into the old NaN window — overnight watch only.

### Promote?

| Change | Next micro-task? | Backlog? |
|---|---|---|
| EVAL-PPI / ATRPO as concrete **non-MaxEnt** avg-reward stay | No (post-ENT035 branch only) | **Yes** — #2 (viii) sharpen |
| PPO-BR ε contract when reward flat after σ tools | No (after RPO/ERA) | **Yes** — #2 (viii) optional |
| Babysit + ENT035 / mid-kill / ENT≥0.05 / MaxEnt on H1 | No (already Next / banned) | — |

**Nothing displaces** babysit → ENT=0.035. **Material:** first open **avg-reward-without-MaxEnt** citation (EVAL+PPI) for the stay branch + clip-contract recipe matched to live B (H↓ + reward flat). NEED_USER_PING **yes** — new paper + new orthogonal lever on the fail stack (not a Next rewrite).

**No code this fire** (overnight owns train; ENT035 already staged).


## Research pass (2026-09-20 ~04:37 CT) — H1: ERA **1-D Listing-2 pin** footgun + MaxEnt-misleads (2506.05615) + RPO α ladder for *hold*

**Sources checked (this fire):** live TB A/B/C (~04:37 CT); overnight/macro @04:23; prior research 04:02 (MaxEnt ban / C ENT null) + 03:28 (RPO α≈0.01) + 01:58 (ERA H₀≈0.5–0.8); **ERA** arXiv:2510.08549 Listing 2 / Eq.11 / PPO-ERA H₀=−0.3A (App A.1.4); **When MaxEnt Misleads** arXiv:2506.05615 (Entropy Bifurcation Extension + soft-Q vs plain-Q); CleanRL RPO dm_control `cartpole-balance-v0` vs IDP α tables; Actor `train/ppo.py` global `log_std` act_dim=1 clamp −5…2. Overnight owns slots — no train start / no mid-kill.

**Phase focus:** P1a walls-on role split. Fresh meters (~04:37 CT; TB HTTP 200):

| Slot | ~u | entropy | nt_at_goal/UUU | nt_align/UUU | other |
|---|---|---|---|---|---|
| **A S1** (LR1e-4) | 80–90 | **~2.09** | 0.047 | 0.168 | hang_align/UUU **−0.97→−0.052** @u80; hang_at_goal~0.003; policy_loss~0.013 — **past u80 clean, still climbing** |
| **B H1** | 330 | **−0.51↓↓** (σ≈0.15) | **flat ~0.049** | **0.169** | oob=0; visit≠hold; ENT035 still staged (~05:14 CT) |
| **C H1-var** | 320 | **1.31** | **~0.050** | 0.154 | ENT=0.05 null reconfirmed (H healthy, nt≈B) |

### Q1 — ERA on *our* Actor: do **not** paste Listing 2

Paper continuous recipe (Listing 2 / Eq.11) does:

```text
k = −D · (log_std_max + H₀ + log√(2πe))
log_stds = k · softmax(pre_stds) + log_std_max
```

For **D=1** (our cart force), `softmax([pre])≡1`, so `log_std` collapses to the **constant** `−H₀ − log√(2πe)` (then clipped). That is a **hard pin**, not a soft floor — same failure class as rsl_rl hard `log_std.clamp(min=floor)` on a global Parameter (zeros free explore; already warned @02:33).

Also: paper default `H₀=−dim(A)/2` = **−0.5** for D=1 ⇒ σ≈0.15 — *exactly* where live B already sits (H≈−0.51). PPO-ERA main runs use `H₀=−0.3A` (=−0.3) with `ent_coef=0.01` still on. Macro/backlog **H₀≈0.5–0.8** (σ≳0.4–0.54) is the right *behavioral* hold floor — **above** paper defaults — but only if implemented as a **soft** bound.

**Steal for H1 (when coding after ENT035 fails):**

1. Keep learnable global `log_std` (do not replace with Listing-2 softmax for D=1).
2. Soft floor that keeps grads when above the bound, e.g. `log_std_eff = log_σ_min + softplus(log_std − log_σ_min)` with `log_σ_min = log(σ)` for target H₀∈[0.5,0.8], **or** `log_std = torch.maximum(log_std, log_σ_min.detach())`-style detached hinge — never bare `.clamp(min=floor)` on the Parameter alone.
3. Keep `--ent` small (0.01–0.035); ERA's point is reward objective stays clean — do not stack ENT≥0.05 (C already null'd that).
4. Optional δ compensation for tanh bias (Eq.12) is overkill for first try on 1-D walls-hold.

### Q2 — MaxEnt misleads (2506.05615) → reinforce ban + AdaEnt-style *anneal*

Zhang/Chang/Gao formalize why MaxEnt fights upright stay (complements AI Olympics 2503.15290 §III-B already logged @04:02):

- At **critical low-entropy states** (narrow feasible action set — upright balance), soft-Q elevates mediocre high-entropy neighbors that drift into irrecoverable flop; plain-Q prefers the narrow precise action. PPO can learn hold; SAC/MaxEnt soft-Q can converge wrong.
- **Entropy Bifurcation Extension**: MaxEnt-optimal policy can be arbitrarily misaligned with true optimal at targeted states while leaving the rest unchanged — not just "slow explore," but wrong *at convergence*.
- Their SAC-AdaEnt: when soft-Q landscape diverges from plain-Q, drop entropy pressure for that update. **H1 port (cheap):** after RPO/ERA restore σ and `nt` starts moving, **anneal `ENT→0`** (or gate entropy bonus off near upright) rather than keep a permanent MaxEnt/entropy-advantage term on the catcher. Still **ban** full AR-EAPO MaxEnt / entropy-advantage on H1.

Average-reward *without* MaxEnt remains the only AR-EAPO half worth stealing later (short ep / ENERGY_W / optional gain-ρ soft bias) — AR-EAPO itself couples avg-reward **with** MaxEnt (2409.08938); do not ship the pair on H1.

### Q3 — RPO α ladder: hold ≠ IDP

CleanRL: α=**0.5** is fine on `dm_control/cartpole-balance-v0` (PPO~790 → RPO~795) but **catastrophic** on InvertedDoublePendulum (~5644 → ~297). H1 is a **near-upright hold specialist** (closer to balance than IDP swing+balance). Fail stack stays:

1. ENT=0.035 natural (staged) — unchanged.
2. If H↓ + nt flat ~u80 → **RPO α≈0.01** first (safe IDP number; perturb **raw pre-tanh μ** on update only) **or** ERA soft floor above.
3. If α=0.01 is a no-op on hold: ladder **0.01 → 0.05 → 0.1** (more justified for pure-balance H1 than for swing) — **stop well below 0.5**.
4. If visit≠hold persists → Spong `ENERGY_W` 0.2→0.35 / Turcato `EPISODE_LEN` 800→600–800 / avg-reward soft bias **without** MaxEnt → then cold wipe / C-recipe (lr/noise) promote.

### Ranked levers (post-ENT035; unchanged order, sharper recipes)

| Rank | Lever | Concrete | Note |
|---|---|---|---|
| 1 | **RPO α≈0.01** | CleanRL: sample unperturbed; update `μ'=μ+U(−α,α)`; α ladder 0.01→0.05→0.1 if no-op | Tiny patch; IDP-safe first |
| 2 | **ERA soft log_std floor** | H₀≈0.5–0.8 via softplus/detached hinge on global Parameter — **not** Listing-2 softmax | Avoids D=1 hard pin |
| 3 | **Non-MaxEnt stay** | ENERGY_W 0.2→0.35; EPISODE_LEN 600–800; later avg-reward bias only | Visit≠hold branch |
| 4 | **ENT anneal→0** once nt moves | AdaEnt spirit (2506.05615) after explore restored | Do not keep MaxEnt on catcher |
| — | AR-EAPO MaxEnt / ENT≥0.05 / hard log_std clamp | **Banned on H1** | C + 2503/2506 |

### Promote?

| Change | Next micro-task? | Backlog? |
|---|---|---|
| ERA: soft floor H₀≈0.5–0.8; **ban Listing-2 paste on D=1** | Sharpen fail branch (2) wording only | **Yes** — #2 (viii) |
| Cite 2506.05615 + ENT anneal after nt moves | No (post-signal) | **Yes** — #2 (viii) |
| RPO hold ladder 0.01→0.05→0.1 (not 0.5) | Already ≈0.01; add ladder note | **Yes** |
| Mid-kill B / ENT≥0.05 / void / TQC / AR-EAPO MaxEnt on H1 | No | Banned |
| Babysit + ENT035 natural | No (already Next) | — |

**Nothing displaces** babysit → ENT=0.035. **Corrects a dangerous ERA coding footgun** (Listing-2 → hard pin at collapse σ) and adds a second MaxEnt-ban pillar before overnight implements the fail stack. NEED_USER_PING **yes** — material recipe footgun + live B H≈−0.51 @u330.

**No code this fire** (overnight owns train; ENT035 already staged).


## Research pass (2026-09-20 ~04:02 CT) — H1 fail-stack: AR-EAPO **MaxEnt fights hold**; C ENT=0.05 is a null for nt

**Sources checked (this fire):** live TB `20260920-063349_*b/c` + `20260920-084629_*a` (~04:02 CT); overnight/macro @03:55; prior research 03:28 (RPO α≈0.01); **AI Olympics lessons** arXiv:2503.15290 §III-B (AR-EAPO on RealAIGym double pendulum); AR-EAPO arXiv:2409.08938; ERA 2510.08549 reconfirm; CAPS / ASAP already in notes (hold thrash → soft-land first). Overnight owns slots — no train start / no mid-kill.

**Phase focus:** P1a walls-on role split. Fresh meters (~04:02 CT):

| Slot | ~u | entropy | nt_at_goal/UUU | nt_align/UUU | other |
|---|---|---|---|---|---|
| **A S1** (LR1e-4 restart) | 30 | **1.91↑** | 0.036 | 0.084 | hang_align/UUU **−0.97→−0.83→−0.17** @u1/10/20; policy_loss still sane — **strong climb, watch NaN** |
| **B H1** | 270 | **−0.29↓↓** | **flat ~0.054** | **0.176↑** | rollout_reward **~0.50** / hold≈0; classic flop + dead-σ |
| **C H1-var** | 260 | **1.47** (healthy) | **~0.042** | 0.142 | **ENT=0.05 live** — explore OK, **nt still ≈B** |

### Q1 — Dead path clarification: do **not** put AR-EAPO MaxEnt on H1

Competition write-up (2503.15290 §III-B) is explicit about AR-EAPO on underactuated double pendulum:

> *“…the entropy component **prevents the pendulum from remaining stationary at the uppermost position**. Instead, it encourages movement toward lower positions where average entropy is higher.”*

So endless swing-up/down is a **feature** of MaxEnt+avg-reward for robustness contests — and a **bug** for a pure **H1 hold specialist**. Our backlog #2 (viii) “AR-EAPO stay pressure” was underspecified and easy to miscode as “crank MaxEnt / ship full AR-EAPO on B”.

**Steal only the stay half:**
1. Turcato short episode / Spong ENERGY_W 0.2→0.35 / `center_hold_w` / staged `VEL_COST` (already next restart)
2. Optional later **average-reward soft bias** without a MaxEnt entropy objective on the hold net
3. **Do not** add AR-EAPO’s separate entropy-advantage / high τ MaxEnt on H1

AR-EAPO full recipe stays a **swing / single-net robustness** cookbook (slot A / P2), not the H1 fail branch.

### Q2 — C is the natural ENT=0.05 experiment (and it failed the hold gate)

C has run ~2.5h with **ENT=0.05**, entropy **~1.47** (not collapsed), same walls/product/near_target family as B — and `nt/at_goal/UUU` is still **~0.042**, statistically the same flat as B’s **~0.054**. So:

- Unbound / high ENT **≠** UUU hold on this plant+reward
- Macro already said “prefer RPO/ERA/stay stack over ENT≥0.05”; **C meters confirm** — do not promote B→ENT=0.05 after ENT035
- After ENT=0.035 + VEL_COST natural restart, fail order stays: **RPO α≈0.01** or **ERA soft log_std floor** (reward objective stays clean) → **non-MaxEnt stay pressure** (ENERGY_W / short ep / avg-reward bias) → cold wipe / promote C-recipe knobs (lr/noise), **not** C’s ENT

### Q3 — Hold thrash / flop after soft-land

If after VEL_COST + ERA/RPO we still see reward↑ / align mid / nt flat (visit≠hold with chatter), prefer **ASAP λ_T** (already noted) or CAPS temporal smoothness on the **hold** actor — not another ENT bump. CAPS is same family as ASAP; ASAP cookbook already preferred.

### Promote?

| Change | Next micro-task? | Backlog? |
|---|---|---|
| Split AR-EAPO: **ban MaxEnt on H1**; keep avg-reward/ENERGY_W/short-ep stay only | **Yes** — fail branch (2) wording | **Yes** — #2 (viii) |
| Cite C ENT=0.05 as null for nt → no B→0.05 | **Yes** — reinforce | **Yes** |
| Mid-kill B / void / TQC / ENT≥0.05 on H1 | No | Banned |
| Babysit + ENT035 natural | No (already Next) | — |

**Nothing displaces** babysit → ENT=0.035. **Corrects a misreadable lever** before overnight codes “AR-EAPO” onto the catcher. NEED_USER_PING **yes** — MaxEnt-vs-hold footgun + C null experiment.

## Research pass (2026-09-20 ~03:28 CT) — RPO α footgun: pendulum needs **0.01**, not 0.3–0.5

**Sources checked (this fire):** live TB `runs/20260920-063349_*` (~u200–209); overnight/macro @03:21; prior research 02:58 (AR-EAPO) + 02:33 (RPO α≈0.3–0.5); **CleanRL RPO docs** (https://docs.cleanrl.dev/rl-algorithms/rpo/) + arXiv:2212.07536 Alg.1 / §4.2.5 α ablation; ERA 2510.08549 reconfirm (soft floor still #2). Overnight owns slots — no train start / no mid-kill.

**Phase focus:** P1a walls-on UUU via role split. Fresh meters (~u200–209, ~03:28 CT / ~115 min post cold @01:33 CT):

| Slot | ~u | entropy | nt_at_goal/UUU | nt_align/UUU | other |
|---|---|---|---|---|---|
| **A S1** | 208 | **1.47** (stable) | 0.035 | **0.23** | hang path still healthy; leave alone |
| **B H1** | 209 | **−0.05↓↓↓** (past 0; σ≈e^(H−0.5ln2πe) ≈ **0.24**) | **flat ~0.046** | **0.163↑** (was +0.15@u197) | reward↑ / hold≈0; visit≠hold; ENT035 still staged |
| **C H1-var** | 201 | **1.67** | **0.053** (slight↑) | 0.139 | still best H1 contrast; leave alone |

### Q1 — Confirmed dead path: RPO α≈0.3–0.5 on inverted-pendulum plants

Prior Next / backlog #2 (viii) said **RPO α≈0.3–0.5**. CleanRL + paper numbers **kill that band** for our plant class:

| Env | PPO | RPO α=**0.5** | RPO α=**0.01** |
|---|---|---|---|
| **InvertedDoublePendulum-v4** | ~5644 | **~297** (catastrophic) | ~5409 (recovers) |
| InvertedDoublePendulum-v2 | ~5675 | **~275** | ~5661 |
| Ant / Reacher / Pusher | OK-ish / mixed | often worse | recommended |

CleanRL explicit note: *“we recommend using `--rpo-alpha 0.01` for Ant, Hopper, **InvertedDoublePendulum**, Reacher, Pusher.”* Paper §4.2.5: α∈[0.1, 3] often fine in general; **α=0.5 is the default that fails on IDP**. Cart-**triple** inverted is *harder* than double — defaulting to 0.3–0.5 would likely thrash the catcher, not fix entropy.

**Mechanism reminder (Alg.1):** sample with unperturbed \(N(\mu,\sigma)\); on the **update** pass only, \(z\sim U(-\alpha,\alpha)\), \(\mu'=\mu+z\), recompute log-prob under \(N(\mu',\sigma)\). Orthogonal to ERA (mean vs log_std). Our Actor: global `log_std` + `tanh(raw)*forceLimit` — perturb **raw pre-tanh mean** (same units as CleanRL continuous Box / latent), so **α≈0.01** is the right first try, not Newtons.

**Also confirmed this fire:** IsaacGym Cartpole + Gym Pendulum still support *that RPO helps when PPO entropy dies under abundant samples* (we run 8192 envs) — the lever stays; only the **α band** was wrong.

### Q2 — Corrected fail stack after ENT035

1. Babysit → natural ENT=0.035 + VEL_COST (already staged) — unchanged.
2. If H↓ + nt flat ~u80 on that stretch → **RPO α≈0.01** (pendulum-class) **or** ERA soft `log_std` floor (H₀≈0.5–0.8). **Do not** ship α=0.3–0.5.
3. If visit≠hold persists (align↑ / nt flat) → AR-EAPO-style stay pressure (short ep / ENERGY_W / later avg-reward) before cold wipe / C-recipe promote.
4. Optional α ladder only if 0.01 is a no-op: 0.01 → 0.05 → 0.1 (stop well below IDP-fail 0.5).

### Promote?

| Change | Next micro-task? | Backlog? |
|---|---|---|
| Correct RPO α **0.3–0.5 → ≈0.01** (pendulum / IDP CleanRL) | **Yes** — fail branch (2) | **Yes** — #2 (viii) |
| Mid-kill B / jump ENT=0.05 / void / TQC | No | Banned |
| Babysit + ENT035 natural restart | No (already Next) | — |

**Nothing displaces** babysit + ENT035. **Corrects a dangerous wrong number** before overnight codes RPO. NEED_USER_PING **yes** — material α footgun + live B H≈−0.05 @u209.

## Research pass (2026-09-20 ~02:58 CT) — B H→0.20 collapse deepening + AR-EAPO average-reward hold

**Sources checked (this fire):** live TB `runs/20260920-063349_*` (~u150–154); overnight/macro @02:44; prior research 02:33 (RPO α); **AR-EAPO** arXiv:2409.08938 (full ar5iv — average-reward + entropy advantage for acrobot/pendubot swing-up+stabilize); RPO 2212.07536 Alg.1 reconfirm (IsaacGym Cartpole PPO degrades with more data). Skimmed VM `continue-triple-b.sh` — ENT=0.035 + VEL_COST=0.015 + `RUN_NAME` …`-ent035`… still staged. Overnight owns slots — no train start / no mid-kill.

**Phase focus:** P1a walls-on UUU via role split. Fresh meters (~u150, ~02:58 CT / ~85 min post cold):

| Slot | ~u | entropy | nt_at_goal/UUU | nt_align/UUU | other |
|---|---|---|---|---|---|
| **A S1** | 150–154 | 1.45→**1.60** | 0.034 | **0.22** | hang_align **−0.98→−0.030** hang_at_goal~0.004 — healthy swing |
| **B H1** | 154 | 1.45→**0.197↓** | **flat 0.041** | −0.07→+0.091 | rollout_rew **↑0.476** — **collapse past tripwire hard** (σ≈0.30) |
| **C H1-var** | 150 | 1.46→**1.91** (from peak~2.17) | 0.044 | **0.136↑** | visit≠hold (align↑ / nt flat); still best H1 contrast |

### Q1 — Collapse trajectory: 0.42@u110 → 0.32@u130 → **0.20@u154**

Tripwire H≈0.30 fired; rate still ~0.005/u. σ decode (1-D latent Gaussian): H=0.197 ⇒ σ≈**0.30** — behavioral point-mass, not f32 underflow. Pair with rollout_reward↑ / nt hold≈0 = product visit/farm under near_target ICs (same class as dead TQC basin, milder). Mean still inching (align +0.09) while σ dies — exactly the RPO failure mode (Pendulum / Isaac Cartpole: PPO entropy↓ then return stalls or degrades under abundant samples; we run 8192 envs).

**Do not mid-kill.** ENT=0.035 continue-from-ckpt on natural exit remains correct first lever (already staged).

### Q2 — New lever: AR-EAPO average-reward for *stay* after entropy tools

arXiv:2409.08938 (IROS 2024 AI Olympics acrobot/pendubot): formulates swing-up+**stabilize** as a **continuing** MDP with average-reward optimality + separate entropy GAE (EAPO soft bias advantage). Steal for H1 *after* ENT035 / RPO / ERA:

1. **Discount bias → visit≠hold:** episodic discounted product rewards early neighborhood visits; average-reward / soft bias pushes long-horizon upright stay (their §I challenge #2: keep exploring after a suboptimal stable point). Matches B/C: align climbing, nt flat, reward not zero.
2. **MaxEnt as first-class objective** (τ≈2.0, separate entropy GAE λ_e=0.6) — orthogonal to cool-ent β crank and to RPO μ-perturb; pairs with ERA soft floor on our global `log_std`.
3. **Practical cheap proxies (no full AR-EAPO port yet):** (a) Turcato-style shorter `EPISODE_LEN` 600–800 on H1 so product cannot bank late visits without early hold; (b) Spong ENERGY_W / denser stay term on H1 only; (c) optional average-reward / differential advantage later if those stall. Quadratic-only reward in the paper is *not* a reason to rip product — keep product+progress; steal the *horizon / optimality criterion*.
4. **Not for live stretch** — code cost high; rank behind RPO α (tiny patch on update log-prob) and ERA soft floor.

### Q3 — Beats live Next / backlog?

| Candidate | Beats live Next? | Beats / sharpens backlog? |
|---|---|---|
| Babysit; B auto-restart ENT=0.035 | — | Status quo |
| Collapse H=0.20 (past 0.30 hard) | No (already staged ENT035) | Confirms tripwire + urgency |
| If ENT035 fails → RPO / ERA first | No (already Next @02:44) | — |
| If still align↑/nt flat after entropy tools → **AR-EAPO-inspired stay pressure** (short ep / ENERGY_W / later avg-reward) | **Sharpens** fail branch tertiary | **Sharpens #2** (viii) |
| Mid-kill B / ENT=0.05 / void / TQC | No | Banned |

**Nothing displaces** babysit + ENT035 natural restart. **Sharpen** post-entropy fail path: RPO/ERA first; if visit≠hold persists, AR-EAPO-style stay pressure before cold wipe / C-recipe promote.

**Promote?** Macro Ranked backlog **#2** (viii) + Next item (2) tertiary. Do **not** rewrite Live slot lines (overnight owns). NEED_USER_PING **yes** — H→0.20 confirmed + new hold lever.

**No code this fire** (overnight owns train; ENT035 already staged).


## Research pass (2026-09-20 ~02:33 CT) — B H1 entropy collapse confirmed + RPO α-perturb

**Sources checked (this fire):** live TB on VM `runs/20260920-063349_*` (~u110); overnight/macro @02:28; prior research 01:58 (H1 cool→tripwire); **RPO** arXiv:2212.07536 (full ar5iv — Algorithm 1 + ent_coef ablation Fig.6 / §4.2.2); rsl_rl PR #190 / rlevo gaussian docs (hard `log_std` clamp zeros grad on global param); ERA arXiv:2510.08549 (already named). Skimmed box: Actor = **global** `nn.Parameter log_std` clamp −5…2 (`train/ppo.py`); continue-b already stages **ENT=0.035** + `VEL_COST=0.015` on natural restart. Overnight owns slots — no train start / no mid-kill.

**Phase focus:** P1a walls-on UUU via role split. Fresh meters (~u100–110, ~02:33 CT):

| Slot | ~u | entropy | nt_at_goal/UUU | nt_align/UUU | other |
|---|---|---|---|---|---|
| **A S1** | 110 | 1.45→**1.76** | 0.039 | **0.21↑** | hang_align **−0.98→−0.032** — healthy swing |
| **B H1** | 110 | 1.45→**0.42↓** | **flat 0.039** | −0.07→+0.054 | rollout_rew **↑0.47** — **collapse confirmed** |
| **C H1-var** | 104 | 1.46→**2.16↑** | 0.044 | **0.12↑** | best H1 signal (ENT=0.05) |

### Q1 — Upgrade vs 01:58: cool → collapse

01:58 table: H=1.12 = early cool; tripwire H≈**0.30** with nt flat. Live B now **H=0.42** and still dropping ~0.005/u → will be ≤0.3 well before stretch end. Pair with reward↑ / hold≈0 (product visit/farm under near_target ICs, milder than dead TQC basin but same class). **Do not mid-kill** — ENT=0.035 continue already staged.

σ decode (1-D latent Gaussian before tanh·forceLimit): H≈0.42 ⇒ σ≈**0.37** — behavioral collapse, not f32 underflow (rsl_rl −20 floor irrelevant here).

### Q2 — RPO mechanism we had only half-stolen

Prior notes used RPO only for the **ent_coef band** (0.01 helps; ≥0.05 can unbound). Full paper steal:

1. **Algorithm (CleanRL-style):** collect with standard \(a\sim\mathcal{N}(\mu,\sigma)\); on the **PPO update**, set \(\mu'=\mu+z\), \(z\sim\mathcal{U}(-\alpha,\alpha)\), evaluate log-prob under \(\mathcal{N}(\mu',\sigma)\). Default **\(\alpha=0.5\)** on normalized action; ablation sweet spot **0.1–3**.
2. Effect: entropy rises early then **holds a floor** without cranking \(\beta\); PPO alone collapses then plateaus/degrades (Pendulum fail; Isaac Cartpole return drop with more data).
3. Ent-coef ablation: 0.01 often helps; **≥0.05** → unbounded entropy / worse return on many envs — validates **not** jumping H1 live to 0.05 (C already owns that band as the explore control).
4. Orthogonal to ERA: RPO perturbs **mean at update**; ERA constrains **log_std activation**. Both beat “another cool-ent wipe.”

**rsl_rl caveat for our Actor:** hard `log_std.clamp(min=floor)` on a **single global** Parameter **zeros the entropy gradient** and pins σ forever. Prefer ERA soft activation / RPO α / soft `max(log_std, log σ_min)` with detached target — not a hard floor at H₀.

### Q3 — Beats live Next / backlog?

| Candidate | Beats live Next? | Beats / sharpens backlog? |
|---|---|---|
| Babysit to stretch end; B auto-restart ENT=0.035 | — | Status quo (Next @02:28) |
| Collapse diagnosis upgrade (H=0.42 past tripwire) | No (already staged ENT035) | Confirms tripwire fired |
| If ENT035 stretch still H↓ + nt flat ~u80 → **RPO α≈0.3–0.5** (or ERA soft floor) **before** cold wipe / C-recipe promote | **Sharpens** fail branch | **Sharpens #2** entropy leftover |
| Jump B live to ENT=0.05 / mid-kill / void / TQC | No | Banned (RPO ≥0.05 risk; C is the 0.05 control) |

**Nothing displaces** babysit + ENT035 natural restart. **Sharpen** the post-ENT035 fail path: prefer RPO α-perturb or ERA soft `log_std` floor over another ENT crank or immediate cold wipe of B weights (mean may still be learning slowly — align +0.05).

**Promote?** Macro Ranked backlog **#2** (viii) + Next item (2) fail branch — add RPO \(\alpha\) as concrete entropy-fail leftover. Do **not** rewrite Live slot lines / kill list. NEED_USER_PING **yes** — confirmed collapse + new lever (RPO mechanism).

**No code this fire** (overnight owns train; ENT035 already staged).


## Research pass (2026-09-20 ~01:58 CT) — H1 entropy floor under ENT=0.02 + BaRC widen-after-mastery

**Sources checked (this fire):** overnight/macro @01:44–01:46 (S1/H1/H1-var role split LIVE); prior research 01:28; fawraw `docs/m4_findings.md` (CDN reconfirm — catch 0.1@ω≈0, widen catcher **before** soft delivery); ERA arXiv:2510.08549 (+ Listing 2 / PPO H₀≈−0.3A); RPO arXiv:2212.07536 (ent_coef 0.01 helps, ≥0.05 can unbound); arXiv:2606.28627 V_aug ẋ→0 handoff ⊆ RoA; BaRC arXiv:1806.06161; airo7 MPC↔PPO soft-land (−ω² / −ẋ²). Skimmed box: `--init-noise` **wired**; `handoff_eval.py` + `eval-handoff-uuu.sh` **staged**; `VEL_COST_COEF=0.015` on continue-b/c for **next** restart; Actor is **global** `log_std` (`train/ppo.py`, clamp −5…2) not state-dep MLP. Overnight owns slots — no train start / no mid-kill.

**Phase focus:** **P1a walls-on UUU via role split**. Live (~01:46, ~10 min post cold @01:33 CT): A S1 ~u16 hang_align −0.98→−0.80 nt~0.035 ent~1.69↑; B H1 ~u18 nt~0.034 align~−0.03 ent **1.45→1.12** (watch) ENT=0.02; C H1-var ~u16 ent~1.74. Reward↑/hold≈0 = cold product, not hacking. OOB=0. Gate ≪0.80.

### Gap close vs 01:28 leftovers

| Leftover @01:28 | Status @01:58 |
|---|---|
| PPO `--init-noise` missing | **DONE** — LIVE B=0.05 / C=0.08 |
| `handoff_eval.py` absent | **STAGED** (`scripts/handoff_eval.py` + `eval-handoff-uuu.sh`, tol=0.1) |
| B vel-cost unset | **STAGED** `VEL_COST_COEF=0.015` on continue-b/c — applies on **natural restart only**; live stretch undisturbed |
| Role split unimplemented | **LIVE** A=S1 / B=H1 / C=H1-var |

Highest *unimplemented* lever is no longer “split swing vs hold” — it is **H1 widen-after-mastery → soft S1 delivery → X1 smoke**.

### Q1 — B entropy 1.45→1.12 under ENT=0.02: cool, not collapse (yet)

1-D Gaussian decode (our Actor logs `Normal.entropy()` on latent force before tanh·forceLimit):

| Logged H | σ ≈ | Read |
|---|---|---|
| 1.45 → 1.12 | 1.03 → 0.74 | Early cool under tight ENT — healthy |
| 0.30 (Next tripwire) | ~0.33 | Floor with nt still flat → act |
| ≤0 (prior cool-ent collapses) | ≲0.24 | True collapse class — avoid |

**Paper steals for the watch (natural restart only if tripwire hits):**

1. **Mild ENT bump first** (already in Next): 0.02→**0.03–0.04**. RPO: 0.01 helps Pendulum/Bipedal; **≥0.05** risk unbounded entropy / worse return — do **not** jump H1 to 0.05 (C already explores at 0.05).
2. **ERA floor on global `log_std`** (arXiv:2510.08549) if ENT crank fails or distorts product hold: our Actor is a **single** `nn.Parameter` `log_std`, so steal the *idea* not the multi-dim softmax Listing 2 — clamp `log_std ≥ log(σ_min)` so H≥H₀ (e.g. H₀≈0.5–0.8 ⇒ σ≳0.4–0.54) and keep `--ent` small / zero. Decouples explore floor from reward objective (ERA’s whole point vs cool-ent β crank). Prefer over another cool-ent wipe.
3. Do **not** mid-kill for entropy alone while nt cold and H≳1.0.

### Q2 — After H1 shows signal: BaRC widen **before** soft-land A (fawraw order)

fawraw M4 + BaRC: catcher basin is the binding constraint; **widen catcher first**, then soft-deliver swing into it.

| Gate | Action (natural restart / post-stretch) |
|---|---|
| B `eval/near_target/at_goal/UUU` ≳ **0.5** (BaRC \(C_{\mathrm{pass}}\)) on noise=0.05 | Expand H1 `INIT_NOISE` **0.05→0.10** (then 0.15) + optional nonzero ω / off-centre \(x\); keep hang=0 |
| B nt ≳ **0.2** or align climbing clearly | Run `eval-handoff-uuu.sh` smoke (already in Next) |
| A has hold / high-align delivery signal | Soft-land S1: −w_ω / cart-centre / V_aug **ẋ→0** (2606.28627) so handoff ⊆ B RoA — **not** while A is still early hang-align climb |
| Live vel-cost | Already staged 0.015 for next H1 restart (Baek/airo7 soft-land cousin on ω); leave live stretch |

Banned: soft-land A while A nt flat; expand H1 hang mixture; void FT before walls hold; TQC relaunch.

### Q3 — Beats live Next / backlog?

| Candidate | Beats live Next? | Beats / sharpens backlog? |
|---|---|---|
| Babysit A/B/C to ~u50–100 | — | Status quo (Next @01:44) |
| ENT 0.02→0.03–0.04 if B H<0.3 + nt flat past ~u80 | No (already Next) | Confirmed by RPO band |
| ERA `log_std` floor if ENT bump fails | No | **Sharpens #2** entropy leftover |
| BaRC H1 expand 0.05→0.10 after nt≳0.5 | No (premature now) | **Sharpens #2** catcher widen order |
| Soft-land A / void / mid-kill / TQC now | No | Banned |

**Nothing clearly beats** the live Next micro-task. Stay the course: babysit; entropy tripwire; handoff smoke when B signals; BaRC expand + soft-land only after mastery.

**Promote?** Macro **Ranked backlog #2** only — add BaRC widen-after-mastery + ERA `log_std` floor as post-signal / entropy-fail leftovers. **Do not** rewrite Next / Live. NEED_USER_PING no.

## Research pass (2026-09-20 ~01:28 CT) — P1a early walls: Spong visit≠hold branch + catcher RoA leftovers

**Sources checked (this fire):** overnight/macro @01:25 (A/B/C walls LIVE); prior research 00:55 / 00:34; `docs/paper-training-lessons.md` Spong visit≠hold (o/y/z/ab) + Turcato short-horizon + Xin `center_hold_w`; fawraw `docs/m4_findings.md` (CDN — catch basin 0.1@ω=0, soft-land plan); Glück Automatica 2013 (~22 m/s² rail-bound); arXiv:2606.28627 V_aug ẋ→0 handoff ⊆ RoA; Baek EAAI 2024 product+VER (already live flip). Skimmed box: `continue-triple-b.sh` walls-v1; `train_triple.py` has `--vel-cost-coef` but **no** `--init-noise`; **no** `scripts/handoff_eval.py`. Overnight owns slots — no train start.

**Phase focus:** **P1a walls-on UUU**. Live meters (~01:25): A ~u50–56 nt~0.036 **align~+0.123↑** ent~1.94 oob=0; B PPO-balance walls ~u10; C walls combo early. Gate ≪0.80.

### Q1 — A's align↑ / nt flat: Spong visit≠hold, not TQC-style hack

| Signal | Live A | Interpretation | Action now |
|---|---|---|---|
| align_UUU climbing (−0.07→+0.12) while nt~0.036 | Yes | **Visit / approach without stay** (Spong 1995; double lessons o/y) — early walls product is teaching orientation | Leave mid-run (<u100) |
| Reward peak then settle (~229→~155) | Yes | Double walls FT lesson (g): reward dip OK while align holds/climbs | **Do not** mid-kill |
| oob=0, ent~1.94 | Yes | Walls plant healthy; not void-center farm; not cool-ent collapse | No plant flip |
| TQC pattern (rew↑ success≈0, align flat/−) | No | A's align **positive climbing** ≠ reward-hack flopping | Do not treat as TQC |

**Flat-eval ladder sharpen (natural exit only, past ~u150)** — branch on meters, not a single knob:

1. Confirm plant still walls (`track_walls=True`, oob≈0).
2. **If align plateaued AND mean \|x\| parks near trackLimit** while angles flop → **bar 10→50** (fawraw rail-slide killer).
3. **If align still climbing / mid but nt flat** (visit≠hold) → **ENERGY_W 0.2→0.35** first (Spong capture-stay / energy-to-goal pressure); keep hang=0.
4. **Third rung** only if (3) stalls: Turcato short-horizon — `EPISODE_LEN` 1200→**600–800** so dense product cannot score late-episode neighborhood visits without early hold (paper-lessons y). Orthogonal to HER (triple UUU-first has no multi-eq HER yet).
5. Optional after first nt>0.1: tighten IC `INIT_NOISE→0.05` (needs PPO CLI — still gap).
6. Banned mid-P1a: hang curriculum, force60, void, elastic walls, mid-kill, TQC relaunch.

Glück: rail length is the binding constraint (~22 m/s² benches) — walls-on is correct; force probe stays backlog #4.

### Q2 — B PPO-balance walls: RoA widen leftovers (still open from 00:34)

Catcher is correctly **walls-on** now (void mis-arm closed @0979eb7). Gaps that still block a *useful* catch basin once B has signal:

| Leftover | Status | Concrete |
|---|---|---|
| PPO `--init-noise` | **Still missing** in `train_triple.py` — gym default **0.15**; shell `INIT_NOISE=` is a no-op for PPO | Wire CLI; balance start **0.05**, then BaRC/fawraw **expand-with-ω** after nt moves |
| `--vel-cost-coef` | Exists (default 0); `continue-triple-b` does **not** set it | Natural restart: **0.01–0.02** (Thiru2006 / SARS −ẋ² drift-kill) on balance only |
| `handoff_eval.py` | **Still absent** | Port fawraw smoke; `capture_tol=0.1` (not 0.35); pin reset options |
| Soft delivery on A | Not trained yet | fawraw M4 + 2606.28627: −w_ω / cart-centre when \(\bar c>0.9\) / V_aug ẋ→0 so handoff lands inside B's RoA — **after** A has hold signal |
| LQI ∫x | Staged classical only if B basin stays 0.1-only | Q_ξ≈0.1; already backlog |

Do **not** mid-kill B to apply init-noise/vel-cost — stage on natural exit / next stretch only.

### Q3 — Beats live Next / backlog?

| Candidate | Beats live Next? | Beats / sharpens backlog? |
|---|---|---|
| Babysit A+B+C walls through ~u100 | — | Status quo (Next @01:25) |
| Branched flat-eval ladder (rail-park→bar50 vs visit≠hold→ENERGY_W→optional short ep) | No | **Sharpens #2** / Next item (2) ops |
| Wire PPO `--init-noise` + vel-cost on B natural restart | No | **Sharpens #2** cold-start (still open) |
| Soft-land A delivery / handoff_eval | No | **Sharpens #2** handoff prep |
| Treat A as reward-hack / mid-kill / force60 / void now | No | Banned |

**Nothing clearly beats** the live Next micro-task. Stay the course: babysit through ~u100; ladder only on natural exit if A flat past ~u150.

**Promote?** Macro **Ranked backlog #2** only — branched Spong/Turcato flat-eval ladder + reaffirm B `--init-noise`/vel-cost leftovers. **Do not** rewrite Next / Live slot lines. NEED_USER_PING no.


## Research pass (2026-09-20 ~00:55 CT) — P1a walls-on hold + PPO-balance under walls + center-farm checklist

**Sources checked (this fire):** local `docs/paper-training-lessons.md` Track walls / (h–j) hardwalls / (ai–aj) void / **(ap) walls-first**; `docs/triple-macro-loop.md` @00:53; `train/train_triple.py` `--track-walls` / product / barrier; `scripts/continue-triple-{a,b,c}.sh` + `next-train-triple-walls.sh`; fawraw `m4_findings.md` + barrier commit (bar50 / (x/L)^8); Baek EAAI 2024 product+VER; Glück Automatica 2013 rail-constrained ~22 m/s²; double inelastic walls history (`wallRestitution=0`, center_w/hold). **Did not redo** 00:34 handoff_eval / LQI / `--init-noise` pass.

**Phase focus:** **P1a walls-on UUU** (A LIVE cold-start ~05:53Z). Prior void A/C dead on hold (nt_UUU ~0.039 / 0.056). B TQC reward-hack finishing → PPO-balance. Overnight owns slots — no train start.

### Q1 — Walls-on UUU hold (P1a): what papers + double say; knobs if A's first evals stay flat

| Claim | Concrete default / evidence | Live A vs gap |
|---|---|---|
| **Walls-first then void** | Double: hard inelastic cart-only walls → upright → strip walls (lessons Track walls / ap). Glück/Graichen: rail length is the binding constraint — train *with* bounded rail, not void-death. | **Correct plant** now on A (`TRACK_WALLS=1`); P1b nowalls FT only after gate nt≳0.80 / align≳0.90 |
| **Inelastic, not bounce-explore** | `wallRestitution=0`: clamp \(x\), kill \(\dot x\), **no pole impulse**. Elastic 0.3 let UU prop at rail (paper-lessons Walls are cart-only). | Do **not** raise restitution for "exploration" |
| **Product under walls** | Lim/Baek product on world angles; walls remove OOB-death as competing "stay mid" teacher so product can teach upright (user+ap diagnosis). | Live: product + progress1 + flip — keep |
| **Barrier under walls** | Soft mid→edge pressure still useful (rail-park ≠ OOB). fawraw swing probe **bar=50** `(x/L)^8`; live hold recipes use **bar=10**. Double rail-park fix was `center_w=0.06` + `center_hold_w=0.18→0.30` (Xin/Spong stay). | A already gets center defaults via `next-train-triple.sh` (`CENTER_W=0.06`, `CENTER_HOLD_W=0.30`) + `CART_BARRIER_COEF=10` |
| **Force** | Glück ~**22 m/s²** cart accel; our benches ~f40–50. Prior void force40 did not unlock UUU — plant phase was the miss, not force. | Stay **f40**; probe 40→60 only if OOB≈0 **and** plant feels underpowered after walls have evals |
| **near_target / hang** | fawraw M2 / BaRC: hold from near_target, hang=0 until hold moves. | Live hang=0 / near_target — keep |
| **Entropy for hold** | Prior triple A cool-ent collapse (ent→negative) killed explore. Catcher later uses ENT=0.02; early walls hold wants **≥0.05**. | Live ENT=**0.05** / lr1e-4 — leave mid-run |
| **Early reward dip OK** | Double walls FT: reward→0/−150 while align/UU held ~0.6–0.68 through warmup, then recovered (lesson g). | Do **not** mid-kill on first flat reward if nt_align climbing |

**If A's first evals (~u10–80) stay flat** (nt_at_goal/UUU ≲0.05 and align not climbing) — natural-exit ladder only, **never mid-kill**:

1. Confirm plant: `track_walls=True` in run name / log; `train/oob_rate≈0` (walls suppress void OOB).
2. **bar 10→50** (fawraw rail-slide killer) if mean `|x|` parks near trackLimit while angles flop.
3. **ENERGY_W 0.2→0.35** + optional shorter episode only if visit≠hold (Spong) after u150.
4. Tighten IC after first nt signal: `INIT_NOISE→0.05` (needs PPO CLI — still gap from 00:34); keep hang=0.
5. Cool-ent only if entropy collapsed ≤−0.1 with flat nt (prior A pattern) — else leave ENT=0.05.
6. Do **not** flip to hang curriculum / force60 / void mid-P1a.

### Q2 — TQC reward↑/hold=0 → PPO-balance (slot B): confirm + sharpeners beyond 00:34

**Confirm do-not-extend TQC:** ep_rew~567 / success≈0 @~273k is classic product flopping (reward hacking). Basin @150k already **0/9**. Natural exit → PPO-balance; no TQC extend / no wide TQC relaunch.

| Sharpener | Concrete | vs 00:34 |
|---|---|---|
| **Plant for catcher** | During **P1a**, PPO-balance must start with **`TRACK_WALLS=1`** (same plant as A/C swing/hold). Live `continue-triple-b.sh` `start_ppo_balance` calls `next-train-triple.sh` with default **`TRACK_WALLS=0`** → **void catcher while walls-on is the phase**. | **New gap** (not in 00:34) |
| Walls vs void | Catcher plant must match the swing policy it will hand off from. P1a → walls catcher; after P1b strip walls on both. | New |
| init-noise / vel-cost | Still: wire PPO `--init-noise=0.05`; `--vel-cost-coef` **0.01–0.02** on balance only | Already 00:34 — do not redo |
| ENERGY_W / ENT | Armed ENERGY_W=0.2 / ENT=0.02 / hang=0 / near_goal=1 — good. Optional ENERGY_W→0.25 only if balance also flops upright | Minor |
| Barrier on catcher | Keep bar10 under walls; bump 50 only if rail-park | Ops |

### Q3 — Center / void-death farming checklist (walls → P1b void later) — document only

Trigger symptoms after walls-off FT starts (do **not** act on walls plant yet):

1. **Tiny mean `|x|`** + thrashing link angles while `ep_rew` / product climbs.
2. `eval/near_target/at_goal/UUU` flat / ≪ align (visit≠hold / flop).
3. `train/oob_rate` near 0 early then spikes as policy learns "suicide for reset" or stays 0 while never erecting.
4. Demo look: cart glued mid-track, poles windmilling — user "not upright" report.

Mitigations (P1b only): keep `oob_penalty≥20` **after** clip; FT from P1a ckpt (do not cold void); retain `center_w≥0.06` / `center_hold_w≥0.30` + barrier during transfer; gate leave-P1a only at nt≳0.80 / align≳0.90; watch `|x|` histogram + oob_rate every fire.

### Q4 — Beats live Next / backlog?

| Candidate | Beats live Next? | Beats / sharpens backlog? |
|---|---|---|
| Babysit A walls; C→walls; B→PPO-balance; then P1b | — | Status quo (Next @00:53) |
| P1a flat-eval retune ladder (bar50 / ENERGY_W / no mid-kill) | No | **Sharpens** ops for A babysit |
| **TRACK_WALLS=1 on PPO-balance start** (script gap) | No (overnight owns B exit) | **Sharpens #2** cold-start — must-fix before/as B flips |
| Center-farm checklist for P1b | No | Doc-only; watch list |
| Extend TQC / mid-kill A / elastic walls / force60 now | No | Banned |

**Nothing clearly beats** the live Next micro-task. Stay the course.

**Promote?** Macro **Ranked backlog #2** only — add `TRACK_WALLS=1` on PPO-balance during P1a + note flat-eval ladder. **Do not** rewrite Next / Live slot lines. NEED_USER_PING no (overnight already pinged first walls-on).

## Research pass (2026-09-20 ~00:34 CT) — P1 UUU-hold: handoff_eval smoke + LQI widen + PPO-balance cold-start gaps

**Sources checked (this fire):** fawraw raw `scripts/handoff_eval.py` + `docs/m4_findings.md` (CDN); arXiv:2606.28627 full (energy→LQR reachability / Σ⊆Ω_c* / V_aug μẋ); ResearchSquare 2026 TIP LQR (rs-10173980 — RoA ~5°≈0.087 rad @ω=0); CoDIT 2024 ILQR + Machines 2025 ILQR-SMC (∫x augmented Q); Thiru2006 SARS −ẋ² drift-kill; IC_ASET 2025 PI/VI-in-LQR (already). Skimmed box: `train/handoff.py` + `train/lqr_uuu.py` + `scripts/measure_catch_basin.py` staged; **no** `scripts/handoff_eval.py`; `continue-triple-b.sh` PPO-balance path; `train_triple.py` argparse (no `--init-noise`).

**Phase focus:** P1 UUU **hold**. Live Next @00:28 = leave B TQC→~300k → auto **PPO-balance**; then `handoff_eval` smoke + LQI widen if thin. Gate ≪0.80. Overnight owns slots — no train start.

### Q1 — `handoff_eval` smoke checklist (port fawraw; do not copy tol=0.35)

| Metric / knob | Concrete default | Why |
|---|---|---|
| Capture | `capture_tol=0.1`, `capture_vel=1.0` (our measured basin ∩ enter gate) | fawraw script default **0.35** is their transition success tol — **too wide** for our LQR/TQC basins (only 0.1@ω=0 survives) |
| Latch / exit | latch=True + exit 0.25 + dwell≥5 (already in `train/handoff.py`) | Beyond fawraw one-way latch |
| LPF | τ≈0.3 on emitted force | 2606.22145 / staged |
| Energy gate (optional AND) | \|Ẽ\| < ε with Ẽ=E−E_UUU (or \|E\|≲1.08 E_UUU) | 2606.28627 Σ includes \|Ẽ\|<ε; keep angle+ω primary |
| Trials | n≥5 hang→UUU (A swing ckpt) + n≥5 near_target tip-in | Pin start/target via reset options (fawraw M4 bug #2) |
| Report | `handoff@step`, reached, held (≥hold_frac·T), cartx[min,max], stab_time%, max_hold_after_handoff | Exact fawraw print contract |
| Catcher order in smoke | (1) PPO-balance zip/pt if basin>0 → (2) LQR soft → (3) skip dead TQC@150k | Live basin data |

**Repo gap:** still **no** `scripts/handoff_eval.py` — Next item (4) already names it. Port fawraw API onto our `TwoPolicyHandoff` + plant (not MuJoCo SB3).

### Q2 — LQI / multi-link widen (when PPO-balance also thin)

Live LQR soft/stiff: **only** `0.1|0` survives — ResearchSquare independently reports ~**5°** tip recovery @ω≈0 (matches). `lqr_uuu.py` is **8-D only** — no ∫x.

| Lever | Concrete | Source |
|---|---|---|
| **LQI augment** | State z₉=ξ=∫x dt (clip/reset each ep); Q_ξ≈**0.1** (Machines 2025: 0.01·diag with integral weight 10 → 0.1); keep Q_θ=100, R=0.01 | CoDIT 2024 / Machines 2025 / Lim x₉ |
| Multi-link ICs | Basin grid: tip each link **alone** ±{0.1,0.2} @ω=0, then pairwise; don't only equal-offset all three | ResearchSquare phase portrait is single-mode; our plant couples |
| Cart-vel at handoff | Prefer \|ẋ\| small before switch (V_aug μ term on **swing** soft-land, not on LQR) | 2606.28627: W(z) grows with ẋ; unaugmented energy law leaves ẋ≠0 |
| PI/VI-in-LQR | IC_ASET fallback if LQI still 0.1-only | Already #2 classical |

### Q3 — PPO-balance cold-start gaps (armed in `continue-triple-b.sh`)

Live `start_ppo_balance` sets near_target / ENERGY_W=0.2 / ENT=0.02 / no hang — **good**. Two holes before B exits:

1. **`train_triple.py` has no `--init-noise`** — gym default **0.15** always. Catcher should start at **0.05** (fawraw M2 / BaRC tighten-first), then expand with ω+off-centre after nt_at_goal/UUU moves. Overnight must wire CLI (mirror TQC) **before or as** balance starts; env-only `INIT_NOISE=` in the shell is currently a no-op for PPO.
2. **`--vel-cost-coef` exists (default 0) unused** — SARS (Thiru2006) −w₆ẋ² kills drift-and-balance. Starter **0.01–0.02** on the balance recipe only (not on A swing).

Optional later: CLF-RL hold shaping / ENERGY_W↑ near upright (already logged); do not displace balance start.

### Q4 — Beats live Next / backlog?

| Candidate | Beats live Next? | Beats / sharpens backlog? |
|---|---|---|
| Leave B→PPO-balance; then handoff_eval + LQI | — | Status quo (Next) |
| handoff_eval smoke checklist (tol=0.1, pin options, metrics) | No | **Sharpens #2** ops |
| LQI Q_ξ≈0.1 + per-link basin grid | No | **Sharpens #2** widen |
| Wire PPO `--init-noise=0.05` + vel-cost 0.01 on balance | No | **Sharpens #2** cold-start (armed script) |
| 2606.28627 \|Ẽ\| gate / V_aug swing ẋ damp | No | Sharpens #2 enter / soft-land |
| Mid-kill B / force 40→60 / energy E→E_UUU as P1 displace | No | Banned / #4 / #3 |

**Nothing clearly beats** the live Next micro-task. Stay the course: no mid-kill; do not rewrite Next.

**Promote?** Macro backlog **#2** only — concrete handoff_eval checklist + LQI Q_ξ + PPO-balance `--init-noise`/vel-cost ops. **Do not** rewrite Next (overnight @00:28). NEED_USER_PING no.

## Research pass (2026-09-20 ~00:00 CT) — P1 UUU-hold: two-policy handoff staging (catcher order / basin measure / latch+exit / LQR fallback)

**Sources checked (this fire):** fawraw raw `sim/handoff.py` + `docs/m4_findings.md` + `scripts/measure_catch_basin.py` (CDN); DiffSwing NN→LQR @12°; arXiv:2606.28627 reachability energy→LQR (handoff ⊆ RoA); FIP CBA2022 hysteresis (θ_bc=0.35 / θ_sc=1.00); Furuta TECS enter |α|<0.2 rad; IIETA JESA 55(1) GA-LQR TLIP Q/R; IC_ASET 2025 PI/VI-in-LQR (already #2); ResearchSquare 2026 Q_θ~100,R~0.01 (19:40); arXiv:2606.22145 LPF τ≈0.3 / hysteresis (already). Skimmed box `train/` — **no** `handoff.py`, **no** catch-basin script; `export.py` is PPO `.pt→json` only; TQC saves SB3 `.zip`.

**Phase focus:** P1 UUU **hold**. Live Next @23:57 = **stage two-policy handoff** (do not mid-kill TQC B → ~300k). Gate ≪0.80. Overnight owns slots — no train start.

### Q1 — Staging recipe sharpeners (concrete numbers)

| Piece | Concrete default | Why / source | Overnight note |
|---|---|---|---|
| **Catcher choice order** (TQC success **0** @≥150k) | (1) **measure** TQC@150k basin → (2) if basin dead/tiny: **LQR** \(Q_\theta\sim100,R\sim0.01\) (+ optional ∫x LQI / IC_ASET PI) → (3) PPO-balance near_target≤0.1 rad short-horizon → (4) TQC ckpt only if basin widens after expand-with-ω | Live B flat on hold; fawraw: never hand off into unmeasured basin | Prefer classical catcher over a success=0 TQC zip until basin proves ≥0.1@ω≈0 |
| **Catch-basin measure first** | Port fawraw grid: offsets `{0.1,0.2,0.3,0.4,0.5}`, vels `{0,1,2,3}`, success = survive ≥**0.8** of max_steps | `measure_catch_basin.py` + M4 table (1.0 only at 0.1/0; 0 at vel≥2) | **First code artifact** before wiring switch; decide catcher from data |
| **Enter gate** | \(\|\phi_i\|<\mathbf{0.1}\) **and** \(\|\omega\|_\infty<\mathbf{1}\) (+ opt \(\bar c>0.9\) / \(E\lesssim 1.08 E_{UUU}\)) | fawraw basin; 2606.28627: handoff set ⊆ RoA; never tol=**0.3** | Angle-only is insufficient |
| **Latch + hysteresis exit** | `latch=True`; **exit** only if \(\|\phi\|_\infty>\mathbf{0.25}\) (or dwell **N≥5–10** steps before commit) | fawraw latch is **one-way only** (no exit); FIP 0.35/1.00 proves band pattern; our 0.1/0.25 tighter for triple | Implement **exit threshold** on top of fawraw latch — do not copy latch-alone |
| **Soft-landing + LPF** | Swing: −w_ω\|ω\|^2 / Baek c_e=0.09 / cart-centre when \(\bar c>0.9\); force LPF **τ≈0.3** before handoff | fawraw plan (no coefs shipped); 2606.22145 | Soft-land swing delivery; optional DiffSwing blend if bangy |
| **LQR fallback stiffening** | If soft ResearchSquare RoA too small: try GA-LQR-style heavier \(Q_\theta\sim 10^3\) (IIETA Q_θ=2500, Q_x=750, R≈1) **retuned** on our plant — or IC_ASET PI | IIETA JESA; plant-dependent — do not copy K | Escalation only after soft LQR fails basin |

### Q2 — Repo gaps overnight must implement (no train start)

1. **No** `train/handoff.py` / `scripts/handoff_eval.py` / `scripts/measure_catch_basin.py` — port fawraw API (swing+stab `predict`, `capture_tol_rad`, `capture_vel_rad_s`, latch) **plus** hysteresis exit + dwell.
2. TQC catcher load path = SB3 `.zip` (`policies/tqc-triple-uuu.zip` / `*_150000_steps.zip`); `export.py` is PPO-only — do not expect policy.json for TQC.
3. No Riccati / LQR module yet — stage a small `train/lqr_uuu.py` (linearize plant @UUU, solve DARE, act = −Kx) as catcher #2.
4. Leave A/C alone until stretches end; leave B TQC running to ~300k (abort handoff staging → EP specialist only if success>0 mid-stretch).

### Q3 — Beats live Next / backlog?

| Candidate | Beats live Next (stage handoff)? | Beats / sharpens backlog? |
|---|---|---|
| Keep staging two-policy (catcher+swing+gate+latch+LPF) | — | Status quo (Next) |
| Catch-basin measure → catcher order (LQR before dead TQC) | No | **Sharpens #2** |
| Latch + **exit 0.25** + dwell N≥5–10 (beyond fawraw one-way) | No | **Sharpens #2** |
| Soft-land + LPF τ≈0.3 / E-gate / GA-LQR stiff fallback | No | Sharpens #2 |
| gSDE / n_steps / M2 tighten on TQC | No | Stay #1 leftovers post-stretch |
| Force 40→60 / energy E→E_UUU / mid-kill B | No | Stay #4 / #3 / banned |

**Nothing clearly beats** the live Next micro-task. Stay the course: stage handoff in-doc/code while B finishes; do not mid-kill; do not rewrite Next.

**Promote?** Macro backlog **#2** wording only (concrete catcher-order + basin-measure + latch/exit/dwell). **Do not** rewrite Next (overnight wrote it @23:57). NEED_USER_PING no.

## Research pass (2026-09-19 ~23:27 CT) — P1 UUU-hold: gSDE / zoo TQC + BaRC expand-with-ω + trainer CLI gaps

**Sources checked (this fire):** sb3-contrib TQC docs (`use_sde` / `n_steps` params); RL Zoo3 `hyperparams/tqc.yml` (Pendulum-v1, PyBullet Inverted* + BipedalWalker, MountainCarContinuous); Raffin gSDE note on TQC PyBullet results; fawraw `m2_upright_tqc.yaml` + `docs/m4_findings.md` reconfirm (CDN/raw); BaRC arXiv:1806.06161 expand-after-mastery; arXiv:2506.17564 uncertainty-gated residual RL (adjacent to #2); CrossQ/DroQ UTD notes (prior 18:36 anti-pattern stands). Skimmed live `train_triple_tqc.py` + `next-train-triple-tqc-uuu.sh` vs macro Next @23:08.

**Phase focus:** P1 UUU **hold**. Live Next = **meter TQC only** on B (~61.5k, ep_rew climbing, success 0). Gate ≪0.80. Overnight owns slots — no train start.

### Q1 — Fresh hold levers not yet on the TQC ladder

| Lever | Evidence | Live trainer | If early-flat after tighten / n_steps |
|---|---|---|---|
| **gSDE** | Zoo TQC: PyBullet InvertedPendulumSwingup / InvertedDoublePendulum / BipedalWalker / MountainCarContinuous all `use_sde: True` (+ often `log_std_init=-3`); sb3 TQC docs note PyBullet curves used gSDE hypers. Pendulum-v1 zoo entry is bare (no gSDE) — not our plant. Lim Table 1 / fawraw M2: **no** gSDE | `use_sde=False` (default); **no CLI** | **New ladder (g):** after (f) `n_steps=3` still flat → `use_sde=True sde_sample_freq=4` (optional `use_sde_at_warmup=True`). Do **not** lead with gSDE on a climbing first stretch |
| BaRC expand **with ω** | fawraw M4 catch-basin: reliable only ≤0.1 rad **and** ~0 vel; any \|ω\|≳2 → catch 0. Widen catcher = larger `init_noise` + **nonzero link velocities** + off-centre cart — *after* near_target mastery, not when early-flat | Live stretch: noise 0.15 / hang 0.05 / wide 0.25 (harder than M2). Ladder (a) still **tighten** first if flat | Sharpen expand-after-mastery: angle noise alone is insufficient; add ω noise + off-centre x before claiming catcher ready for handoff |
| Zoo `train_freq=8` / `gradient_steps=8` | PyBullet TQC defaults keep UTD≈1, just batched | Lim/M2 = 1/1 | Optional wall-clock tweak only; **not** DroQ UTD=20. Skip unless coding convenience |
| VecNormalize | Zoo Pendulum/PyBullet TQC: usually off; HER Fetch uses normalize | None | Low priority vs ∫x / n_steps / gSDE |
| Uncertainty-gated residual (2506.17564) | Focus residual explore where base is uncertain; critic on combined action | N/A (no residual yet) | Form of backlog **#2** if LQR/TQC catcher + residual; do not interrupt B |
| CrossQ / high-UTD DroQ | Prior 18:36: CrossQ poor on sparse pendulum-swingup | — | Still **banned** for sparse UUU hold |

### Q2 — Trainer ops gaps (when overnight codes the next ladder patch)

Live `train_triple_tqc.py` still missing vs staged ladder:
1. No `--n-steps` (SB3 TQC supports it; unset → 1)
2. No `--use-sde` / `--sde-sample-freq`
3. `EvalCallback` shares train init mix — still **no** `eval/near_target/at_goal/UUU` (P1 gate invisible on TQC TB)
4. No ∫x obs / VER flip / `ry_scale` CLI (already on ladder a–e)

Wire (1)+(3) first when patching; (2) with ladder (g).

### Q3 — Beats live Next / backlog?

| Candidate | Beats live Next (meter TQC)? | Beats / sharpens backlog? |
|---|---|---|
| Keep metering to ~150k | — | Status quo |
| M2 tighten / ry / ∫x / VER / M2 arch / n_steps | No | Already #1 (a–f) |
| **gSDE after n_steps** | No | **Sharpens #1** → new **(g)** |
| BaRC expand-with-ω after mastery | No | Sharpens #1 expand + #2 catcher basin |
| Uncertainty residual / Zoo 8/8 UTD-batch | No | #2 form / skip |
| Force 40→60 / energy E→E_UUU / CrossQ | No | Stay #4 / #3 / banned |

**Nothing clearly beats** the live Next micro-task. Stay the course: meter B TQC; no mid-kill; no entropy/PPO knobs; two-policy only if ~150k flat after hold ladder (now through gSDE).

**Promote?** Macro backlog **#1** only — append ladder **(g) gSDE** + note expand-with-ω. **Do not** rewrite Next (overnight running). NEED_USER_PING no.

## Research pass (2026-09-19 ~23:01 CT) — P1 UUU-hold: live TQC gap audit + n-step / eval-meter / LQR-PI

**Sources checked (this fire):** fawraw `m2_upright_tqc.yaml` (raw reconfirm); sb3-contrib TQC `n_steps` / NStepReplayBuffer docs; Kuznetsov TQC truncation notes; IC_ASET 2025 PI/VI+LQR triple (IEEE); Cambridge Robotica 2026 CSAC-QI (∫θ reward, already logged); IJMLC 2025 LQR+SAC residual; arXiv:2606.22145 curriculum DR (timeout on full HTML — abstract/prior notes only); Baek VER / Lim Table 1 already in 21:56–22:34. Skimmed live `train_triple_tqc.py` + `next-train-triple-tqc-uuu.sh` vs macro Next.

**Phase focus:** P1 UUU **hold**. Live Next = **meter TQC only** on B (leave A/C). Gate ≪0.80. Overnight owns slots — no train start.

### Q1 — What is the running TQC missing vs Lim / fawraw M2?

Code audit of `train/train_triple_tqc.py` (live B path) against Lim Table 1 + fawraw M2 yaml:

| Lever | Lim / fawraw M2 | Live TQC trainer | If early-flat (~50–150k) |
|---|---|---|---|
| Hypers | Lim: buffer 1e6, π 400→300, qf 3×512, N=25 drop2; M2: buffer **200k**, net **[128,128]**, N=**20**, **150k** steps | Matches **Lim** sized; 300k steps; noise **0.15** hang **0.05** wide **0.25** | Keep Lim arch first stretch. Ladder already: **(a)** M2 tighten noise=0.05 hang=0 wide=0 → **(e)** M2 arch/buffer/150k |
| `train_freq` / `gradient_steps` | 1 / 1 | SB3 defaults (=1/1) | No change |
| **`n_steps`** | Not in Lim/M2 yaml; Raffin/FastTD3 notes (this log ~15:xx) favor **n_steps=3** on hard continuous control | **Unset (=1)** — no CLI | **New ladder step:** after M2 tighten still flat, try `n_steps=3` before declaring TQC dead / before M2 arch shrink |
| VER / flip | Baek off-policy native; Lim none | **No** replay flip | Stays **(d)** |
| Obs ∫x / reward ∫θ | Lim x₉=∫y; CSAC-QI ∫θ in **reward** | Neither | ∫x obs = **(c)**; CSAC-QI ∫θ reward = optional **after** nt moves (orthogonal to cart integral) |
| `ry_scale` Lim cart | meter Lim | product uses track_limit soft scale; no TQC CLI | Stays **(b)** |
| **P1 eval meter** | Curriculum near_target | `EvalCallback` on **same** train init mix — **no** `eval/near_target/at_goal/UUU` | **Ops gap:** overnight cannot read P1 gate from TQC TB the way PPO does. Stage curriculum eval env for TQC when coding next ladder patch |

### Q2 — Hold catcher extras (backlog #2 only; do not interrupt B)

1. **IC_ASET 2025** (Policy Iteration / Value Iteration inside LQR on linearized cart-triple) — PI damps faster, VI smoother effort. Cheap classical **UUU catcher** alternative to ResearchSquare LQR Q/R if we stage two-policy after TQC~150k flat.
2. IJMLC 2025 residual SAC-on-LQR (friction) — still sim2real-tilted; keep as residual form of #2, not a P1 displace.
3. CSAC-QI ∫θ reward — hold polish after nt rises; not a reason to mid-kill TQC.

### Q3 — Beats live Next / backlog?

| Candidate | Beats live Next (meter TQC)? | Beats / sharpens backlog? |
|---|---|---|
| Keep metering to ~150k | — | Status quo |
| M2 tighten / ry_scale / ∫x / VER / M2 arch | No | Already #1 (a–e) |
| **`n_steps=3`** after tighten | No | **Sharpens #1** (new (f)) |
| Curriculum `eval/near_target/*` on TQC | No | **Sharpens #1 ops** (gate visibility) |
| IC_ASET PI/VI+LQR catcher | No | Sharpens #2 classical hold half |
| CSAC-QI ∫θ / residual SAC | No | Optional after nt moves / #2 form |
| Force 40→60 / energy E→E_UUU | No | Stay #4 / #3 |

**Nothing clearly beats** the live Next micro-task. Stay the course: meter B TQC; no mid-kill; no entropy/PPO knobs; two-policy only if ~150k flat after hold ladder.

**Promote?** Macro backlog **#1** only — append ladder **(f) `n_steps=3`** + note curriculum eval-meter ops. **Do not** rewrite Next (overnight running). NEED_USER_PING no.

## Research pass (2026-09-19 ~22:34 CT) — P1 UUU-hold: Lim PDF deep-read + fawraw M2 yaml + ∫x / VER ladder

**Sources checked (this fire):** Lim/Ju/Lee KIEE 2025 PDF full (Table 1 + §2.2 + §3.1 x₉=∫y + §4.1–4.3 reward/ICs); fawraw live `training/configs/m2_upright_tqc.yaml` (raw); Kuznetsov TQC defaults / sb3_contrib TQC API; Cambridge Robotica 2026 CSAC-QI (already logged); ILQR CoDIT 2024 + Machines 2025 integral-cart LQR; IJMLC 2025 LQR+SAC friction residual; arXiv:2506.17564 residual RL (adjacent). Prior 21:56 TQC-mirror + two-policy recipe stands — no overnight-status duplicate.

**Phase focus:** P1 UUU **hold**. Live Next = do not touch B/C; wait B PPO exit → Lim TQC UUU. Gate ≪0.80.

### Q1 — What does Lim actually do that our staged TQC still lacks?

Re-read Table 1 + §3.1/§4.2 against `train_triple_tqc.py` + `product_reward` + `observe`:

| Lever | Lim (hardware) | Ours staged | Action if TQC early-flat |
|---|---|---|---|
| Hypers | lr 3e-4, γ 0.99, τ 0.005, buffer **1e6**, batch 256, N=**3**, M=**25**, drop 2, π 400→300, qf 3×512, **1 env-step / 1 grad-step** | Matches Table 1 | **Do not retune arches mid-run** |
| ICs | **Wide** eq (10): y±0.3, ẏ±1.2, θ±π, ω±10/20/30 — specialists from random | near_target noise=**0.15** + wide_frac 0.25 + hang 0.05 | First: **M2 tighten** noise=0.05, hang=0, wide=0 (21:56). Only expand toward Lim-wide **after** near_target mastery (BaRC) |
| Reward R_ω | Relative-joint summed world rates: \|θ̇₁\|, \|θ̇₁+θ̇₂\|, \|θ̇₁+θ̇₂+θ̇₃\| | Absolute plant → per-link world ω — **equivalent** | No change |
| Reward R_y | exp(−0.3·\|y\|) meters; early-stop \|y\|>**0.48** | exp(−0.3·\|x\|/track_limit) → **softer** cart center | Optional hold patch: `ry_scale=1.0` (meter Lim) once TQC lives |
| **Obs ∫y** | State x₉ = ∫₀ᵗ y(τ)dτ to kill cart steady-state offset (§3.1) | `observe()` is 11-D: x,ẋ,sin/cos×3,ω×3 — **no integral** | **Stage ∫x obs** (clip/reset each ep) for TQC hold / LQR catcher — classical LQI + Lim |
| VER / flip | **None** (DR via wide ICs only) | TQC trainer has **no** Baek VER replay doubling | Off-policy VER is native here (unlike PPO port); add buffer flip if M2 tighten still flat |
| Eval meter | ep return ~700–800/1000 under wide ICs | SB3 EvalCallback on **same** init mix; no `eval/near_target/*` | Wire curriculum near_target eval into TQC trainer so P1 gate is visible |

**fawraw M2 yaml (confirmed):** `near_target` noise=**0.05**, 150k steps, buffer **200k**, net **[128,128]**, n_quantiles **20**, n_critics 3, train_freq=1, gradient_steps=1. Smaller/faster than Lim Table 1 — valid **fallback arch** if Lim-sized net is sample-hungry on near_target after tighten.

### Q2 — Hold catcher extras for backlog #2 (do not interrupt B→TQC)

1. **∫x-augmented LQR / ILQR** (CoDIT 2024 / Machines 2025 / Lim x₉) — same switch ≤0.1 rad + ω∞<1; integral kills cart bias the pure LQR miss.
2. **LQR + residual SAC** (IJMLC 2025 friction compensation) — residual over nominal LQR; more sim2real than sim-hold, but a clean form of residual upright if pure TQC/LQR wobbles.
3. CSAC-QI ∫θ still optional after nt moves (21:31) — orthogonal to ∫x (angles vs cart).

### Q3 — Beats live Next / backlog?

| Candidate | Beats live Next (wait B→TQC)? | Beats / sharpens backlog? |
|---|---|---|
| M2 tighten / BaRC expand | No | Sharpens #1 (already 21:56) |
| `ry_scale=1.0` Lim cart term | No | Sharpens #1 reward mirror |
| **∫x in obs** + ILQR catcher | No | **Sharpens #1 and #2** |
| Baek VER on TQC replay | No | Sharpens #1 (TQC-native; Lim didn't need it) |
| M2 arch [128,128] / buffer 200k / 150k | No | Fallback under #1 if Lim arch stalls |
| Curriculum eval meters on TQC | No | Ops for P1 gate reading |
| Residual SAC-on-LQR | No | Form of #2 |
| Force 40→60 / energy E→E_UUU | No | Stay #4 / #3 |

**Nothing clearly beats** the live Next micro-task. Stay the course: B→TQC → mirror ladder below → two-policy only if ~150k flat after tighten.

**Promote?** Macro backlog **#1** only (TQC hold mirror ladder). **Do not** rewrite Next / interrupt B. NEED_USER_PING no.

## Research pass (2026-09-19 ~21:56 CT) — P1 UUU-hold feeder: TQC mirror + staged two-policy recipe

**Sources checked (this fire):** Lim/Ju/Lee KIEE 2025 PDF (Table 1 + §4.2–4.3 product + ICs); fawraw `m2_upright_tqc.yaml` + live `sim/handoff.py` + `docs/m4_findings.md` (CDN/raw); Baek EAAI 2024 lineage (already in product/VER); Spong/Xin energy→LQR pattern; DiffSwing (energy NN→LQR @12°); Oh et al. QIP IJCAS 2025 (TQC+VER scale); Glück Automatica 2013 (accel budget only); ResearchSquare 2026 cart-triple LQR Q/R; arXiv:2606.22145 / 2312.11311 / prior 15:42–21:31 notes. Skimmed top of this log (21:30 status + 21:31 BaRC) — no duplicate overnight-status section.

**Phase focus:** P1 UUU **hold** only (near_target meter). Live Next = leave B PPO → Lim TQC UUU (~ETA 22:40 CT). Diagnosis already locked: single-policy PPO dead (nt~0.04–0.06).

### Q1 — Once Lim TQC is live, what should overnight mirror?

Staged `next-train-triple-tqc-uuu.sh` already matches Lim Table 1 (lr **3e-4**, γ **0.99**, τ **0.005**, buffer **1e6**, batch **256**, policy **400→300**, critic **3×512**, n_quantiles **25**, n_critics **3**, top_drop **2**). Product + progress + barrier already wired. **Do not retune arches mid-run.**

| Lever | Lim / fawraw M2 | Our staged default | Overnight action |
|---|---|---|---|
| Init | Lim: wide random; **fawraw M2 hold:** `near_target` **noise=0.05**, 150k | `near_target` **noise=0.15**, wide_frac **0.25**, hang_frac **0.05**, 300k | Keep first stretch as staged. If **flat after ~50–100k** (nt_at_goal still ≲0.1): **tighten** to M2-like `INIT_NOISE=0.05 HANG_FRAC=0 WIDE_FRAC=0` before declaring TQC dead (BaRC ladder 21:31: expand only after mastery). |
| Reward | Lim pure product (eq 8–9); fawraw additive+progress | Product + `progress_w=1` + barrier10 | Keep. Optional later: CSAC-QI ∫θ on hold-only after nt moves (21:31 #83). |
| Done signal | Lim ep return plateaus **~700–800 / 1000** under wide ICs | sb3 `ep_rew_mean` + our `eval/near_target/*` | Judge hold by **`eval/near_target/at_goal/UUU`** (gate ≳0.80) + align ≳0.90; treat ~700–800 return as healthy specialist, not failure. |
| Force / early-stop | Lim \|a\|>**2.5** m/s², \|y\|>0.48 m (their plant) | forceLimit **40**, soft barrier | Do **not** copy Lim 2.5 m/s² — different units/plant. Keep f40; force→60 only if OOB≈0 **and** rail-slide (backlog #4). |
| Eval | Lim: wide IC robustness | Curriculum near_target + hang | Keep curriculum meters; harsh random eval stays secondary for P1. |

**QIP / Baek:** reinforces off-policy TQC(+VER/flip) for multi-link hold — already why we left PPO. No new numeric hypers beyond Lim Table 1.

### Q2 — Concrete two-policy handoff if TQC plateaus (~150k flat)

Ready-to-stage recipe (consolidates fawraw handoff + DiffSwing + ResearchSquare LQR + 15:42/19:40 notes). **Do not interrupt B→TQC.**

1. **Hold half (catcher):** prefer the live TQC UUU ckpt if nt rising; else **cart-triple LQR** about UUU with starter \(Q_\theta\sim 100\), \(R\sim 0.01\) (ResearchSquare 2026), ICs inside **≤0.1–0.26 rad**; or PPO-balance / residual around upright with **short horizon** + dense near_target only.
2. **Swing half:** leave slot A (or energy / DiffSwing-style \(w_E(E-E_{UUU})^2\)) — separate net; never one Gaussian for both.
3. **Switch (enter):** all world angles \(\|\phi_i\| < \mathbf{0.1}\,\mathrm{rad}\) **and** \(\|\omega_i\|_\infty < \mathbf{1.0}\,\mathrm{rad/s}\) (tighten ω from measured basin; fawraw table: any vel≥2 → 0 catch). Optional AND \(\bar c>0.9\) (Mon) and/or energy near \(E_{UUU}\). **Never** fawraw code default `capture_tol_rad=0.3`.
4. **Latch + hysteresis:** `latch=True`; exit only if \(\|\phi\|_\infty > \mathbf{0.25}\,\mathrm{rad}\) (or dwell) so near-misses recover (arXiv:2606.22145).
5. **Soft-landing on swing near target:** Baek \(c_e=0.09\) rate floors / Lim \(R_\omega\) / light \(-w_\omega\|\omega\|^2\) + cart centre — gated by \(\bar c>0.9\) (19:40 pack).
6. **Action smoothing:** first-order LPF **τ≈0.3** on swing force before handoff (2606.22145); ASAP only if still bangy.
7. **Obs window:** full state (pos/vel/angles/rates); DiffSwing trig encoding optional for energy net only.

### Q3 — Hold-specific levers vs backlog #1–3?

| Candidate | Beats live Next (B→TQC)? | Beats / sharpens backlog? |
|---|---|---|
| Pure M2 tighten (`noise=0.05`, no hang) if TQC early-flat | **No** — same #1 stretch, curriculum detail | Sharpens #1 implementation |
| BaRC ladder expand after mastery | No | Sharpens #1 (already 21:31) |
| Two-policy recipe above | No — only after TQC ~150k flat | **Sharpens #2** (was vague) |
| Energy E→E_UUU | No | Stays #3; hold-phase energy is soft-landing / E-gate, not swing pump |
| Obs noise / residual upright / shorter hold horizon | No new winner | Optional hold-net knobs after TQC lives; residual LQR+RL is a *form* of #2 |
| Force 40→60 | No | Stays #4 gated |

**Nothing clearly beats** the live Next micro-task (Lim TQC already staged on B). Recommendation: **stay the course** — let B exit → TQC; mirror table above; if TQC flat ~150k → stage two-policy with the concrete switch/LQR pack (do not invent a new #1).

**Promote?** Macro backlog item **#2 only** (concrete recipe). **Do not** change Next micro-task / interrupt B→TQC. NEED_USER_PING no.

## Overnight fire — status (2026-09-19 ~21:30 CT)

**VM:** `cartpole-train-od` RUNNING us-east1-b L4 ~99%/16.5GB; TB http://34.148.138.48:6006/ up; **no double/xonly**. Uptime ~42.5h ≈ **~$30–36** (budget uncapped — stay up). continue-a/b/c armed. GPU healthy. No NaNs.

**Jobs (alive — do not mid-kill):**
| Slot | Recipe | ~u | entropy | nt at_goal/UUU | nt align/UUU | hang at_goal/UUU | hang align/UUU |
|---|---|---|---|---|---|---|---|
| A | cool-ent v6 swing f50 e0.05 lr1e-4 | ~120 | ~0.92 healthy | ~0.045 flat | ~0.18 | ~0.009 | ~−0.020 |
| B | entboost v7b hold f40 e0.08 lr5e-5 | ~283 | **~3.42 saturated** since ~u140 | ~0.050 flat | ~0.22 | ~0.003 | ~−0.11 |
| C | entboost v7 combo f40 e0.08 lr5e-5 | ~105 | ~0.38 cooling | ~0.053 flat | ~0.21 | ~0.010 | ~−0.083 |

**Diagnosis:** P1 gate ≪0.80. Single-policy PPO is **dead for UUU hold** — cool-ent v6 (full prior stretch + live A), entboost v7b (B saturated), entboost v7 (C) all stuck nt at_goal/UUU ≈ 0.04–0.06. Matches user call: stop entropy/LR roulette.

**Action:** Staged **Lim TQC UUU** on slot B for natural v7b exit (rewrote `continue-triple-b.sh`; marker `.triple-b-tqc-uuu-v1`). A/C finish current PPO stretches undisturbed. Next after TQC flat: two-policy handoff. NEED_USER_PING **yes** (material strategy change: leaving PPO for TQC).

## Overnight fire — status (2026-09-19 ~21:10 CT)

**VM:** `cartpole-train-od` RUNNING us-east1-b L4 ~98%/16.1GB; TB http://34.148.138.48:6006/ up; **no double/xonly**. Uptime ~42.2h ≈ **~$29–35** (budget uncapped — stay up). continue-a/b/c armed; scripts md5-match box. GPU healthy. No NaNs.

**Jobs (alive — do not mid-kill):**
| Slot | Recipe | ~u | entropy | nt at_goal/UUU | nt align/UUU | hang at_goal/UUU | hang align/UUU |
|---|---|---|---|---|---|---|---|
| A | cool-ent v6 swing f50 e0.05 lr1e-4 | ~80 | ~0.95 healthy | ~0.048 flat | ~0.18 | ~0.012 | ~−0.018 |
| B | entboost v7b hold f40 e0.08 lr5e-5 | ~249 | **~3.42 saturated** since ~u80 | ~0.040 flat | ~0.21↑ | ~0.002 | ~−0.15↑ |
| C | entboost v7 combo f40 e0.08 lr5e-5 | ~70 | ~0.43 cooling | ~0.060↑ | ~0.22 | (early) | (early) |

**Diagnosis:** P1 gate ≪0.80. **B** entropy maxed / explore useless; nt UUU hold still ~0.04 — **v8** (ENT0.05 cold) already staged for u400 (~22:05 CT). **A/C** early in stretch, oob=0; A nt flat, C nt slightly rising. Not a plant/force issue.

**Action:** None mid-run. Watch B→v8 cold; A/C to u400. Next lever if still flat: **two-policy handoff**. NEED_USER_PING no.

## Implementation status (2026-09-19 ~14:35 CT)

**Built on `main`:** cart-triple plant + Lim/fawraw product + **P0/P1 progress + flip-augment + curriculum eval**.

| Piece | Path | Notes |
|---|---|---|
| Physics | `train/physics_triple.py` | Batched torch, 4×4 mass solve, θ=0 upright, no track walls |
| Constants | `shared/constants-triple.json` | 3 equal links; `obsDim=25`; **`forceLimit=40.0`** |
| Goals | `train/goals_triple.py` | 8 eqs; product reward + **`progress_w` Δ cos-align**; Baek α floors; UP×5/DOWN×1; **`energy_w` = height/align proxy (not E→E_UUU)** |
| Train | `train/train_triple.py` | `--progress-w` (default 1.0 product); `--flip-augment` (Baek VER, default on); `--eval-curriculum` → `eval/near_target/*` + `eval/hang/*` |
| Launch | `scripts/next-train-triple.sh` | Passes `PROGRESS_W` / `FLIP_AUGMENT` |
| Slots | `scripts/continue-triple-{a,b,c}.sh` | **A** swing f50 ent0.05 lr1e-4; **B** hold f40 ent0.03 lr1e-4; **C** combo f40 ent0.04 lr1e-4 (cool-ent **v6** staged). Live stretch still v5 until u400. |

**Symmetry (Baek VER / `--flip-augment`):** planar reflect across the vertical midline maps `(x,ẋ,θᵢ,θ̇ᵢ,F)→(−x,−ẋ,−θᵢ,−θ̇ᵢ,−F)`. Dynamics + product reward are equivariant; θ*∈{0,π} goal encodings invariant. After GAE, PPO batch is duplicated with flipped obs/raw and recomputed logπ.

**Eval meters:** keep harsh `eval/*` (random ICs). Also log `eval/near_target/at_goal/UUU`, `eval/near_target/align/UUU`, `eval/hang/*` — diagnose hold vs swing with the right meter.

**Slot policy:** **Kill double/xonly.** All 3 L4 slots on `cartpole-train-od` are triple-a/b/c only.

**Cold start:** cool-ent v6 → wipe once via `.triple-a-cool-ent-v6`, `.triple-b-cool-ent-v6`, `.triple-c-cool-ent-v6` (after v5 finishes).

**Launch:**
```bash
NUM_ENVS=8192 FORCE_LIMIT=40 PROGRESS_W=1.0 FLIP_AUGMENT=1 bash scripts/next-train-triple.sh
# Specialists: bash scripts/next-train-triple-{a,b,c}.sh
# A/B/C on VM: continue-triple-{a,b,c}.sh
```


## Overnight fire — status (2026-09-19 ~16:02 CT)

**VM:** `cartpole-train-od` RUNNING us-east1-b L4 ~99%/15.7GB; TB http://34.148.138.48:6006/ up; **no double/xonly**. Uptime ~37.1h ≈ **~$26** @~$0.70/hr (≤$30; ~2h headroom to u400 ≈$27.5). continue-triple-a/b/c armed with **cool-ent v6** (live still v5). GPU healthy. No NaNs.

**Jobs (alive, progress+flip v5 — do not mid-run kill):**
| Slot | Recipe | ~u | rollout_r | entropy | goal_frac/UUU | near_target at_goal/UUU | near_target align/UUU | hang align/UUU |
|---|---|---|---|---|---|---|---|---|
| A | swing hang+near f50 bar10 e0.5 prog1+flip | ~160 | ~0.52 | **~−0.54** ↓↓ | ~0.93 | ~0.036 flat | ~+0.173 | ~−0.076↑ |
| B | hold near_target f40 bar10 e0.15 prog1+flip | ~157 | ~0.54 | **~−0.33** ↓ | ~0.92 | ~0.046 flat | ~+0.127↑ | ~−0.29↑ |
| C | combo hang0.3 f40 bar10 e0.35 prog1+flip | ~150 | ~0.52 | **~−0.26** ↓ | ~0.93 | ~0.040 flat | ~+0.154↑ | ~−0.20 |

**Diagnosis:** Entropy collapse deepened since 15:50 (A −0.34→**−0.54**; B/C more negative). near_target at_goal/UUU still ~0.04 (no hold). B/C near_target UUU *align* still climbing slowly; A hang UUU align inching toward 0. oob≈0. Force 40–50 + barrier10 still OK vs Glück/Lim — **not a plant/force issue**; explore/β under fixed `--ent=0.01`.

**Action:** None mid-run. cool-ent **v6** already staged (ENT 0.05/0.03/0.04, LR 1e-4, cold markers on next start). u400 ETA ~18:15 CT (~$27.5). **Budget watch:** overnight v6 past ~21:00 CT pushes past ~$30 — next fire should gate continue vs stop if cap binds.

**Watch next:** v5 finish → auto cold v6; entropy floor; near_target at_goal/UUU. Stage-gate still UUU hold ≳0.80 before multi-eq. NEED_USER_PING no (same story as 15:50; path already staged).

## Overnight fire — status (2026-09-19 ~15:50 CT)

**VM:** `cartpole-train-od` RUNNING us-east1-b L4 ~99%/15.7GB; TB http://34.148.138.48:6006/ up; **no double/xonly**. Uptime ~36.9h ≈ **~$26–28** @~$0.70–0.76/hr (≤$30; ~2–3h headroom to u400). continue-triple-a/b/c rearmed with **cool-ent v6** (live trains still v5). GPU healthy.

**Jobs (alive, progress+flip v5 — do not mid-run kill):**
| Slot | Recipe | ~u | rollout_r | entropy | goal_frac/UUU | near_target at_goal/UUU | near_target align/UUU | hang align/UUU |
|---|---|---|---|---|---|---|---|---|
| A | swing hang+near f50 bar10 e0.5 prog1+flip | ~133 | ~0.49 | **~−0.34** ↓↓ | ~0.94 | ~0.042 flat | ~+0.168 | ~−0.087 |
| B | hold near_target f40 bar10 e0.15 prog1+flip | ~131 | ~0.52 | **~−0.17** ↓ | ~0.93 | ~0.042 flat | ~+0.099 | ~−0.30↑ |
| C | combo hang0.3 f40 bar10 e0.35 prog1+flip | ~130 | ~0.51 | **~−0.14** ↓ | ~0.93 | ~0.041 flat | ~+0.144 | ~−0.21 |

**Diagnosis:** Gate from 15:32 fired — **all three slots now entropy-negative** (A worst; B/C crossed after u100). near_target at_goal/UUU still ~0.04 (no hold progress). oob=0; force 40–50 + barrier10 still OK vs Glück/Lim (not underpowered / no rail-slide). Collapse is explore/β under fixed `--ent=0.01`, matching 15:42 research note.

**Action (this fire):** Staged **cool-ent v6** for u400 handoff (no mid-run kill): wire `ENT` in `next-train-triple.sh`; continue-a/b/c → `LR=1e-4`, `ENT=0.05/0.03/0.04`, cold markers `.triple-*-cool-ent-v6`. Watchers restarted; live v5 trains undisturbed. u400 ETA ~18:20 CT (~$28).

**Watch next:** v5 finish → auto cold v6; entropy floor + near_target at_goal/UUU. Stage-gate still UUU hold ≳0.80 before multi-eq. NEED_USER_PING yes (all-slot entropy collapse + staged retune).

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


### Research pass (2026-09-19 ~16:10 CT) — AR-EAPO / random truncation + adjacent out-of-scope

Digged AR-EAPO (arXiv:2409.08938 IROS'24 AI Olympics; arXiv:2505.07516 / ICRA 2025 global-policy update), axPPO (arXiv:2405.04664 return-scaled ent), Li RA-L 2024 CoM UTPR, re-checked fawraw README/CHANGELOG (still last code **2026-06-25**). Live context: cool-ent **v6** already staged for u400 (ENT 0.05/0.03/0.04); P0 progress + P1 flip live on v5. **Direction unchanged.** Top *unimplemented* lever remains **two-policy swing↔hold**. No user ping (new cookbook fills under existing P1/P2 entropy/explore family; does not displace handoff as #1).

#### AR-EAPO — single-policy swing+hold that worked on acrobot/pendubot

IROS/ICRA AI Olympics solutions hit **swing-up + stabilize with one policy** and *light* reward engineering via average-reward MaxEnt PPO (not our discounted product+progress). Plant is **2-DOF underactuated** (joint torque), not cart-triple — transfer is algorithmic, not plant numbers.

| Knob | IROS'24 | ICRA'25 global | Steal for us? |
|---|---|---|---|
| Objective | Average-reward MaxEnt | same | Full port = P2 rewrite of `ppo.py` advantage |
| Reward GAE λ / entropy GAE λₑ | **0.8 / 0.6** | same | Separate entropy advantage (beyond fixed `--ent`) |
| Temperature τ | 2.0 | **1.5** | MaxEnt scale; related to our β |
| PPO clip ε | **0.05** | same | Tighter than typical 0.2 — consider on cool retune |
| Gain step η | 0.01 | same | Average-reward specific |
| **p_trunc** (per-step random truncate) | **1e-3** | **5e-3** | **Cheap MDP steal** — biases *faster* swing-up (horizon ≈1/p; stops amortizing swing cost over long holds) |
| Reset noise variance | (smaller) | **4.0** | Wide ICs for global policy; we already have Lim-wide / hang mixes |
| Reward shape | light quadratic; **omit torque penalty** when speed matters | same | Don't over-penalize |u| during swing phase |

**Why it matters now:** overnight entropy collapse under fixed `--ent=0.01` is exactly the explore failure AR-EAPO's separate entropy advantage + MaxEnt temp target. cool-ent v6 (raise ENT floor) is the right *cheap* first fix already staged. If v6 still leaves near_target at_goal/UUU flat, next options in order:

1. Keep PPO; add **`p_trunc≈0.001–0.005`** random episode truncate on swing slots (A/C) — one-line MDP change, no algo rewrite.
2. AE-PPO / H(a) from 15:42 pass (adaptive β / executed-action entropy).
3. Optional full AR-EAPO or SAC/TQC slot (P2) if single-policy path is kept vs split nets.

**Does not demote handoff:** AR-EAPO shows single-policy *can* work on 2-DOF with MaxEnt+avg-reward; cart-triple papers that hit hardware (Baek/Lim) still used SAC/TQC + (often) specialists or VER. Two-policy remains highest-ROI *architecture* change if UUU hold stays ~0 after v6.

#### axPPO (arXiv:2405.04664) — note only

Scales entropy coef by recent return. Weaker evidence than AE-PPO / AR-EAPO for continuous swing-up; prefer AE-PPO schedule or AR-EAPO λₑ split if coding adaptive explore.

#### Li RA-L 2024 CoM UTPR — out of scope

Operational-space QP balancing for **passive-first-joint** vertical UTPR (active joints 2–3). Balance/tracking only — not cart-actuated, not swing-up, not 56. Same adjacent bucket as Cambridge Robotica 2026 CSAC-QI. No recipe steal.

#### Still empty / unchanged

- fawraw M4 soft-landing coefs: still unspecified; last code **2026-06-25**.
- Serial cart-triple classical energy coeffs: still none.
- Force 80–100: still demoted (Glück+Baek).
- Rank: after v5→v6 cook, **split swing vs hold** still #1 unimplemented; **p_trunc** joins PBRS + ent floor as cheap co-travelers.

#### Amend recommended redesign (additions only)

40. If cool-ent v6 still flat on near_target at_goal/UUU: try **`p_trunc∈[1e-3,5e-3]`** on swing slots before more Newtons or full algo rewrite.
41. Optional P2 single-policy path: AR-EAPO cookbook (λ=0.8, λₑ=0.6, τ≈1.5–2, clip ε=0.05, η=0.01) **or** Baek/Lim SAC·TQC — prefer handoff if coding budget allows only one architecture change.
42. Ignore Li CoM / UTPR QP for cart-triple UUU.

**No code this fire** (overnight owns train; wait for green-light / overnight ask). NEED_USER_PING no.


### Research pass (2026-09-19 ~16:40 CT) — on-policy flip fix + hybrid classical catcher

Digged Mittal et al. ICRA 2024 (arXiv:2403.04359) + Su/Huang IROS 2024 symmetry RL (PDF), Aström–Furuta energy bang-bang thresholds, Machines 2025 hybrid PPO–SMC (Mon), re-checked fawraw commits (still last *code* **2026-06-25** / docs **2026-07-02**). Live v5 still cooking (~u220; A entropy **~−1.0**, B/C ~−0.7…−0.75; near_target at_goal/UUU still ~0.04). cool-ent **v6** already staged — do not mid-kill. **Direction unchanged.** Top *unimplemented* lever remains **two-policy swing↔hold**. No user ping (refines shipping P1 flip + handoff hold options; does not displace handoff as #1).

#### Live `--flip-augment` ≠ Baek VER and ≠ literature PPOaug

Baek VER is **off-policy SAC replay** doubling (mirror transitions into buffer). Our shipping code (`train_triple.py` after GAE) is an on-policy port:

```text
obs/raw/log ← concat(real, flipped);  log_f = π_θ(−a | s_flip)  # recomputed at update start
adv/ret ← concat(adv, adv)           # reused, not re-GAE'd on flipped values
```

| Source | Recipe | Vs ours |
|---|---|---|
| Baek EAAI 2024 | SAC + VER in **replay** | Off-policy-native; not our PPO path |
| Mittal ICRA'24 / rsl_rl | Augment **after minibatch sample**; keep original π_old denominator; repeat adv/ret | Prefer per-minibatch, not full-rollout concat |
| Su IROS'24 PPOaug | Same: augment inside update loop so orig+mirror share each grad step; init near-symmetric | Warns rollout-storage mirror creates off-policy samples |
| Su IROS'24 PPOeqic | Hard equivariant actor + invariant critic (EMLP) | Strict; best sample-eff in their tasks; overkill for cart C₂ |

**Pitfalls of our shipping shape:** (1) flipped actions were never sampled by the rollout policy — recomputed `log_f` is a behavior proxy, not true π_old; (2) full-batch concat before epochs weights every update on 50% synthetic samples; (3) reused adv assumes perfect reward/dynamics equivariance (true for our planar plant+product) but does not recompute V(s_flip). This may **dilute** on-policy signal under product+progress and is a plausible co-factor with entropy collapse (not proven causal).

**Implementable probes (next cold / green-light — not mid-run):**

1. **A/B one slot `FLIP_AUGMENT=0`** while others stay on — if near_target/entropy improve with flip off, shipping VER-port is net-negative.
2. **Fix to Mittal/Su PPOaug:** move mirror inside the minibatch loop; for C₂ cart, `log_old(g▷a|g▷s) ≈ log_old(a|s)` when π≈equivariant — can **repeat** stored `log_t` instead of recomputing under θ_update.
3. Optional later: equivariant actor head (PPOeqic) — only if soft augment stays weak.

Does **not** demote handoff; cheap co-traveler beside PBRS / cool-ent / `p_trunc`.

#### Hybrid classical catcher fills handoff hold half

Machines **2025** (Mon): **PPO swing-up → SMC stabilize** on cart-pole + Acrobot (not cart-triple). Same Astrom-style split as fawraw M4 / arXiv:2606.22145, but hold is **classical** not RL. Steal for our P1 when coding split nets:

| Hold option | When |
|---|---|
| RL stabilize (fawraw M3 / our hold slot B) | Prefer if we already have a near_target specialist |
| **LQR / TV-LQR** (Glück local) | Best model-based RoA once near upright |
| **SMC** (Machines 2025) | Robust to model error; no train hold policy |

Aström–Furuta (single pole, reference): energy bang-bang ∝ sign(θ̇ cos θ)·Ẽ; catch when near upright (~±30° classically). Triple still lacks published energy coeffs (Glück=feedforward) — use measured basin ≤0.1 rad for the switch, not 30°.

#### Still empty / unchanged

- fawraw M4 soft-landing coefs: still unspecified; last code **2026-06-25**.
- Serial cart-triple classical energy coeffs: still none.
- Force 80–100: still demoted.
- Rank: after v5→v6 cook, **split swing vs hold** still #1 unimplemented; **flip A/B or minibatch PPOaug fix** joins PBRS + ent floor + `p_trunc` as cheap co-travelers.

#### Amend recommended redesign (additions only)

43. Treat shipping `--flip-augment` as **approximate** Baek port — next cold: A/B flip-off on one slot **or** rewrite to **minibatch-time** PPOaug (Mittal/Su); do not assume VER sample-efficiency until that probe.
44. Handoff hold half may be **LQR/SMC** instead of a second PPO — Machines 2025 + Glück local; still switch on measured basin ≤0.1 rad + latch/hysteresis.
45. Ignore Aström ±30° as a triple gate; keep energy bang-bang only as optional classical *swing teacher*, not the switch threshold.

**No code this fire** (overnight owns train; wait for green-light / overnight ask). NEED_USER_PING no.


### Research pass (2026-09-19 ~17:07 CT) — saturated E bonus + E-gated handoff + EBERL

Digged airo7 MPC↔PPO cart-pole study (June 2026), EBERL (IEEE Access 2025, Taets et al. OA PDF), SuPLE (arXiv:2411.13613), re-checked fawraw (still last *code* **2026-06-25**). Live: cool-ent **v6** cold ~16:42 CT (~u10 @16:48; entropy healthy again); do not mid-kill. **Direction unchanged.** Top *unimplemented* lever remains **two-policy swing↔hold**. No user ping (fills empty energy/soft-landing gaps under existing P1/P2; does not displace handoff as #1).

#### Saturated energy bonus (airo7) — fills P2 E→E_UUU shape

Our 15:14 pass locked true `|E−E_UUU|` vs height-proxy `energy_w`. airo7's working **single-pole PPO** reward adds the missing anti-spin form:

\[
r = (1+\cos\phi) + 0.75\min\!\bigl(E/E^*,\,1\bigr) - 0.005\,u^2 - 0.05\,\dot\theta^2 - 0.20\,x^2 - 0.03\,\dot x^2
\]

| Knob | Value | Why |
|---|---|---|
| Energy term | **`0.75·min(E/E*, 1)`** | Dense swing signal; **saturates** so excess KE is not rewarded (spinning trap) |
| \(E^*\) | upright potential at \(\omega=\dot x=0\) | Must match plant inertia (airo7: uniform-rod \(I=\tfrac43 ml^2\); ours: mass-matrix \(T+U\) from `physics_triple`) |
| Success meter | held **1 s** inside \(\|\phi\|<0.05\) (not a single crossing) | Matches fawraw "hold not touch-and-go" |
| Curriculum | **35%** near-upright / **65%** hang + small vel noise | Same spirit as our hang/near_target mix; airo7 numbers for single pole |
| PPO hypers (their cookbook) | γ0.99, λ0.95, clip **0.2**, ent **0.01**, lr 3e-4 linear decay, 4 envs × 1024 steps, mb 256, 10 epochs | Generic; our cool-ent v6 already raised ent above 0.01 |

**Steal for our P2 energy coding:** prefer **saturated progress toward \(E_{\mathrm{UUU}}\)** over raw \(-|E-E_{\mathrm{UUU}}|\) alone. Still use our plant \(U_{\mathrm{UUU}}\approx +0.736\,\mathrm{J}\) / \(\Delta U\approx 1.47\,\mathrm{J}\) from 15:14. Does **not** replace product reward — add as optional `energy_w` replacement inside/alongside product when green-lit.

#### Dual-gate handoff with energy (airo7) — fills handoff enter condition

Already had angle≤0.1 rad / ~0 ω (fawraw) + hysteresis exit (2606.22145) + optional LQR/SMC hold (16:40). airo7's hybrid MPC adds the **energy gate** classical swing-up always needed:

| Phase | Gate |
|---|---|
| Enter balance | \(\|\phi\|\le 0.35\) **and** \(E \le 1.08\,E^*\) |
| Exit back to swing | \(\|\phi\| > 0.60\) (wider → hysteresis) |
| Rest kick (classical energy law only) | if \(\|u_E\|<0.2\) and \(E<0.05\,E^*\): kick \(\mathrm{sign}(-x)·3\,\mathrm{N}\) |

Scale 0.35/0.60 to **measured** triple basin (still ≤0.1 rad for fawraw-style catcher); keep the **\(E\le 1.08 E^*\)** idea so we do not hand off a fast overshoot. Soft-landing still: arrive with low \(\|\omega\|\) (2312.11311 −r_vel).

#### EBERL (IEEE Access 2025) — energy controller as *exploration*, not just reward

Taets / Lefebvre / Ostyn / Crevecoeur: SAC fails swing-up from **rest** on Cartpole/Furuta/Pendubot under torque limits (local min = hang). Fix: bias SAC exploration with a classical energy pump while learning.

| Piece | Recipe |
|---|---|
| Energy law | \(u_{\mathrm{eb}} = -K\,\mathrm{sat}_5\bigl((H-H^*)\,G\,\dot q\bigr)\) |
| sat₅ deadzone | **zero** when \(H\) within **±5%** of path from \(H_0\) to \(H^*\) (stops chatter at target energy) |
| SAC compose | sample \(u \sim \mathcal{N}(\mu(x,u_{\mathrm{eb}}), \Sigma)\); after train drop \(u_{\mathrm{eb}}\) → pure \(\mu(x,0)\) |
| Hold | **LQR** when \(\|x-x^*\| < b\) (DeLaN-linearized Riccati) |
| Reward | sparse cost-style \(r=\exp(-4 q_{\mathrm{tip}}^2)-1\) (negative; matches 15:14 polarity note) |
| Energy model | **DeLaN** learned online (or exact plant energy for sim-only) |
| Starts | **rest / bottom only** — no Lim-wide random ICs |

**Steal for us (cheap → expensive):**

1. **Cheap (sim):** expose exact \(E\) from `physics_triple` mass matrix; optional classical \(u_E \propto -\mathrm{sat}((E-E_{\mathrm{UUU}})\,\dot\phi\cos\phi)\) as **action bias / teacher** on swing slot A — same role as EBERL's \(u_{\mathrm{eb}}\) without DeLaN.
2. **Medium:** feed \(E\) (and \(E_{\mathrm{UUU}}\)) into obs / product term with airo7 saturation.
3. **Heavy P2 SAC slot:** full EBERL (DeLaN + energy-mean + LQR catch) if we leave PPO.

Does **not** demote two-policy handoff — EBERL *is* a swing+LQR-hold split with energy-directed explore. Reinforces P1 architecture; gives a concrete energy-explore recipe if we keep one net longer.

#### SuPLE (arXiv:2411.13613) — note only

Sum of **positive truncated Lyapunov exponents** as intrinsic reward; SAC swings + holds **double** pendulum from bottom **without** random resets (sparse/quadratic fail). Needs Jacobian / LE estimation each step — **P3** curiosity alternative if product+progress+energy+handoff all stall. Not a next-code lever.

#### Still empty / unchanged

- fawraw M4 soft-landing coefs: still unspecified; last code **2026-06-25**.
- Serial cart-triple **classical** energy bang-bang coeffs: still none (airo7/EBERL are single / Furuta / Pendubot).
- Force 80–100: still demoted.
- Rank: after v6 cook, **split swing vs hold** still #1; **saturated \(E\)** + **E-gate on handoff** + optional **energy-bias explore** join PBRS / cool-ent / `p_trunc` / flip A/B as cheap–medium co-travelers.

#### Amend recommended redesign (additions only)

46. When coding P2 energy: use **`w_E · min(E/E_UUU, 1)`** (airo7 saturate) rather than uncapped \(-|E-E_{\mathrm{UUU}}|\) or height-proxy `energy_w`.
47. Handoff enter: require **angle basin AND \(E\le\sim 1.08\,E_{\mathrm{UUU}}\)**; exit wider (hysteresis); scale angles to measured triple basin.
48. Optional swing explore: **energy-controller action bias** (EBERL-style; exact \(E\) OK in sim) before DeLaN/SAC rewrite.
49. Ignore SuPLE unless product+energy+handoff fail — then as P3 intrinsic, not overnight.
50. Rank unchanged: **two-policy swing↔hold** still highest-ROI unimplemented after v6.

**No code this fire** (overnight owns train; wait for green-light / overnight ask). NEED_USER_PING no.


### Research pass (2026-09-19 ~17:33 CT) — Dyad E_margin + RoA handoff + ASAP soft actions

Digged Dyad `CartWithSwingup` (JuliaHub MultibodyComponents example), arXiv:2606.28627 (cart-pole energy→LQR reachability / certified RoA handoff), ASAP AAAI'26 (arXiv:2601.18479 action smoothness; code AIRLABkhu/ASAP), re-checked fawraw (still last *code* **2026-06-25** / no new M4 soft-landing). Live: cool-ent **v6** cooking on A/B/C; VM `cartpole-train-od` RUNNING; do not mid-kill. **Direction unchanged.** Top *unimplemented* lever remains **two-policy swing↔hold**. No user ping (fills classical energy-law + soft-delivery cookbooks under existing P1/P2; does not displace handoff as #1).

#### Dyad energy swing-up law — fills empty classical \(u_E\) cookbook

17:07 / EBERL locked *shape* of energy explore / saturated reward, but still had **no concrete cart-pole bang-bang coeffs**. Dyad ships a working single-pole hybrid with copy-paste numbers (plant-scale differs — treat as **ratios**, retune \(k_{\mathrm{swing}}\) / \(E_{\mathrm{margin}}\) on our \(U_{\mathrm{UUU}}\approx0.736\,\mathrm{J}\)):

\[
u_{\mathrm{swing}} = k_{\mathrm{swing}}\bigl(E-(E_r+E_{\mathrm{margin}})\bigr)\,\mathrm{sign}(\dot\phi\cos(\phi-\pi)) - k_x x - k_v \dot x
\]

| Knob | Dyad default | Steal |
|---|---|---|
| \(E_r\) | upright total energy @ \(\omega=\dot x=0\) | our mass-matrix \(E_{\mathrm{UUU}}\) (not height-proxy) |
| **\(E_{\mathrm{margin}}\)** | **0.4** (same units as \(E_r\)) | Pump **above** upright so tip **passes** with small \(\omega\) for catcher |
| \(k_{\mathrm{swing}}\) | 100 | Scale to forceLimit 40–50; start lower if saturates always |
| \(k_x, k_v\) | 2, 4 | Cart centering during pump (fights rail-slide without barrier-only) |
| \(\phi_{\mathrm{switch}}\) | 0.4 rad (~23°) | Single-pole only — **triple still use measured ≤0.1 rad** basin |
| Saturation | \(\|u\|\le 12\) | Maps to our forceLimit; not a reason to jump to 80–100 |
| \(E\) definition | **cart-frame** (Galilean: subtract cart×link momentum cross terms) | Prefer plant \(T+U\) consistent with `physics_triple`; document frame |

**Complements airo7 dual-gate (do not confuse):**

| Role | Target |
|---|---|
| Swing energy *setpoint* (Dyad / classical) | \(E\to E_{\mathrm{UUU}}+E_{\mathrm{margin}}\) (slight overshoot) |
| Handoff *enter* gate (airo7) | angle basin **and** \(E\le\sim 1.08\,E_{\mathrm{UUU}}\) (reject fast flyers) |
| Soft-landing | low \(\|\omega\|\) at enter (2312.11311 −r_vel) |

So: classical teacher **aims slightly hot**; catcher **refuses too-hot**. Same split EBERL/Machines'25 already argued (energy swing + LQR/SMC hold).

#### Certified RoA handoff (arXiv:2606.28627) — why catch-basin measurement matters

Formalizes the architecture we already ranked #1:

1. Energy shaping drives onto the upright **homoclinic** (energy error → 0).
2. Augmented Lyapunov also drives **cart velocity → 0** (almost-global).
3. Local LQR has a certified ellipsoidal **RoA**; the **switching region must lie strictly inside** that RoA (one-way handoff).
4. End-to-end reachability = swing delivers into that interior set.

**Steal (design rule, not new #1):** when coding handoff, treat fawraw `measure_catch_basin.py` as the empirical RoA probe — widen hold basin **or** soft-deliver until swing's arrival set ⊂ measured catch set. Do not switch on angle alone if \(\dot x\) / \(\omega\) leave the LQR ellipsoid. Matches Glück ~0.6 m overshoot hypersensitivity already noted.

#### ASAP (AAAI 2026) — PPO soft-delivery co-traveler

fawraw M4 soft-landing coeffs still unspecified. ASAP gives a **trainable** smoothness prior for our PPO swing slot (complements LPF τ≈0.3 from 2606.22145):

| Term | Idea |
|---|---|
| Spatial \(L_S\) / predictor \(L_P\) | Align \(\pi(s_t)\) with action predicted from preceding state \(s_{t-1}\) (transition-induced similar states) |
| Temporal \(L_T\) | Penalize **second-order** action diffs (high-freq chatter / bang-bang) |
| Total | \(J_\pi + \lambda_S L_S + \lambda_P L_P + \lambda_T L_T\) |
| PPO tip | Prefer **smaller \(\lambda_P\)** + many parallel envs (we already run 8192) |
| Code | https://github.com/AIRLABkhu/ASAP |

**When:** after / with two-policy handoff if swing still arrives bangy; optional cheap A/B on slot A before full split. Not a substitute for split nets. Chinese "smooth exploration PPO" (mutation + oscillation regs, Electronic Science & Technology 2025) is same family, thinner cookbook — prefer ASAP if coding.

#### Still empty / unchanged

- fawraw M4 soft-landing **numeric** coefs: still unspecified; last code **2026-06-25**.
- Serial **cart-triple** classical energy coeffs: still none (Dyad/2606.28627 are single-pole — scale + measure).
- Force 80–100: still demoted.
- Rank: after v6 cook, **split swing vs hold** still #1; **Dyad \(E_{\mathrm{margin}}\) teacher** + **RoA-strict switch** + **ASAP / LPF soft actions** join saturated-\(E\) / E-gate / EBERL-bias / PBRS / cool-ent / `p_trunc` / flip A/B as co-travelers.

#### Amend recommended redesign (additions only)

51. Classical / EBERL energy bias: target **\(E_{\mathrm{UUU}}+E_{\mathrm{margin}}\)** (start \(E_{\mathrm{margin}}/E_{\mathrm{UUU}}\sim0.05\)–0.2, Dyad's 0.4/\(E_r\) is ~20% — retune); keep cart \(k_x,k_v\) terms; use \(\mathrm{sign}(\dot\phi\cos\phi)\) form adapted to θ=0 upright.
52. Handoff switch set must be **strict subset of measured hold RoA** (2606.28627); angle∩energy∩low-\(\omega\) gates are the practical proxy — run catch-basin measure before trusting latch.
53. Soft-delivery: prefer **LPF τ≈0.3** first; if chatter remains, add **ASAP** \(\lambda_T\) (and light \(\lambda_S\)) on swing PPO rather than inventing fawraw soft-landing weights.
54. Rank unchanged: **two-policy swing↔hold** still highest-ROI unimplemented after v6.

**No code this fire** (overnight owns train; wait for green-light / overnight ask). NEED_USER_PING no.


### Research pass (2026-09-19 ~18:08 CT) — Raffin massive-parallel SAC + EvolSAC height-gate + PBRS bias

Digged Raffin Jul 2025 / ICLR Blogposts 2026 SAC-on-Isaac (massive parallel), EvolSAC (arXiv:2507.10030 IROS'24 AI Olympics underactuated), Müller/Kudenko PBRS effectiveness (arXiv:2502.01307), Xin coupling-energy limited-track swing-up (JVC 2025 — single-pole + barrier), IEEE Access 2025 DIPC hybrid energy→SMC (architecture only; no open numeric switch), re-checked fawraw commits (last *code* still **2026-06-25**, last *docs* **2026-07-02** — no soft-landing coefs), Baek VER (no new on-policy / multi-link paper). Live: cool-ent **v6** cooking; do not mid-kill. **Direction unchanged.** Top *unimplemented* lever remains **two-policy swing↔hold**. No user ping (fills SAC/TQC + PBRS cookbooks under existing P1/P2; does not displace handoff as #1).

#### Raffin massive-parallel SAC — copy-paste hypers for *our* env count

Prior passes locked Baek/Lim SAC·TQC as the off-policy alternative and AR-EAPO/`p_trunc` as PPO explore fixes. Missing: **wall-clock** hypers when you already run thousands of envs (we use 8192). Raffin's Optuna-for-speed recipe on Isaac (1024 envs) is the first concrete steal:

| Knob | Default SAC (sample-eff) | Raffin speed-tuned | Steal for optional P2 SAC/TQC slot |
|---|---|---|---|
| Replay ratio \(g/(N_{\mathrm{env}}·f_{\mathrm{train}})\) | ≳1 | **≈0.03** | Cheap data → few grads per env-step |
| `batch_size` | 256 | **512** | |
| `gamma` | 0.99 | **≈0.983** | Slightly shorter horizon |
| `learning_rate` | 3e-4 | **≈4.5e-4** | |
| `tau` | 0.005 | **≈0.0023** | |
| `policy_delay` | 1–2 | **8** | Critic-heavy like TD3 |
| `net_arch` | [256,256] | **[512,256,128]** + LN + AdamW | Match PPO-scale nets |
| `ent_coef` | auto | auto with init **~0.01** | Same order as our cool-ent floor |
| Hard-task add-ons | — | `train_freq=10`, `gradient_steps=320` (keep RR), **`use_sde=True`**, **`n_steps=3`** | gSDE for consistent explore; FastTD3-style n-step closes PPO gap |
| Action bounds | full plant limit | **PPO 2.5–97.5% percentiles** of a trained policy | Do **not** jump force 80–100; shrink/reshape effective \(u\) first |
| TQC vs SAC | — | TQC easier to tune; SAC faster once tuned; TQC wins on hardest env | Prefer **TQC** if one hard UUU slot; else SAC+n-step |

**Implication:** overnight PPO entropy collapse is *not* an argument to crank force. If we ever leave PPO for a single-net path, use **low RR + n-step + gSDE**, not textbook SAC. Still prefer **two-policy handoff** if coding budget allows only one architecture change (Baek/Lim hardware path used SAC/TQC *and* often specialists).

#### EvolSAC (arXiv:2507.10030) — height-gated surrogate + SNES polish

SAC (then SNES) on cart-pole + RealAIGym acrobot/pendubot. Does **not** replace handoff; fills reward/finetune cookbook under the SAC path:

| Piece | Recipe |
|---|---|
| Surrogate (cart-pole) | \(R=-\|p_{\mathrm{tip}}-p_{\mathrm{up}}\|\) (dense tip distance) |
| "Held" definition | tip within **0.1** of upright for rest of episode (matches fawraw basin scale) |
| \(u_{\max}\) cart-pole | **2.5 N** (toy plant — scale, don't copy) |
| Double-pend height gate | high-reward regime if \(y>y_{\mathrm{th}}\): **0.375 m** acrobot / **0.35 m** pendubot (\(y_{\max}=0.5\)) |
| Weights (Table 1) | \(\tau_{\max}=3\), \(\alpha=2\), \(\beta=1\), \(\rho_1=0.1\), \(\rho_2=0.02\), \(\phi_1=\phi_2=0.15\), \(\eta=0.02\) |
| High regime | \(V+\alpha[1+\cos\theta_2]^2-\beta T-\rho_1 a^2-\phi_1\Delta a\) |
| Low regime | \(V-\rho_2 a^2-\phi_2\Delta a-\eta\|\dot q\|^2\) |
| SNES polish | pop **40**, \(\sigma=0.02\) (cart) / **0.01** (double); optimize sparse score from SAC warm-start |
| Torque note | \(\tau_{\max}=1.5\) too weak (stuck mid configs); **5.0** too thrashy; **3.0** sweet spot |

**Steal:** optional **height/align gate** that switches from velocity-penalized pump → energy/align-rich hold terms (same spirit as our hang→near_target curriculum). SNES is P3 polish after a working UUU policy — not overnight.

#### PBRS effectiveness (arXiv:2502.01307) — bias + exponential Φ

15:42 locked PBRS form \(F=\gamma\Phi(s')-\Phi(s)\) vs raw Δ. Müller/Kudenko add the missing scale/offset cookbook:

| Knob | Recipe | Steal |
|---|---|---|
| Shifted potential | \(\Phi_b(s)=\Phi(s)+\frac{b}{\gamma-1}\) (non-terminal) | Set \(b=(1-\gamma)Q_{\mathrm{init}}-r_\infty\) so first TD steps actually follow Φ |
| Terminal | \(\Phi(\mathrm{terminal})=0\) (required for policy invariance) | Truncation / goal / oob death → Φ=0 |
| Continuous Φ bug | small \(\delta\Phi\) can get the **wrong sign** of \(F\) | Prefer **exponential** \(\mathrm{e}^{\Phi}\) with base **\(e\approx 32\)** (their default) so small upright steps still incentivize |
| Scale bound (goal-directed) | \(r_\infty-(1-\gamma)Q_{\mathrm{init}}<\Phi<r_g-(1-\gamma)Q_{\mathrm{init}}\) | Don't crank `progress_w` alone — mismatch Φ vs reward/init breaks guidance |

Our shipping progress is still raw Δ (`progress_w*(align_now−align_prev)`). Next green-lit progress touch: **PBRS γ-diff + optional constant bias + exp Φ** (align or −‖θ−θ*‖), not a bigger `progress_w`.

#### Still empty / unchanged

- fawraw M4 soft-landing **numeric** coefs: still unspecified; last code **2026-06-25** (docs-only **2026-07-02**).
- Serial **cart-triple** classical energy bang-bang coeffs: still none (Xin JVC 2025 = **single-pole** coupling-energy + track barrier; Xin TIE 2025 n-link = **antiswing at downward** EP, not swing-up; Glück = feedforward BVP).
- Force 80–100: still demoted (Glück ~22 m/s²; Raffin says reshape action bounds before raising plant limit).
- Baek VER: no new on-policy / multi-link paper this pass (shipping flip A/B fix from 16:40 still the on-policy lever).
- Rank: after v6 cook, **split swing vs hold** still #1; Raffin SAC hypers + EvolSAC height-gate + PBRS bias/exp join prior co-travelers.

#### Amend recommended redesign (additions only)

55. If coding optional P2 SAC/TQC slot at our env count: start from **Raffin speed hypers** (RR≈0.03, batch 512, γ≈0.983, policy_delay 8, net [512,256,128], then `n_steps=3` + gSDE on hard UUU) — not SB3 defaults.
56. Prefer **TQC** for the single hardest UUU specialist; SAC+n-step otherwise. Cap effective actions via **percentile bounds** before raising `forceLimit`.
57. Progress retune: after γ-PBRS, add **Φ bias** \(b/(γ-1)\) matched to typical return scale + try **exp Φ** (base ~32) if small align steps farm wrong sign.
58. Optional curriculum: EvolSAC-style **height/align gate** switching pump vs hold reward terms (numbers in table above — retune to our \(U_{\mathrm{UUU}}\)).
59. Rank unchanged: **two-policy swing↔hold** still highest-ROI unimplemented after v6.

**No code this fire** (overnight owns train; wait for green-light / overnight ask). NEED_USER_PING no.


### Research pass (2026-09-19 ~18:36 CT) — FastTD3 / CrossQ anti-pattern + CLF-RL hold shaping

Digged FastTD3 (arXiv:2505.22642 + younggyoseo/fasttd3; PQL recipe), CrossQ+WN (arXiv:2506.03758), CLF-RL stability (arXiv:2605.01978 / Li–Olkin RA-L 2026 practical form), re-checked fawraw (still last *code* **2026-06-25** / docs **2026-07-02**). Live cool-ent **v6** ~u200–210 (A entropy~1.31 nt_align/UUU~**+0.24** hang_align/UUU~**+0.025**; B entropy~0.39 cooling — v7b ENT0.08 staged; C entropy~1.20 nt_align~+0.19); **nt_at_goal/UUU still ~0.04–0.06**. Do not mid-kill. **Direction unchanged.** Top *unimplemented* lever remains **two-policy swing↔hold**. No user ping (fills P1-hold + P2-off-policy cookbooks; does not displace handoff as #1).

#### Live read (why handoff still #1)

| Slot | ~u | entropy | nt_at_goal/UUU | nt_align/UUU | hang_align/UUU |
|---|---|---|---|---|---|
| A swing | ~210 | **~1.31** | ~0.061 | **~+0.241** | **~+0.025** |
| B hold | ~201 | ~0.39 ↓ | ~0.042 | ~+0.168 | ~−0.16 |
| C combo | ~200 | ~1.20 | ~0.048 | ~+0.190 | ~−0.07 |

**Diagnosis:** progress+flip+cool-ent **moves align** (A hang UUU align crossed positive; best curriculum signal so far) but **does not unlock hold** (at_goal stuck ~0.04–0.06). Same failure mode as fawraw M4 (reach without catch). Overnight owns B v7b; research does not retune mid-run.

#### FastTD3 (arXiv:2505.22642) — wall-clock off-policy at our env scale

18:08 locked Raffin **SAC** speed hypers. FastTD3 is the **TD3/distributional** cousin from Parallel Q-Learning (PQL): same “thousands of envs → few grads / big batches” philosophy, often **matches PPO wall-clock** on Isaac/Playground.

| Knob | Steal for optional P2 slot |
|---|---|
| Parallel envs | Already have **8192** — FastTD3 is built for this |
| Critic | **Distributional** (Bellemare atoms) — same family as Lim TQC |
| Batch | **Large** (repo examples **8192**; scale to GPU) |
| `n_steps` | **>1** (repo notes Raffin fix Jun 2025 stabilizes n-step) |
| Architecture | Prefer **FastTD3 + SimbaV2** (maintainers’ default since 2025-06) |
| Reward caveat | Off-policy may need **different** reward shaping than PPO-tuned product — retune penalties if gait/force looks wrong at same return |

**Vs Raffin SAC:** use FastTD3/TQC when we want distributional critics + TD3 delay; use Raffin SAC when we want entropy auto-tune + gSDE. Both beat textbook high-RR SAC at our env count. Still prefer **two-policy handoff** if only one architecture change.

#### CrossQ+WN (arXiv:2506.03758) — do **not** pick for UUU

CrossQ+weight-norm scales UTD on DMC dog/humanoid, but authors explicitly call out **poor performance on sparse `pendulum-swingup`** (attribute to sparse reward). Our harsh `eval/at_goal/UUU` is the same sparse-hold meter. **Anti-pattern:** skip CrossQ for the optional off-policy slot; stick **TQC / Raffin SAC / FastTD3**.

#### CLF-RL hold shaping (arXiv:2605.01978) — soft-landing / hold cookbook

Fills the still-empty fawraw soft-landing **numeric** gap with a **control-Lyapunov** reward used in practice (IsaacLab + PPO) and proven exponentially stable for the optimal policy (cart-pole verified). **Local around upright** — use on the **hold** net after handoff, not as the hang→UUU swing objective.

Practical discrete reward (paper §IV-B):

| Term | Formula | Role |
|---|---|---|
| \(r_V\) | \(\beta\exp(-V/\sigma^2)\) | Dense proximity to upright (V = CLF / quadratic on error) |
| \(r_{\Delta V}\) | \(-\rho\,\mathrm{clip}\bigl((\Delta V+\lambda V)/\sigma_{\dot V},\,0,\,1\bigr)\) | Penalize CLF increase (stability decrease condition) |
| \(r_{\mathrm{reg}}\) | \(-w_u\|u\|^2\) | Soft actuation |
| Total | \(R=r_V+r_{\Delta V}+r_{\mathrm{reg}}\) | Max \(\beta\) at \(V=0\) |

Starter scales (retune): \(\lambda\) slightly below CLF rate \(\alpha\); \(\beta\) ~ product-hold magnitude; \(\sigma\) so typical near-basin \(V\) sits mid-exp; \(\rho\) so one bad \(\Delta V\) step ≈ a few \(r_V\) units. For our plant, a cheap V before a true CLF: \(V=\sum_i w_i(1-\cos\phi_i)+c_x x^2+c_\omega\|\omega\|^2\) near UUU (world angles), or LQR Riccati quadratic once a linearization exists.

**Steal:** when coding two-policy, train **hold** with CLF-RL (or quadratic+decrease) from **near_target / measured basin** ICs; keep product+progress (+ optional saturated E) on **swing**. Complements ASAP/LPF soft delivery on the swing side and E-gate on the switch.

#### Still empty / unchanged

- fawraw M4 soft-landing **numeric** coefs: still unspecified (CLF-RL is our substitute cookbook).
- Serial cart-triple classical energy bang-bang coeffs: still none.
- Force 80–100: still demoted.
- Baek VER: no new multi-link / on-policy paper (shipping flip A/B from 16:40 still the on-policy lever).
- Rank: after v6 cook, **split swing vs hold** still #1; FastTD3 (+SimbaV2) joins Raffin SAC under P2; **CLF-RL hold reward** joins ASAP/E-gate/soft-landing under P1; **CrossQ banned** for sparse UUU.

#### Amend recommended redesign (additions only)

60. Optional P2 off-policy: prefer **TQC / Raffin SAC / FastTD3(+SimbaV2)**; **never CrossQ** for sparse UUU hold.
61. FastTD3 starter: distributional critic + large batch + `n_steps>1` + SimbaV2; retune product penalties if behavior diverges from PPO at same return.
62. Hold net reward: adopt **CLF-RL** \(r_V+r_{\Delta V}+r_{\mathrm{reg}}\) (formulas above) from basin ICs; swing keeps product+progress (+ saturated E when coded).
63. Rank unchanged: **two-policy swing↔hold** still highest-ROI unimplemented — live v6 align↑/at_goal flat is the empirical confirmation.

**No code this fire** (overnight owns train / B v7b; wait for green-light / overnight ask). NEED_USER_PING no.


### Research pass (2026-09-19 ~19:00 CT) — ERA entropy floor + Fattahi energy-mod PPO + SimbaV2 fills

Digged ERA (arXiv:2510.08549 + nothingbutbut.github.io/era), Fattahi UniPD thesis *Learning-based Energy Control of Underactuated Robots* (Padova, July 2026 PDF), MPC-informed residual RL (KU Leuven LearnOpTra / Furuta 2026 poster), SimbaV2 cookbook fills (arXiv:2502.15280; FastTD3 already named it). Re-checked fawraw (GitHub API rate-limited this fire; last known *code* still **2026-06-25** / docs **2026-07-02**). Overnight owns train (cool-ent v6 / B v7b staged earlier); research does not mid-kill. **Direction unchanged.** Top *unimplemented* lever remains **two-policy swing↔hold**. No user ping (fills entropy-floor + energy-residual cookbooks under existing P1/P2; does not displace handoff as #1).

#### ERA (arXiv:2510.08549) — entropy floor via *activation*, not β crank

Our cool-ent / AE-PPO / axPPO path still fights collapse by **raising the entropy *coefficient***. ERA instead constrains sampling entropy with a specially designed **output activation** on the actor’s `log_std`, so the PPO/SAC *loss stays pure reward* (no objective distortion from a large β). Continuous Gaussian recipe (their Listing 2, JAX):

```text
# h_0: target entropy (fixed or learnable); default −dim(A)/2
# pre_stds: raw actor head
k = −action_dim * (log_std_max + h_0 + log(sqrt(2*π*e)))
log_stds = k * softmax(pre_stds, axis=−1) + log_std_max
log_stds = clip(log_stds, log_std_min, log_std_max)
```

vs the usual tanh squash of `log_std` into `[log_std_min, log_std_max]`. Provable lower bound on policy entropy; <7% overhead; +25–30% on hard DMC / HumanoidBench SAC; also ships PPO hypers (Table 6: clip 0.2, ent_coef **0.01**, γ0.99, λ0.95, batch 2048 / mb 64 — generic).

**Steal for us:** if v6/v7b entropy cools again under fixed `--ent`, prefer **ERA on the Gaussian head** (or ERA + modest β) over another cold wipe + higher ENT. Orthogonal to two-policy; works on **both** swing and hold nets. Code: https://nothingbutbut.github.io/era

#### Fattahi UniPD thesis (July 2026) — PPO *modulates* energy, does not output raw force

Closest published **on-policy energy residual** to our P2 E→E_UUU + EBERL bias ideas. Plant is Acrobot/Pendubot (joint torque), **not** cart-triple — steal the *architecture*, retune numbers on `physics_triple` / \(U_{\mathrm{UUU}}\approx0.736\,\mathrm{J}\).

| Piece | Recipe |
|---|---|
| Action | Policy outputs scalar \(a\in[-1,1]\); applied \(u=\mathrm{clip}(\tau_{\mathrm{PD}}+a\,\bar u_{\max}\phi(x),\,\pm u_{\max})\) with \(\bar u_{\max}=0.7\,u_{\max}\) |
| Energy gain | \(\phi=\tanh\bigl((E-E^\star)/\sigma_E\bigr)\), \(\sigma_E=e_{\mathrm{scale}}\sigma_V\), \(e_{\mathrm{scale}}=0.3\); \(\sigma_V=\|V(x_0)-V^\star\|\) |
| PD co-term | Collocated on actuated joint (Acrobot \(k_p{=}0.60,k_d{=}0.10\); Pendubot \(1.20/0.25\)) — leaves headroom so sum rarely saturates |
| Obs | \(\sin/\cos\) joints + \(\dot q\) + \(\tanh((T-T^\star)/\sigma_T)\) + \(\tanh((V-V^\star)/\sigma_V)\) (+ VecNormalize ±10, freeze after train) |
| Reward | \(r=-(E-E^\star)^2/\sigma_V^2 + w_H H(q) - \lambda_u (a\bar u_{\max}/u_{\max})^2 + c_1 e^{-5e_1^2}+c_2 e^{-5e_2^2}\) (\(w_H{=}1\), \(\lambda_u{=}10^{-3}\)) |
| PPO (Table 4.4) | nets [256,256]; **8** envs; \(n_{\mathrm{steps}}{=}4096\); batch 512; 20 epochs; γ**0.995**; λ0.98; clip **0.1**; lr \(3{\times}10^{-4}\) const; ent \(10^{-3}\); **gSDE** sample freq **4**; grad clip 0.5; 1e5 / 3e5 steps |
| Starts | Near-bottom random (not Lim-wide); early-term on upright success |
| Classical sibling | EnergyLQR (Xin-style swing) → LQR with **latch** + outer safeguard re-swing; capture e.g. Acrobot \(\delta{=}0.05\), \(\omega\approx0.03\)–0.10 |

**Steal for cart-triple swing slot A (when green-lit):**

1. Expose plant \(E=T+U\) from `physics_triple` mass matrix (already planned 15:14 / 17:07).
2. Replace raw force head with **\(a\)-modulated** \(\bar F_{\max}\phi(E)\) + light cart PD (\(k_x,k_v\) from Dyad 17:33) — same role as EBERL \(u_{\mathrm{eb}}\) mean-bias, but **PPO-native** and residual-capped at 70% forceLimit.
3. Append \(\tanh((E-E_{\mathrm{UUU}})/\sigma_E)\) (and optional \(T,V\) splits) to obs; keep product+progress as outer reward **or** swap swing reward to energy-dominated form above during a specialist swing stretch.
4. Still hand off to a **hold** net / LQR / CLF-RL once basin∩energy∩low-\(\omega\) (rank #1 unchanged).

Optional velocity-direction multiplier on \(\phi\) (classical \(\mathrm{sign}(\dot\phi\cos\phi)\)) was tried in the thesis and **not** used in the reported controllers — keep Dyad/EBERL sign form as a separate A/B, not default.

#### MPC-informed residual (LearnOpTra 2026) — same residual family, heavier base

Furuta with domain-randomized tip mass: \(a = a_{\mathrm{MPC}} + a_{\mathrm{RL}}\) with **MPC planned sequence + predicted traj in the RL obs**. Converges ~400k steps vs ~600k plain residual vs ~3.5M plain RL. **Too heavy** for overnight (MPC each step), but validates the residual pattern: classical energy/Dyad/EBERL as \(a_0\), PPO residual as \(a_{\mathrm{RL}}\) — Fattahi is the cheap sim-native version.

#### SimbaV2 fills (for optional P2 FastTD3/SAC slot)

18:36 named FastTD3+SimbaV2; missing cookbook:

| Knob | Steal |
|---|---|
| Norm | Replace LayerNorm with **hyperspherical** \(\ell_2\) feature + project weights onto unit sphere after each update |
| Critic | **Distributional** + **reward scaling** (stable grads under reward magnitude swings — relevant if we mix product + energy + CLF) |
| Default compute | **UTD=2**, batch **256**, Adam **no** weight decay, lr linear **1e-4 → 3e-5** |
| Widths | Actor ~128 / critic ~512 (scale critic first) |
| Reset | Periodic reinit **hurts** SimbaV2 — skip |

Prefer with Raffin low-RR / FastTD3 large-batch at our 8192 envs; still **never CrossQ** for sparse UUU.

#### Still empty / unchanged

- fawraw M4 soft-landing **numeric** coefs: still unspecified (CLF-RL / ASAP remain substitutes).
- Serial **cart-triple** classical energy bang-bang coeffs: still none (Fattahi/Dyad/EBERL = Acrobot/Pendubot/Furuta/single-pole — scale + measure).
- Force 80–100: still demoted.
- Baek VER: no new multi-link / on-policy paper.
- Rank: after v6 cook, **split swing vs hold** still #1; **ERA log_std floor** joins cool-ent / AE-PPO / axPPO under explore; **Fattahi energy-mod PPO** joins saturated-E / EBERL-bias / Dyad \(E_{\mathrm{margin}}\) under P2 swing; SimbaV2 hypers fill P2 off-policy.

#### Amend recommended redesign (additions only)

64. If entropy collapses again under fixed `--ent`: try **ERA** on the Gaussian `log_std` head (target \(\mathcal{H}_0\approx-\dim(\mathcal{A})/2\)) before another ENT cold wipe — keeps reward objective clean.
65. When coding P2 energy swing: prefer **Fattahi structured action** \(u=\tau_{\mathrm{PD}}+a\cdot 0.7 F_{\max}\cdot\tanh((E-E_{\mathrm{UUU}})/\sigma_E)\) + energy-error obs features over raw-force PPO with only an energy *reward* term.
66. Optional residual ladder: Dyad/EBERL classical \(a_0\) → Fattahi-style learned modulation → (heavy) MPC-informed residual only if sim energy residual stalls.
67. Optional P2 FastTD3/SAC: pair with **SimbaV2** hyperspherical + reward scaling + UTD≈2 (or Raffin low-RR at our env count); skip weight-decay resets.
68. Rank unchanged: **two-policy swing↔hold** still highest-ROI unimplemented after v6 — align↑ / at_goal flat from 18:36 still the empirical confirmation.

**No code this fire** (overnight owns train; wait for green-light / overnight ask). NEED_USER_PING no.


### Research pass (2026-09-19 ~19:40 CT) — Mon verticality switch + cart-triple LQR Q/R + soft-landing pack

Digged Mon Machines 2025 PPO→SMC verticality formula (MDPI), ResearchSquare 2026 *Dynamics and LQR Control of a Triple Inverted Pendulum on a Cart* (rs-10173980/v1), and re-confirmed fawraw `m4_findings.md` via CDN (still mid-2026; GitHub API rate-limited). Live: overnight owns train (cool-ent v6 / B entboost **v7b** ENT0.08 LR5e-5 / C entboost **v7** staged); research does not mid-kill. **Direction unchanged.** Top *unimplemented* lever remains **two-policy swing↔hold**. No user ping (fills switch + hold-LQR + soft-landing gaps under existing P1; does not displace handoff as #1).

#### Mon Machines 2025 — concrete verticality switch (was table-only)

Already had "PPO swing → SMC hold" as a classical catcher option (16:40). Missing cookbook:

| Piece | Steal |
|---|---|
| Verticality (Acrobot) | \(V = \bigl[-\cos\theta_1-\cos(\theta_1+\theta_2)\bigr]/2\) |
| Switch PPO→SMC | \(V > 0.9\) |
| Near-success after switch | \(V > 0.98\) within ~30 steps |
| Cart-pole sibling | angle error \(<0.5^\circ\) after SMC catch; long-horizon balance OK |
| Acrobot caveat | **SMC failed sustained upright** (chatter / unmodeled dyn) — near-vertical OK, indefinite hold not |

**For our UUU:** analogous mean uprightness \(\bar c = \tfrac13\sum_i\cos\phi_i\) (world angles, θ=0 upright) with switch \(\bar c > 0.9\) ≈ mean |φ| ≲ 25.8°. **Do not use as the sole gate** — fawraw measured basin is ≤0.1 rad / ~0 ω; Mon's 0.9 is a *delivery* meter for swing, not a proven triple catch RoA. Prefer: enter on **measured basin ∩ energy ∩ low-ω** (17:07 / 17:33); treat \(\bar c>0.9\) as an optional *progress* soft-landing cue on the swing net.

**Hold choice:** given Mon's SMC chatter on Acrobot, prefer **LQR / CLF-RL / RL hold (slot B)** over SMC for cart-triple unless we add a chatter-robust sliding law. Rank #1 (split nets) unchanged.

#### ResearchSquare 2026 — first *cart-triple* LQR numeric cookbook

Closest published **serial cart-triple upright LQR** with open Q/R / force numbers (stabilization only — not swing-up; still the right hold-half teacher):

| Case | \(Q_\theta\) diag | \(R\) | Settle \(\theta_1\) | Notes |
|---|---|---|---|---|
| LQR-1 | 10 | 0.01 | ~4.2 s | Slow / gentle |
| **LQR-2 baseline** | **100** | **0.01** | **~2.8 s** | Reported baseline |
| Hot | →1000 | 0.01 | ~70% faster vs Q=10 | Diminishing returns **beyond \(Q_\theta\approx 200\)** |

Other locks from the same preprint:

- Peak force ≈ **18 N** in the first 0.5 s for simultaneous small 3-angle ICs (~±5° class); sat ±50 N never hit.
- Linear LQR **fails beyond ~±15°** (actuator sat + linearization) — need nonlinear / RL outside that cone.
- Controllability: full-rank Kalman on 8-state linearization about UUU with single cart force.

**Steal for hold half (when coding two-policy / LQR catch):**

1. Start Riccati with \(Q_\theta\sim 100\), \(R\sim 0.01\), cart-pos weight ≪ angle weights; retune on our \(m_c,m_i,\ell_i\) (do not copy K blindly).
2. Train / gate **hold ICs inside ~0.1–0.26 rad** (fawraw basin ∪ LQR validity) — matches "widen catcher before soft-landing" (11:06 / m4).
3. Peak-18 N on a similar mass scale **re-confirms forceLimit 40–50 is ample for hold**; still demote 80–100 (Glück+Baek+this).
4. Pair with CLF-RL \(r_V+r_{\Delta V}\) (18:36) using the LQR Riccati quadratic as \(V\) once linearized.

#### Soft-landing **numeric pack** (fills empty fawraw M4 coefs)

fawraw still has **no** soft-landing implementation (CDN `m4_findings.md` unchanged: plan only — arrive slow + cart-centred). Compose a starter from published pieces already adjacent in this doc — **not** invented weights:

| Term | Starter | Source |
|---|---|---|
| Angle calm near UUU | Baek \(e(\dot\theta)=\alpha_e+(1-\alpha_e)\exp(-c_e\dot\theta^2)\), \(\alpha_e=0.50\), \(c_e=0.09\); take **min** over links | Baek EAAI 2024 (already in product) |
| Rate product (Lim lineage) | \(R_\omega=\exp(-0.02\|\omega\|)\) per link | MDPI Machines 2025 double / Lim cousin |
| Additive near-upright brake | \(-0.05\,\dot\theta^2\) (scale to 3 links) once \(\bar c>0.9\) | airo7 saturated-E PPO (17:07) |
| Cart centre | Baek \(g(x)\) with \(c_g=0.57\) **or** Lim \(R_y=\exp(-0.3\|y\|)\) | Already locked |
| Delivery gate (swing→hold) | \(\bar c>0.9\) **and** \(\|\omega\|_\infty < \omega_{\mathrm{catch}}\) **and** \(\|x\|\) small; latch + hysteresis exit | Mon 0.9 + fawraw vel gate + 2606.22145 |
| Action smoothness | LPF \(\tau\approx 0.3\) first; ASAP \(\lambda_T\) if still bangy | 15:42 / 17:33 |

**Order of operations (unchanged):** (1) widen catcher init_noise / nonzero ω / off-centre x, (2) measure basin, (3) add soft-landing pack on **swing** only near target, (4) hand off with tol ≤ measured basin (never code default 0.3).

#### Still empty / unchanged

- fawraw M4 soft-landing **implementation**: still absent (we now have a *starter pack* from cousins; no fawraw ground truth).
- Serial cart-triple **classical energy bang-bang** coeffs: still none (Glück=feedforward; Xin/Fattahi/Dyad/EBERL = other plants).
- Force 80–100: still demoted (now also by cart-triple LQR peak ~18 N).
- Baek VER: no new multi-link / on-policy paper.
- Rank: after v6/v7 cook, **split swing vs hold** still #1; **Mon \(\bar c>0.9\)** + **LQR Q_θ≈100/R≈0.01** + **soft-landing pack** join ASAP/E-gate/CLF-RL/catch-basin under P1 handoff.

#### Amend recommended redesign (additions only)

69. When coding handoff enter: optional Mon-style \(\bar c>0.9\) as a *swing delivery* cue, always AND'd with measured basin ≤0.1 rad + low-ω (+ optional E-gate); never replace the measured RoA with 0.9 alone.
70. Hold classical catcher starter: cart-triple LQR with \(Q_\theta\sim 100\), \(R\sim 0.01\), ICs inside ~±15° / prefer ≤0.1 rad; escalate to CLF-RL / RL hold if linear RoA is too small.
71. Soft-landing on swing near UUU: enable Baek \(c_e=0.09\) / Lim \(R_\omega\) / airo7 \(-0.05\omega^2\) **gated by** \(\bar c>0.9\) (or align threshold), plus cart centre — before inventing new coefs.
72. Prefer LQR/CLF-RL hold over Mon SMC for triple (SMC chatter on Acrobot).
73. Rank unchanged: **two-policy swing↔hold** still highest-ROI unimplemented.

**No code this fire** (overnight owns train / B v7b / C v7; wait for green-light / overnight ask). NEED_USER_PING no.


### Research pass (2026-09-19 ~20:15 CT) — NNM energy walk + SA-TGaussian + staged rewards

Digged DLR/TUM ENOC 2024 *Swing-Up of a Double, Triple, and Quadruple Pendulum via Nonlinear Normal Modes* (Sachtler / Albu-Schäffer / Della Santina; elib.dlr.de/205593), AAAI-25 *Truncated Gaussian Policy for Debiased Continuous Control* (Lee et al.; ojs.aaai.org download 33988), and arXiv:2603.05113 *Decoupling Task and Behavior: A Two-Stage Reward Curriculum* (RC-SAC/RC-TD3). Re-checked fawraw (still last *code* **2026-06-25**). Live peek (do not mid-kill): A swing still cool-ent **v6** `ent05-lr1e4` f50; B hold already **v7b** `ent08-lr5e5`; C combo still **v6** `ent04` (continue-c armed for v7). L4 ~99%/15.7GB. Overnight owns train. **Direction unchanged.** Top *unimplemented* lever remains **two-policy swing↔hold**. No user ping (fills classical energy-architecture + PPO action-geometry + reward-staging cookbooks under existing P1/P2/explore; does not displace handoff as #1).

#### DLR NNM (ENOC 2024) — closest *serial triple* classical energy architecture

Still no cart-triple bang-bang coeffs (Glück=feedforward BVP; Xin/Fattahi/Dyad = other plants). NNM fills the **architecture** gap for n-link serial swing-up:

| Piece | Steal |
|---|---|
| Plant in paper | Fixed-base n-link (joint torque), equal thin rods \(l{=}0.5\,\mathrm{m}\), \(m{=}0.662\,\mathrm{kg}\) — **not** cart-actuated; retune on our \(m_c{=}1,m_i{=}0.1,\ell_i{=}0.5\) |
| Insight | All NNMs (modes 1…n) approach **homoclinic orbits through the upright** as \(E\to V_{\max}\); period \(\to\infty\) |
| Controller | Superimpose **eigenmanifold stabilizer** + **energy injection** — walk up one NNM from tiny amplitude to the homoclinic |
| Actuation | Feasible with **weak** actuators — torques capped at **~10% of max gravity torque** in the double demo |
| Modes | Swing-up works via Mode 1 *or* Mode 2 (double shown); pick the mode whose generator matches available actuation |

**Steal for cart-triple swing (when green-lit):** treat cart force as the energy-injection actuator along a soft “modal phase” / height-progress manifold rather than inventing bang-bang gains. Pair with Fattahi/Dyad/EBERL residual modulation (19:00 / 17:33) as the injection law; still **hand off** to LQR/CLF-RL hold once basin∩low-ω (rank #1). Do **not** expect NNM numbers to transfer — measure \(V_{\max}\) / \(E_{\mathrm{UUU}}\) on `physics_triple` first (\(U_{\mathrm{UUU}}\) already ~0.736 J in prior notes).

#### SA-TGaussian (AAAI-25) — concrete alternative to latent-Gaussian + clip/tanh

Our `train/ppo.py` samples unbounded `Normal(mean,std)` on **raw**, logs **latent** entropy `dist.entropy()`, then `tanh_action(raw)*forceLimit`. That is exactly the H(u)/clip geometry arXiv:2608.24488 flagged at 15:42. AAAI-25 gives the **bounded-support** cookbook that keeps Gaussian shape:

| Knob | Steal |
|---|---|
| Location | \(\mu=\frac{u-l}{2}(\tanh g(s)+1)+l\) — keep µ **inside** bounds |
| PDF | Truncated Gaussian \(f(x;\mu,\sigma,l,u)=\phi/\sigma\big/(\Phi_u-\Phi_l)\) |
| Sample | Inverse-CDF via \(\Phi^{-1}\) (no post-clip) |
| Scale-adjust | \(\sigma'=\sigma\cdot d(\mu)\); semi-ellipse \(d\) with **\(k{=}2\)**, **\(d_{\min}{=}0.01\)** (best overall Norm score) |
| \(\sigma_{\mathrm{init}}\) | **0.5** preferred over 1.0 (large init worsens bound saturation) |
| Meter | **aBAR** = fraction of actions in outer **1%** of \([-1,1]\) — Gaussian family often **10–30%**; TGaussian/SA ~**0–2%** |
| Vs cousins | Prefer SA-TGaussian over plain TGaussian (over-avoids bounds) and over Beta/logit-normal on MuJoCo Norm; CAPG helps return but **does not** cut aBAR |

**Steal for us:** optional next explore retune (beside ERA / H(a) / cool-ent): replace latent Normal+tanh with **SA-TGaussian on normalized force** \([-1,1]\) then scale by `forceLimit`. Log **aBAR** + executed entropy. Orthogonal to two-policy; helps both swing and hold nets if bound-bang is farming rail/force.

#### Two-stage reward curriculum (arXiv:2603.05113) — stop stacking behavior terms too early

RC-SAC / RC-TD3: phase-1 train on **task** rewards only; phase-2 unlock **behavior** terms (energy efficiency, smoothness, …). Switch when actor–critic fit stays below a threshold for a window; **recompute** replay rewards so phase-1 samples stay usable. Strongest gains when auxiliary weights are large.

**Steal for our product stack:** treat Lim/Baek **product + progress** as task; defer soft-landing (\(R_\omega\), airo7 \(-\omega^2\)), CLF \(r_{\Delta V}\), cart-centre crank, and saturated-E / Fattahi modulation to a **phase-2** (or hold-net-only) once near_target at_goal/UUU is moving. Matches fawraw “widen catcher before inventing soft-landing coefs” and our live symptom (align↑ / at_goal flat under full product+progress+flip).

#### Still empty / unchanged

- fawraw M4 soft-landing **implementation**: still absent (starter pack from 19:40 stands).
- Serial **cart-triple** classical energy **bang-bang coeffs**: still none (NNM = joint-torque architecture, not cart force gains).
- Force 80–100: still demoted.
- Baek VER: no new multi-link / on-policy paper (shipping flip + optional SA-TGaussian / H(a) are the on-policy levers).
- Rank: after v6/v7 cook, **split swing vs hold** still #1; **NNM energy-walk architecture** joins Fattahi/Dyad/EBERL under P2 swing; **SA-TGaussian \(k{=}2,d_{\min}{=}0.01\)** joins ERA / H(a) / cool-ent under explore; **task→behavior reward stages** join soft-landing / CLF under P1 hold.

#### Amend recommended redesign (additions only)

74. When coding classical energy swing: prefer **NNM-style** manifold stabilize + gradual energy inject (weak actuation OK) over inventing cart bang-bang; measure \(E_{\mathrm{UUU}}\) on our plant; still hand off to hold.
75. Optional PPO action head A/B: **SA-TGaussian** on \([-1,1]\) force with \(k{=}2\), \(d_{\min}{=}0.01\), \(\sigma_{\mathrm{init}}{=}0.5\); track aBAR; prefer over another ENT wipe if bound saturation shows in aBAR/H(u).
76. Reward staging: ship product+progress first; unlock soft-landing / CLF-ΔV / heavy centre / saturated-E on a **second phase** or on the **hold net only** (2603.05113).
77. Rank unchanged: **two-policy swing↔hold** still highest-ROI unimplemented.

**No code this fire** (overnight owns train / B v7b / C→v7; wait for green-light / overnight ask). NEED_USER_PING no.


### Research pass (2026-09-19 ~20:45 CT) — DiffSwing energy→LQR + QIP VER/TQC scale

Digged DiffSwing (yunusemredanabas.com/projects/mujoco_cartpole — Sabancı ME58006; JAX energy-NN + LQR), Oh/Lee/Ryoo/Koh/Han/Lee *Reinforcement Learning to Achieve Real-time Control of a Quadruple Inverted Pendulum* (IJCAS 23:2797–2806, 2025; DOI 10.1007/s12555-025-0235-y), re-checked fawraw (still last *code* **2026-06-25** / docs **2026-07-02** — no soft-landing). Live: VM `cartpole-train-od` RUNNING; overnight owns train (do not mid-kill). **Direction unchanged.** Top *unimplemented* lever remains **two-policy swing↔hold**. No user ping (fills energy-swing + off-policy scale cookbooks under existing P1/P2; does not displace handoff as #1).

#### DiffSwing — closest *same-mass-scale* neural energy → LQR handoff

Plant matches our link/cart masses almost exactly (single pole only — retune \(E^\star\) for triple):

| Knob | Steal |
|---|---|
| Plant | \(M{=}1.0\,\mathrm{kg}\), \(m{=}0.1\,\mathrm{kg}\), \(\ell{=}0.5\,\mathrm{m}\) (half-length) — same \(m_c,m_i,\ell_i\) class as `constants-triple.json` |
| \(E_{\mathrm{target}}\) | **\(2mgl\)** (upright−hang ΔU for single COM at \(\ell\)) → for us use measured \(E_{\mathrm{UUU}}\) / \(U_{\mathrm{UUU}}{\approx}0.736\,\mathrm{J}\), not \(2mgl\) blindly |
| Swing objective | \(J=\sum_t\bigl[w_E(E-E^\star)^2 + w_x x^2 + w_u u^2\bigr]\) |
| Weight schedule | **Energy 1.0 → Position 0.1 → Control 0.01** (task energy first; behaviour later — same spirit as 2603.05113 staging @20:15) |
| Net | MLP **2×64**, tanh; Adam **1e-3**; batch **256** ICs; ~5k steps (diffrax / analytical grads — not PPO; architecture still stealable) |
| Obs | \([x,\cos\theta,\sin\theta,\dot x,\dot\theta]\) |
| Handoff | neural energy pump until **\(\|\theta\|<12^\circ\approx0.209\,\mathrm{rad}\)** → LQR |
| Peak force | ~**12 N** swing / ~6 N LQR hold — reconfirms forceLimit 40–50 ample; demote 80–100 |
| Result | 98% success over \([-\pi,\pi]\times\pm2\,\mathrm{rad/s}\); ~1.9 s mean swing-up |

**Steal for cart-triple (when green-lit):**

1. P2 swing reward / residual: prefer DiffSwing-style **\(w_E(E-E_{\mathrm{UUU}})^2\)** (+ light \(x,u\)) over height-proxy `energy_w`; still prefer Fattahi *modulated action* (19:00) if coding structured \(u\), DiffSwing loss if coding energy *reward* only.
2. Weight order: keep product+progress (task) dominant; unlock centre / soft-landing / ctrl crank only after near_target at_goal moves — DiffSwing schedule is the single-pole numeric twin of 20:15 task→behavior.
3. Handoff angle: DiffSwing **12°** is a *single-pole LQR RoA* cue. For triple, **do not** replace measured basin ≤0.1 rad / low-ω (fawraw + ResearchSquare ±15° LQR limit @19:40) with 12° alone — treat 12° / Mon \(\bar c>0.9\) as optional *delivery* meters AND'd with basin∩energy∩low-ω.
4. Architecture confirmation: neural (or PPO) **energy swing** + **classical LQR hold** = same rank-#1 two-policy pattern; DiffSwing is the cleanest published single-pole cookbook with our mass numbers.

#### Oh et al. QIP (IJCAS 2025) — VER+TQC scales past TIP

Same Inha/POSTECH lineage as Baek TIP (EAAI 2024). First model-free **quadruple** cart inverted-pendulum swing-up+balance on hardware via **TQC + VER** (geometric left↔right mirror into replay). Paywalled — **no open numeric reward / hypers cookbook** this pass (Springer / OASIS abstract only).

**Steal (qualitative only until PDF unlocks):**

1. Strengthens P2 off-policy: hardware winners at TIP *and* QIP used **TQC+VER**, not PPO — keep PPO overnight, but a one-slot TQC(+flip) A/B remains justified when green-lit.
2. VER is the sample-efficiency lever that survived 3→4 links; our shipping `--flip-augment` is the on-policy cousin — no new multi-link on-policy VER paper yet.
3. Do **not** chase QIP plant / 4-link reward until UUU hold works.

#### Still empty / unchanged

- fawraw M4 soft-landing **implementation**: still absent (19:40 starter pack stands).
- Serial **cart-triple** classical energy **bang-bang coeffs**: still none (DiffSwing = single-pole neural energy, not triple bang-bang).
- Force 80–100: still demoted (DiffSwing peak ~12 N on our mass scale).
- Baek VER: no new multi-link *on-policy* paper (QIP = off-policy VER confirmation only).
- Rank: after v6/v7 cook, **split swing vs hold** still #1; **DiffSwing \(2mgl\) / 12° / \(w_E{:}w_x{:}w_u{=}1{:}0.1{:}0.01\)** joins Fattahi/Dyad/NNM under P2 swing + LQR handoff under P1; **QIP TQC+VER** joins Lim/Baek under P2 off-policy scale.

#### Amend recommended redesign (additions only)

78. When coding P2 energy *reward* (not just action modulation): start from DiffSwing \(w_E(E-E_{\mathrm{UUU}})^2 + 0.1\,x^2 + 0.01\,u^2\) with \(E_{\mathrm{UUU}}\) measured on `physics_triple`; keep product+progress as outer task or stage DiffSwing weights after product moves.
79. Optional handoff delivery cue: \(\|\phi\|_\infty < 12^\circ\) (DiffSwing) AND'd with measured basin ≤0.1 rad + low-ω + optional E-gate — never 12° alone on triple.
80. P2 off-policy side slot: prefer **TQC + VER/flip** (Baek TIP → Oh QIP lineage) over inventing a new algo; still second to two-policy under PPO.
81. Rank unchanged: **two-policy swing↔hold** still highest-ROI unimplemented.

**No code this fire** (overnight owns train; wait for green-light / overnight ask). NEED_USER_PING no.

## Fresh approach audit (2026-09-19 ~21:30 CT)

**User ask:** PPO UUU stuck (~0.04–0.06 near_target); want a **non-PPO** path grounded in what actually worked for others. Live A/B/C PPO left running.

### Who succeeded (primary sources)

| Who | Algo | Reward / curriculum | Force / budget | Outcome |
|---|---|---|---|---|
| **Lim / Ju / Lee KIEE 2025** ([PDF](http://ecsl.inha.ac.kr/publication/KIEE2025_b.pdf), [YouTube](https://youtu.be/vVx3ffGo2mk)) | **TQC** (Kuznetsov); **8 separate policies** (one per EP) — **not** UVFA | Product of [0,1] terms \(R_u R_y R_{\theta1..3} R_{\dot\theta1..3}\); wide random ICs; train until return ~700–800/1000 | Cart accel early-stop 2.5 m/s², \|y\|≤0.48 m; lr 3e-4, γ 0.99, τ 0.005, buffer 1e6, 3 critics, 25 atoms, policy 400→300, critic 3×512, batch 256 | **All 56 transitions on hardware** (sim→real) |
| **Baek et al. EAAI 2024** | Off-policy actor-critic (**SAC-class**) + **VER** (left↔right mirror) | Product with α floors; dense height/center/vel; **trained on hardware** | Their plant ~±10 N | **DDD→UUU swing-up on real TIP** (1 EP, not 56) |
| **fawraw/triple-pendulum-sim2real** | **TQC** (sb3-contrib); M2 UUU hold → M3 8 EPs → M4 handoff | Additive + barrier + progress; M2 `near_target` ~150k; M3 hard-EP weight schedule | MuJoCo; A5000 ~$0.27/hr; M3 ~1.1M steps | **8 EPs 72.5% in sim**; 56 still blocked on catch basin → building **two-stage handoff** |
| **Glück Automatica 2013** | Classical: nonlinear feedforward (BVP) + time-varying Riccati | Precomputed swing trajectory | Accel ≤~22 m/s² | Experimental **DDD→UUU** (not RL, not 56) |
| **Graichen / Spong / DiffSwing** | Energy swing → **LQR** handoff | Energy error + local RoA switch (~0.1–0.2 rad) | DiffSwing peak ~12 N on m_c=1, m=0.1, ℓ=0.5 (our mass scale) | Single/double proven; serial **cart-triple** energy coeffs still unpublished |

### Why single-policy PPO fails UUU on 3-link

1. **Mode conflict:** swing needs bang-bang energy injection; hold needs small precise forces — one on-policy Gaussian collapses to neither.
2. **Credit assignment:** rare UUU captures are washed out by on-policy batch noise; off-policy (TQC/SAC) replays them.
3. **Entropy collapse:** our overnight slots repeatedly hit negative / saturated entropy before `near_target/at_goal/UUU` rises (see 21:10 CT table).
4. **Literature mismatch:** every hardware 56 / TIP swing success used **TQC or SAC**, not PPO; Lim explicitly chose TQC for high-variance tail rewards.

### Ranked fresh approaches for OUR repo

| Rank | Approach | Effort (1–2 days) | Expected upside | Notes |
|---|---|---|---|---|
| **1** | **TQC UUU specialist** (Lim EP7) via sb3-contrib + Gym wrapper | Low–med | **Highest** — exact algo that got 56 | Product reward already in `goals_triple`; forceLimit 40 OK |
| 2 | Two-policy handoff (TQC/energy swing → TQC or LQR hold) | Med | High once hold exists | fawraw M4 + DiffSwing/Spong template |
| 3 | Lim **8×TQC** (all EPs) after UUU holds | Med–high compute | Required for 56 | Specialists, not UVFA |
| 4 | Energy E→E_UUU + LQR catch | Med | Good classical fallback | Need measured E_UUU + linearize plant |
| 5 | Keep PPO cool-ent / ent-boost | Low | **Low** — already flat ~0.04–0.06 | Do not expand; leave A/B cooking until C swap |

### Top pick (next 1–2 days) — LOCKED

**Ship TQC UUU specialist (Lim path), then swap L4 slot C.**

Implemented on box (2026-09-19):
- `train/envs/triple_gym.py` — Gymnasium wrapper, 11-D obs, product reward, near_target+wide+hang ICs
- `train/train_triple_tqc.py` — sb3-contrib TQC with Lim Table-1 hypers
- `scripts/next-train-triple-tqc-uuu.sh` / `continue-triple-tqc-uuu.sh`
- `.venv-tqc` with torch+gymnasium+sb3+sb3-contrib
- **Smoke:** 2500 steps CPU, `n_updates>0`, ckpt written — PASSED

### How to launch (box smoke / VM slot C later)

```bash
# Box smoke
cd /workspace/double-cart-pole
SMOKE=1 bash scripts/next-train-triple-tqc-uuu.sh

# Full UUU specialist (VM when swapping slot C — do not kill A/B)
FORCE_LIMIT=40 TOTAL_STEPS=300000 \
  INIT_MODE=near_target HANG_FRAC=0.05 WIDE_FRAC=0.25 \
  bash scripts/next-train-triple-tqc-uuu.sh
```

### Citations (primary)

- Lim, Ju, Lee — KIEE 74(8):1363–1372, 2025 — TQC, 8 policies, product reward, all 56
- Baek et al. — EAAI 128:107518, 2024 — SAC+VER, product+floors, hardware UUU
- Kuznetsov et al. — TQC, arXiv:2005.04269
- Glück, Eder, Kugi — Automatica 2013 — feedforward+TV-Riccati DDD→UUU
- fawraw/triple-pendulum-sim2real — open TQC MuJoCo attempt; M3 72.5%, M4 handoff
- DiffSwing / Spong — energy→LQR handoff pattern (single-pole mass scale matches ours)


### Research pass (2026-09-19 ~21:31 CT) — BaRC reverse curriculum + hold-reward integral + handoff defaults

Digged StanfordASL BaRC (arXiv:1806.06161; Florensa reverse-curriculum cousin), re-read Fu/Guo/Li et al. *Robotica* 44(2):698–722 (2026) CSAC-QI abstract, and re-fetched live `fawraw/triple-pendulum-sim2real` `sim/handoff.py` + `docs/m4_findings.md` (still last *code* **2026-06-25**). Macro loop (21:30 CT) already pivoted **Next micro-task → TQC UUU specialist (Lim)** on slot C — overnight owns scp/swap; **do not mid-kill A/B PPO**. **Direction unchanged** (P1 UUU hold). This pass fills **TQC hold-net curriculum + steady-state reward** under that live task; does **not** displace TQC as #1 or rewrite Next micro-task. No user ping.

#### BaRC (arXiv:1806.06161) — formal reverse curriculum for sparse-goal hold

Missing from prior notes. Model-free wrapper that **starts ρ₀ inside the goal basin** and expands the initial-state set via approximate **backward reachable sets** once mastery clears a threshold — exactly the hold-specialist path fawraw M4 wants (“widen catcher before soft-landing”), with dynamics-aware frontiers instead of isotropic `init_noise` alone.

| Knob | Steal |
|---|---|
| Start | Sample ICs from a tiny set around UUU (goal / near_target) |
| Expand | Once success rate ≥ \(C_{\mathrm{pass}}\) (paper default **0.5**), grow ρ₀ by short-horizon BRS (paper \(T{=}0.1\,\mathrm{s}\)) or a cheap proxy |
| Mix | Keep \(N_{\mathrm{old}}\) mastered starts + \(N_{\mathrm{new}}\) frontier (paper **100 / 200**) to avoid forgetting |
| Mastery select | Keep starts with success ≥ \(C_{\mathrm{select}}\) (**0.5**) |
| PPO inner | Any model-free algo; paper used PPO — fits our stack |
| Vs isotropic noise | BRS expands in **dynamically feasible** directions (angles + ω + \(x\)); random-action reverse curricula (Florensa CoRL'17) break on unstable underactuated plants |

**Practical proxy without HJ PDE (overnight-feasible):** stage hold-net `init_noise` / nonzero ω / off-centre \(x\) in **ladder steps** (e.g. 0.05 → 0.10 → 0.15 → 0.20 rad) only after `eval/near_target/at_goal/UUU` clears a gate on the current rung — same spirit as BaRC \(C_{\mathrm{pass}}\), matches fawraw catch-basin table (reliable only ≤**0.1 rad** + near-zero vel). Do **not** jump straight to hang_start on the hold net.

#### CSAC-QI (Fu et al., Robotica 2026) — hold steady-state term

Already noted at a high level; this pass locks the **steal for the hold half**:

1. Reward = **quadratic** angle/state costs + **integral of cumulative joint-angle error** (QI) — cuts steady-state bias that pure product/progress leaves on the table.
2. **Adaptive curriculum** easy→hard initial joint deviations (same direction as BaRC / fawraw widen-basin).
3. Plant is **UTPR** (passive-first joint, not cart-actuated) — retune; steal structure only. No open numeric coefs this pass (Cambridge Core paywall).

**Steal when coding hold net:** add a small running \(\int e_\theta\,dt\) (or discrete sum of link angle errors vs UUU) gated after product is already high near target; pair with Baek \(e(\dot\theta)\) soft-landing. Defer to **hold-net-only / phase-2** per 2603.05113 staging (20:15) — do not stack on live single-policy A/B/C.

#### fawraw handoff defaults (re-confirmed live)

`HandoffController(..., capture_tol_rad=0.3, capture_vel_rad_s=None, latch=True)` — **default 0.3 rad is unsafe**. M4 basin table still: success only at **~0.1 rad / ~0 ω**. Soft-landing implementation still **absent** upstream. When overnight codes handoff: ship **tol ≤ measured basin (≤0.1)**, optional vel gate, **latch=True**, never the 0.3 default (already in amend #38 / #52; restate because it is the #1 footgun).

#### Still empty / unchanged

- fawraw M4 soft-landing **implementation**: still absent (19:40 starter pack stands).
- Serial **cart-triple** classical energy bang-bang coeffs: still none.
- Force 80–100: still demoted.
- Baek VER: no new multi-link on-policy paper.
- Rank: live #1 is **TQC UUU specialist (Lim)** (macro 21:30); **BaRC / ladder init expand** + **CSAC-QI ∫θ** are cookbooks *inside* that TQC hold stretch. Two-policy handoff stays #2 after UUU holds. Cool-ent stays A/B-only until C swap.

#### Amend recommended redesign (additions only)

82. Hold-net curriculum: BaRC-style **near-UUU → expand** (ladder `init_noise`/ω/\(x\) with mastery gates) before hang mixture; prefer over a single hot near_target distribution.
83. Hold-net reward (phase-2 / hold only): optional **∫ cumulative link-angle error** (CSAC-QI) on top of product + Baek rate floors — after near_target at_goal is moving.
84. Handoff footgun reminder: `capture_tol_rad` ≤ **measured** basin (≤0.1), never fawraw default 0.3; latch on; vel gate recommended.
85. Rank: live Next micro-task (**TQC UUU on C**) still correct — do not preempt. BaRC ladder + CSAC-QI ∫θ are **implementation details for the TQC hold recipe**, not a competing micro-task. Two-policy remains #2 after hold works.

**No code this fire** (overnight owns train / slots). NEED_USER_PING no.


## Walls-first UUU curriculum (2026-09-20 ~00:45 CT)

Patrick: no-walls triple demo looks like center/void farming, not upright. Adopt double’s path — **P1a hard inelastic track walls** (cart clamp + \(\dot x=0\), no pole impulse), then **P1b void FT**. Flag: `trackWalls` / `--track-walls`. See macro loop P1a/P1b and paper-training-lessons (ap).
