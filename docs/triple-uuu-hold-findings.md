# UUU hold findings (2026-09-21)

Training on homemade W2 knobs is paused. Next work copies a published recipe. It is not a new method.

Operational plan stays in `docs/triple-training-wheels.md`. Method notes stay in `docs/working-impls-reverse-eng.md`. Old PPO/ATRPO fire logs stay in `docs/triple-pendulum-research.md`.

## Honest verdict

We said we were copying Lim / fawraw / Glück. The **stack** was copied (TQC, continuous force, quiet basin, product reward). The **W2 campaign** was not. Mix ICs, VER, hard-IC oversample, looser BC-reg, and 0.03/0.035/0.04 LQR-cloning are our knobs. They did not widen the basin. They proved TQC cloned linear LQR.

W1 quiet hold is real. W2 at **0.03** is real (0.90). Past that we were inventing. Stop.

## What already works elsewhere (copy these, do not remix)

| Source | Code? | What it actually did | Honest status |
|---|---|---|---|
| **Lim / Ju / Lee KIEE 2025** | No public repo | TQC, **8 EP specialists**, product reward, 56 transitions on **hardware** | Gold standard. Paper only: http://ecsl.inha.ac.kr/publication/KIEE2025_b.pdf |
| **fawraw/triple-pendulum-sim2real** | **Yes** | Their YAML + trainer on **their** MuJoCo plant | Best thing we can run unmodified. M2 UUU still marked partial; M3 8-EP ~72.5% sim; M4 56 **not done** |
| **Glück Automatica 2013** | Classical write-up | Energy / BVP feedforward + local Riccati, then **switch** | Experimental DDD→UUU. Not RL. Proves two-DOF, not “widen LQR to 0.05” |
| **Baek EAAI 2024** | No official repo | SAC + left-right VER, hardware swing-up to **one** top | Validates off-policy + symmetry. Not a 56-policy |

fawraw M2, as published (`training/configs/m2_upright_tqc.yaml`):

- Task: stabilize EP7 UUU. Start already near upright. Policy rejects perturbations.
- `init_mode: near_target`, `init_noise: 0.05`
- TQC, lr `3e-4`, buffer 200k, batch **256**, net `[128,128]`, 3 critics, 20 quantiles, drop 2
- 150k steps, 1000-step episodes
- Gate: `ep7_success_rate ≥ 0.80`, then they go to **M3** (all 8 EPs), not a homemade 0.04 clone ladder
- Their README does **not** list LQR-BC, actor freeze, BC-reg 2.5, VER, or hard-IC banks for M2

Their next published files: `m3_all_eps_tqc.yaml`, `m3b_all_eps_tqc.yaml`, then M4 two-stage handoff. M4 note (`docs/m4_findings.md`): catcher on **their** plant is ~**0.1 rad / near-zero rate**. Swing delivers ~0.2 rad with rate and drops. They also hit a small basin. They did not solve it by inventing our W2 knobs.

## What we have that is real (our plant)

Keepers. Do not overwrite.

- Demo: `web/public/policy-triple.json` from n004-lqrbc
- W1 TQC: `policies/tqc-m2-uuu-hold-mac.zip` — survival 1.0 at noise 0.02
- W2 lock: `policies/tqc-m2-uuu-hold-w2-n004-lqrbc.zip` — **0.90 / 0.90 / 0.99 at 0.03**; 1.0 at 0.02
- LQR/BC oracles PASS at 0.01 and 0.02 (`policies/bc-lqr-uuu.pt`)
- Env contract is right: rates fixed ±0.01, fall-kill 0.6, `progress_w=0` (`train/envs/triple_gym.py`)

Plant is holdable. The old “impossible hold” was the IC/recipe, not broken dynamics.

## Scoreboard (official N=50, quiet rates ±0.01)

Survival / at_goal. Align was ~0.98–1.0 whenever survival was high.

| Controller | 0.02 | 0.03 | 0.035 | 0.04 | 0.05 |
|---|---|---|---|---|---|
| LQR | 1.00 | ~0.86–0.90 | **0.80** | **0.72** | 0.70 |
| BC from that LQR | 1.00 | 0.84 | 0.78 | 0.62 | 0.68 |
| TQC n004-lqrbc (keeper) | 1.00 | **0.90** | 0.74 | **0.72** | — |
| TQC W1 mac zip | 1.00 | 0.80 | — | 0.62 | — |

At 0.04, LQR and the TQC keeper die on the **same 14 ICs**, same lengths (both 0.72). TQC cloned the linear region of attraction. The cliff is 0.03 → 0.035, not a missing critic trick.

## What we invented (do not repeat)

These were not in fawraw’s M2 YAML. They did not beat the LQR ceiling.

| Our knob | Result |
|---|---|
| Mix ICs `[0.02, 0.04]` | 0.72 then 0.64 at 0.04 |
| Baek VER replay flip | Same 0.72 at 0.04 |
| Oversample 14 death ICs | 0.68 at 0.04; **0.03 slipped 0.90 → 0.74** |
| BC-reg 1.0 (leave the LQR clone) | Still 0.72 at 0.04; 0.03 slipped to 0.80 |
| LQR-BC + freeze 20k + BC-reg 2.5 | **This is how W1/0.03 passed.** It does not create a wider basin. |
| Unfreeze with no BC-reg | Survival 1.0 → 0 in ~4k steps |
| Train at 0.05 because fawraw’s YAML says 0.05 | Overnight hang; official 0.48. 0.05 is above **our** LQR (0.70) |
| 0.035 LQR dump + BC | Dump 138/160; BC 0.78. Job killed before another TQC clone. |

Retired algorithms (older campaign): PPO, ATRPO, HER, UVFA, bang-bang. Do not revive.

## Why fawraw’s `init_noise: 0.05` is not our next experiment

Their 0.05 is a **near-upright spawn on their MuJoCo plant**, not a certified RoA we must match here. On this plant, linear LQR already fails the W2 bar at 0.04 (0.72) and only just clears it at 0.035 (0.80). Copying the **number** without their plant, masses, and force scale is how we started inventing.

If we want their 0.05 result, run **their** trainer on **their** env.

## Fidelity rule

Every train job must cite a **file or paper section**. If the knob is not in that source, we do not add it.

**Allowed**
- fawraw `training/configs/m2_upright_tqc.yaml` + `training/train_m2_upright.py` (hold)
- Lim KIEE 2025: one TQC specialist per equilibrium (not 56 nets, not one UVFA)
- Glück 2-DOF and fawraw `docs/m4_findings.md` (swing then latch; later)

**Already-justified plant tweak (keep, do not grow)**
- LQR-BC + actor freeze 20k + BC-reg 2.5 on **our** plant only. Vanilla fawraw M2 has none of that. We needed it because unfreeze without the anchor died in ~4k steps.

**Forbidden**
- W2 0.035 / 0.04 / 0.05 cloning, mix ICs, VER, hard-IC, looser BC-reg
- fawraw M3 one-net + one-hot + `target_mode: random` (UVFA family; we retired it; they reached ~72.5% sim and M4 is still open)
- PPO / ATRPO / HER / UVFA / bang-bang
- GCP unless asked

## Recommended next (copy, in parallel)

**Pick:** fawraw M2 hold recipe × Lim 8 specialists. Do not port fawraw M3.

**Track A — vendor reference.** Clone https://github.com/fawraw/triple-pendulum-sim2real to `vendor/fawraw-triple-pendulum-sim2real`. Do not merge their tree into `train/`. Run their command as published:

```text
python -m training.train_m2_upright --config training/configs/m2_upright_tqc.yaml
```

Success = their meter on their plant (`ep7_success_rate ≥ 0.80`). This is a reference, not the demo.

**Track A result (2026-09-21).** Unmodified `python -m training.train_m2_upright --config training/configs/m2_upright_tqc.yaml` finished 150k (exit 0, 2365s). Run `58faaec73a13444ebf4fa6e6e725d8f3`. Trainer `final_eval_reward_mean=-241.20`. Their YAML does not log `ep7_success_rate`; same length≥800 rule on `checkpoints/m2_upright_20260921_140207/final.zip` (EP7, N=20, `init_noise=0.05`) gave **ep7_success_rate=0.50**. Gate 0.80 **FAIL**. Last periodic eval (140k, N=5) mean length 723. Do not treat this as a reason to invent W2 knobs.

**Track B — our demo plant (W6 specialist #2).** Freeze n004-lqrbc as the UUU hold specialist at **0.03** (zip stays). Live demo is the DDD best actor, not that UUU export. Next copied job: **DDD hold**, same W1 stay recipe, change **only** the target angles. Gate: survival ≥ 0.80, at_goal ≥ 0.80, align ≥ 0.90 at noise 0.03, N=50. Then stop. Do not start UDD/DUU/… in the same breath.

**Track B result (2026-09-21).** DDD best checkpoint **PASS**: N=50, noise 0.03, survival=1.000, at_goal=1.000, align=0.999, mean_len=1000. UDD best checkpoint (`policies/tqc-m2-UDD-hold-best-100k.zip`, 100k eval 10/10) **PASS**: N=50, noise 0.03, survival=1.000, at_goal=1.000, align=0.998, mean_len=1000. Demo loads DDD, UDD, and the UUU keeper by goal. DUD vanilla 150k **FAIL** (best N=50 survival=0.720, at_goal=0.440). LQR at DUD then BC **PASS** (both 1.000 / 1.000 at noise 0.03). TQC fine-tune from that anchor (`policies/tqc-m2-DUD-hold-lqrbc-best.zip`) **W1 GATE PASS**: N=50, noise 0.03, survival=1.000, at_goal=1.000, align=1.000, mean_len=1000. Demo loads that actor for DUD. The fine-tune was stopped at the 75k checkpoint (training eval still 100% at 70k). The official pass stays `policies/tqc-m2-DUD-hold-lqrbc-best.zip`. LQR then BC at noise 0.03, N=50, survival=1.000 at_goal=1.000: UUD (`policies/bc-lqr-uud.pt`), UDU (`policies/bc-lqr-udu.pt`), DDU (`policies/bc-lqr-ddu.pt`). DUU LQR PASS, then BC PASS (`policies/bc-lqr-duu.pt`, N=50 noise 0.03 survival=0.980 at_goal=0.980). UUD TQC best (`policies/tqc-m2-UUD-hold-lqrbc-best.zip`) **W1 GATE PASS**: N=50, noise 0.03, survival=1.000, at_goal=1.000, align=1.000, mean_len=1000. Demo loads that actor for UUD. The UUD fine-tune was stopped after the official pass. UDU TQC best (`policies/tqc-m2-UDU-hold-lqrbc-best.zip`) **W1 GATE PASS**: same meters, 1.000 / 1.000 / 1.000. Demo loads that actor for UDU. The UDU fine-tune was stopped after the official pass (last flushed log 32k). DDU TQC best (`policies/tqc-m2-DDU-hold-lqrbc-best.zip`) **W1 GATE PASS**: N=50, noise 0.03, survival=1.000, at_goal=1.000, align=1.000, mean_len=1000. Demo loads that actor for DDU. The DDU fine-tune was stopped after the official pass (training eval still 100% at 70k). Next TQC is **DUU**, the last hold. Do not start a second MPS train. DUU TQC best (`policies/tqc-m2-DUU-hold-lqrbc-best.zip`) **W1 GATE PASS**: N=50, noise 0.03, survival=1.000, at_goal=1.000, align=1.000, mean_len=1000. Demo loads that actor for DUU. All 8 quiet holds now pass the same gate. The DUU fine-tune was stopped after the official pass.

Product-reward UUU swing from the bottom never entered the catch box (enter-rate 0.000 through 75k; closest 0.111 rad at 25k, then 1.87 rad). That job is stopped. The replacement uses the published M4 swing cost (weighted error², vel 0.02, cart 0.2, barrier 50, bonus 200 after 100 steps inside 0.3 rad). Holds stay on the product. Enter-gate stays `|θ|≤0.03`, `|ω|≤0.01`. Glue stub: `train/handoff.py`.

**Do not** design a third “do everything” net. **Do not** start GCP unless asked.

## Locked stack (still copied; keep)

TQC. Continuous cart force. Product reward in `[0, 1]` for holds. Published M4 quadratic cost for the swing net only. One specialist per equilibrium. LQR only as a **local** oracle / catcher. Walls on until hold works.

## Resume rule

W2 homemade knobs stay dead. Resume means the next Lim specialist TQC (UUD, then UDU, DDU, DUU), same M2 hold recipe. UUD, UDU, and DDU already have a passing LQR-BC anchor, so those jobs use freeze 20k and BC-reg 2.5.
