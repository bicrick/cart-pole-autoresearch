# double-cart-pole

Browser cart + double pendulum. A trained **goal-conditioned** policy runs **on the page** (phone or laptop). You grab the cart or either link and shove it. The sim does not reset.

Pick one of the four plant equilibria; the same MLP drives toward that goal while you poke it:

| id | configuration |
| --- | --- |
| `UU` | both links upright (`θ1=0`, `θ2=0`) |
| `UD` | lower up, upper hanging (`θ1=0`, `θ2=π`) |
| `DU` | lower hanging, upper up (`θ1=π`, `θ2=0`) |
| `DD` | both hanging (`θ1=π`, `θ2=π`) |

Convention: `θ = 0` is upright (Xin / IFAC 2008 plant). Cart `x` near the origin is a soft shaping term, not a fifth discrete goal.

A later grok-bot loop is meant to own training, checkpoints, and policy swaps. This repo is the gym, the browser, and the paper shelf that loop should read.

## Layout

```
shared/constants.json   # masses, lengths, dt, force clip, obs bounds (shared truth)
train/physics.py        # batched PyTorch step (GPU worlds)
train/goals.py          # UU/UD/DU/DD encodings, goal reward, nearest-goal / HER helpers
web/src/physics.js      # same equations in the browser
web/src/goals.js        # same discrete goals + encoding for the picker
train/ppo.py            # actor-critic + JSON export
train/train.py          # goal-conditioned PPO (`--smoke` locally)
train/gcp/              # create / sync / train / pull / teardown a GCE GPU VM
web/                    # Vite + Canvas demo + goal picker
policies/policy.json    # exported weights the page loads (plain JS MLP)
papers/                 # PDFs the trainer / bot should cite
```

## Papers

PDFs live in [`papers/`](papers/). Index and “why we have this” notes: [`papers/CITATIONS.txt`](papers/CITATIONS.txt).

Shipped architecture:

1. **Goal-conditioned policy** (UVFA): observe state **and** goal. Discrete goal id (one-hot) plus target `sin/cos` of the two link angles. One net; the picker just changes `g`.
2. **HER** on failed episodes: relabel with the equilibrium actually nearest at episode end so misses teach other goals (`papers/CITATIONS.txt` → HER / UVFA).
3. Dense shaping as auxiliary: cos-alignment to target angles, soft `x` penalty, action cost, plus a sparse at-goal bonus.
4. Inference stays a tiny MLP in plain JS. No ONNX. No TF.js. No server.

Specialist weights per goal are a fallback if one net will not cover all four. Prefer one conditioned policy so the picker is a UI, not a model zoo.

## Observation layout

`obs_dim = 16`:

```
[x, xd, sinθ1, cosθ1, sinθ2, cosθ2, θ1d, θ2d,
 onehot_UU, onehot_UD, onehot_DU, onehot_DD,
 sinθ1*, cosθ1*, sinθ2*, cosθ2*]
```

Exported in `policy.json` as `obs_layout`, `goals`, and `layers` (same linear/tanh format the web loader already understands).

## Run the demo

```bash
cd web
npm install
npm run dev
```

Open the printed localhost URL. Use the **goal** buttons (`UU` / `UD` / `DU` / `DD`). Drag the cart or either pole. There is no reset control.

A smoke-exported `web/public/policy.json` ships so the page loads even before a long GPU train. Real multi-goal behavior needs a full run (below).

## Train

Local smoke (CPU / MPS) — writes a loadable goal-conditioned `policy.json`:

```bash
python3 -m pip install -r requirements.txt
python3 train/train.py --smoke
```

That writes `policies/policy.json` and `web/public/policy.json`. Event files go under `runs/`.

Watch a run:

```bash
tensorboard --logdir runs --port 6006
```

Scalars: `train/rollout_reward`, `train/policy_loss`, `train/value_loss`, `train/entropy`, `train/goal_frac/{UU,UD,DU,DD}`, `eval/reward`, `eval/align`, `eval/upright` (UU align), plus per-goal `eval/reward|align|at_goal/{goal}`.

Real run: batched worlds on a GPU, not SB3 over one numpy env.

```bash
python3 train/train.py --num-envs 4096 --updates 400 --logdir runs
```

Optional: `--her-ratio 0.8` (default) controls how often failed episodes are relabeled with the nearest achieved equilibrium.

GCP scripts (personal account, project `cartpole-demo`, sibling to `qwop-wr`):

```bash
./train/gcp/create.sh
./train/gcp/sync.sh
./train/gcp/train.sh
./train/gcp/status.sh
./train/gcp/pull.sh
./train/gcp/teardown.sh    # delete the training VM when done
```

Do not train on `pumpkin-minecraft-server`.

### Bot service account

Persistent SA on `cartpole-demo` (not the Minecraft project):

- Email: `cartpole-bot@cartpole-demo.iam.gserviceaccount.com`
- Roles: Compute Admin, Storage Admin, Service Account User, Logging Admin, Monitoring Admin, Service Usage Admin, Secret Manager Admin, OS Admin Login, IAP tunnel
- Can attach the Compute default SA to VMs it creates
- OS Login is on for the project; IAP SSH firewall `allow-iap-ssh` (tcp:22 from `35.235.240.0/20`)
- Bucket: `gs://cartpole-demo-413636930404` (`configs/`, `checkpoints/`, `policies/`)
- JSON key (local only, not in git): `~/.config/gcloud/cartpole-bot-cartpole-demo.json`
- GPU: `GPUS_ALL_REGIONS` is 1 (T4 can launch). Request in flight to 4 (`cartpole-gpus-all-regions-1`).

Grok-bot should set `GOOGLE_APPLICATION_CREDENTIALS` to that key path. See [`train/gcp/bot.env.example`](train/gcp/bot.env.example).

## Physics convention

State is `[x, xdot, theta1, theta1dot, theta2, theta2dot]`. `theta = 0` is **upright**. Episodes do not die when a pole falls. Training may time-cap; the browser never resets. `train/physics.py` and `web/src/physics.js` stay bit-for-bit equivalent via `shared/constants.json`.

## License / papers

Code is for this project. The PDFs in `papers/` remain under their original publishers’ terms (ICML / NeurIPS / IEEE / IFAC / arXiv author versions).
