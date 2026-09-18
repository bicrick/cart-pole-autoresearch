# double-cart-pole

Browser cart + double pendulum. A trained policy runs **on the page** (phone or laptop). You grab the cart or either link and shove it. The sim does not reset.

The intended product is not “always balance both upright.” It is a **goal picker** over the discrete equilibria of the plant, plus the same physical poking:

| id | configuration |
| --- | --- |
| `UU` | both links upright |
| `UD` | lower up, upper hanging |
| `DU` | lower hanging, upper up |
| `DD` | both hanging |

Pick a goal. The policy should drive there from wherever you left it, and keep fighting while you mess with it.

A later grok-bot loop is meant to own training, checkpoints, and policy swaps. This repo is the gym, the browser, and the paper shelf that loop should read.

## Layout

```
shared/constants.json   # masses, lengths, dt, force clip (shared truth)
train/physics.py        # batched PyTorch step (GPU worlds)
web/src/physics.js      # same equations in the browser
train/ppo.py            # actor-critic + JSON export
train/train.py          # PPO trainer (`--smoke` locally)
train/gcp/              # create / sync / train / pull / teardown a GCE GPU VM
web/                    # Vite + Canvas demo
policies/policy.json    # exported weights the page loads
papers/                 # PDFs the trainer / bot should cite
```

## Papers

PDFs live in [`papers/`](papers/). Index and “why we have this” notes: [`papers/CITATIONS.txt`](papers/CITATIONS.txt).

Architecture we should train toward (not all shipped yet):

1. **Goal-conditioned policy** (UVFA): observe state **and** goal. Discrete goal id (one-hot) plus target `sin/cos` of the two link angles. One net, picker just changes `g`.
2. **HER** on sparse “are we in the requested equilibrium?” so misses become data for the config we actually hit.
3. **Energy / hybrid priors** (Spong, IFAC 2008 DIP analysis) as reward shaping or a capture region, not as the thing the user plays.
4. Keep inference a tiny MLP in plain JS. No ONNX. No server.

Specialist weights per goal are a fallback if one net will not cover all four. Prefer one conditioned policy first so the picker is a UI, not a model zoo.

## Run the demo

```bash
cd web
npm install
npm run dev
```

Open the printed localhost URL. Drag the cart or either pole. There is no reset control.

## Train

Local smoke (CPU / MPS):

```bash
python3 -m pip install -r requirements.txt
python3 train/train.py --smoke
```

That writes `policies/policy.json` and `web/public/policy.json`. Event files go under `runs/`.

Watch a run:

```bash
tensorboard --logdir runs --port 6006
```

Scalars: `train/rollout_reward`, `train/policy_loss`, `train/value_loss`, `train/entropy`, `eval/reward`, `eval/upright`.

Real run: batched worlds on a GPU, not SB3 over one numpy env.

```bash
python3 train/train.py --num-envs 4096 --updates 400 --logdir runs
```

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

## Physics convention

State is `[x, xdot, theta1, theta1dot, theta2, theta2dot]`. `theta = 0` is **upright**. Observation is `[x, xdot, sin θ1, cos θ1, sin θ2, cos θ2, θ1dot, θ2dot]`. Episodes do not die when a pole falls. Training may time-cap; the browser never resets.

## License / papers

Code is for this project. The PDFs in `papers/` remain under their original publishers’ terms (ICML / NeurIPS / IEEE / IFAC / arXiv author versions).
