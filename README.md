# cart-pole-autoresearch

GitHub: [`bicrick/cart-pole-autoresearch`](https://github.com/bicrick/cart-pole-autoresearch)

Interactive cart–pendulum demos. Drag a link, shove the cart, and switch the target equilibrium while the controller recovers.

<p align="center">
  <img src="docs/demo.gif" alt="Browser demo" width="800">
</p>

| Plant | Targets | What runs it |
| --- | --- | --- |
| **Triple** (3 links) | 8 equilibria, DDD through UUU | MPPI teacher in a local Python sim. The browser only draws. |
| **Double** (2 links) | UU, UD, DU, DD | Small MLP in the browser. No server. |

`θ = 0` is upright. The ship plant has no track walls: leaving `|x| > 2.4 m` falls off and respawns.

## Triple demo

The server steps the same plant the teacher was gated on, at 120 Hz, with the 4096-sample teacher in the loop. On an M2 Pro a replan takes about 12 ms, which is inside the 33 ms budget, so the sim holds real time.

```bash
python3 -m pip install -r requirements.txt
WARM=1 bash scripts/mppi-server.sh
```

In another shell:

```bash
cd web && npm install && npm run dev
```

Open [http://localhost:5173/#triple](http://localhost:5173/#triple). Click the canvas or press a key to start. Only the focused tab drives the sim.

- Grab the cart or a joint.
- `1`–`8` pick a goal. `Tab` cycles. The pendulum goes there from wherever it is.
- `A` / `D` shove. `P` turns the teacher off.
- The footer shows speed and replan time.

`SAMPLES=1024 bash scripts/mppi-server.sh` is the lighter server. `PORT` and `THREADS` override the defaults (`8765`, 8).

## Double demo

```bash
cd web && npm install && npm run dev
```

Open [http://localhost:5173/#double](http://localhost:5173/#double). The page loads `web/public/policy.json` and steps the physics itself.

Same plant in pygame: `python3 train/play.py`.

## Triple controller

Sampling MPC with a gated LQR anchor. Each sample is

```text
u = clip(gate(s) * (-K e) + u_ff + noise, ±40 N)
```

`gate` is 1 inside the target's catch region and fades out by the time the angle distance `Σ(1 − cos eᵢ)` reaches 0.10. Swing-up plans are open-loop residuals, scored on whether the local LQR can catch the arrival. The plan is 45 knots of 4 steps (1.5 s) and is replanned at every knot. Config: `train/mppi/configs/uuu.json`. The same file covers every goal via `--goal`.

Rollouts run in a fused multicore numba kernel (`train/mppi/fast_rollout.py`), checked against the torch path.

| Target | Start | Episodes | Swing-up and hold | Median time to quiet |
| --- | --- | --- | --- | --- |
| UUU | hang | 50 | 49/50 | 5.5 s |
| DDD | near UUU | 10 | 10/10 | 7.4 s |
| DDU, DUD, DUU, UDD, UDU, UUD | hang | 10 each | 10/10 | 4.5–7.2 s |

Quiet means all three angle errors under 0.03 rad and rates under 0.01 rad/s, then a hold inside the fall band. Reports: `policies/mppi/`.

```bash
GOAL=UUU N=50 bash scripts/mppi-teacher-gates.sh
```

A browser-native student is the next step. The first DAgger fits hold near the target and do not swing up; the live demo still uses the teacher.

## Layout

```text
train/mppi/                 teacher, costs, numba rollout, sim server
train/mppi/configs/uuu.json 4096 samples, 1.5 s horizon
train/physics_triple.py     reference triple step
train/mppi/dynamics_fast.py batched step used by the teacher (matches the reference)
web/src/remote-loop.js      triple: render + send keys and drags
web/src/loop.js             double: physics and policy in the page
web/public/policy.json      double MLP
scripts/mppi-server.sh      start the triple sim
scripts/mppi-teacher-gates.sh
shared/constants-triple.json
```

## Physics

- State: `[x, ẋ, θ1, θ̇1, θ2, θ̇2]` and, for the triple, `θ3, θ̇3`.
- Semi-implicit Euler at `dt = 1/120` s. Force limit ±40 N on the triple.
- Rates clamp at ±30 m/s (cart) and ±50 rad/s (poles).
- Walls, when on, stop the cart only. The triple demo and the teacher both run with walls off.

## Earlier work

PPO, TQC, behavior cloning, energy pumping, and open-loop trajectory tracking are in the repo and in [`docs/`](docs/). They hold near a target and do not deliver a swing-up the hold can catch. [`docs/triple-training-wheels.md`](docs/triple-training-wheels.md) is the record of that path and of the MPPI result that replaced it.
