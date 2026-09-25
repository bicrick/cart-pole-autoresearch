# cart-pole-autoresearch

GitHub: [`bicrick/cart-pole-autoresearch`](https://github.com/bicrick/cart-pole-autoresearch) · Live: [cart-pole-autoresearch.vercel.app](https://cart-pole-autoresearch.vercel.app/#triple)

Interactive cart–pendulum demos. Drag a link, shove the cart, and switch the target equilibrium while the controller recovers.

<p align="center">
  <img src="docs/demo.gif" alt="Browser demo" width="800">
</p>

| Plant | Targets | What runs it |
| --- | --- | --- |
| **Triple** (3 links) | 8 equilibria, DDD through UUU | MPPI teacher in the browser, on Web Workers. No server. |
| **Double** (2 links) | UU, UD, DU, DD | The same MPPI controller, on Web Workers. No server. |

`θ = 0` is upright. The ship plant has no track walls: leaving `|x| > 2.4 m` falls off and respawns.

## Triple demo

The page runs the MPPI teacher that was gated in Python, with no server. Samples run on one of three backends:

| Backend | When | 4096-sample replan (M2 Pro) |
| --- | --- | --- |
| WebGPU (`web/src/mppi/gpu.js`) | default where `navigator.gpu` exists | ~3 ms |
| Web Workers (`pool.js`) | no WebGPU | ~15–23 ms on 10 workers |
| In-thread (`?backend=local`) | no Worker support | too slow for real time |

`?backend=workers` forces the CPU path.

Two profiles, both gated in Python (`python -m mppi.speed_sweep`) and in node:

- **full:** 4096 samples, replan every 4 steps. Used on WebGPU and on desktop workers.
- **lite:** 2048 samples, replan every 8 steps, about 4x less compute. Used on phones without WebGPU. It reaches upright 95% of the time within 12.5 s, against 100% for full.
- **Both:** while the pendulum is quiet at the target they drop to 128 samples, and they stop scoring a sample once it has left the track.
- `?profile=full|lite` overrides the choice.

**Smoothness.** Physics steps at 120 Hz and every display frame is drawn interpolated between the last two physics states, so motion is even at 60, 120 or 144 Hz. Each plan is computed one replan interval early, from the state predicted under the current plan. That prediction is exact unless you drag or shove in between, and physics only waits if a replan takes longer than a whole replan interval. `?pipeline=0` plans from the exact boundary state instead, as the Python server does; physics then waits on about 60% of frames with workers.

```bash
cd web && npm install && npm run dev
```

Open [http://localhost:5173/#triple](http://localhost:5173/#triple). Click the canvas or press a key to start.

- Grab the cart or a joint.
- `1`–`8` pick a goal. `Tab` cycles. The pendulum goes there from wherever it is.
- `A` / `D` shove. `P` turns the teacher off.
- The footer shows the sample count, sim speed, replan time and fps, plus the share of frames that waited on a plan when it's above zero. `?mppi=1024` caps the samples.

**Phones and benchmarks.**
- `?bench` times real replans at 256 to 4096 samples on the current device and says how many fit in real time. On WebGPU it also checks the GPU costs against the JS kernel.
- To try a phone on the same Wi-Fi: `cd web && npm run build && npm run preview -- --host`, then open `http://<your-mac-ip>:4173/?bench#triple`. Note that browsers only expose WebGPU on `localhost` or HTTPS, so over plain LAN HTTP the phone falls back to workers and the lite profile.

**Checks.**
- The JS rollout (`web/src/mppi/rollout.js`) matches the numba kernel to about 1e-16 on all 8 goals.
- The WebGPU kernel matches in f32, with a median relative error of about 2e-7 and the same 32 cheapest samples.
- The pipelined JS controller in node swings up and holds 8/8 from a hang: `cd train && python -m mppi.test_web_mppi`.
- After changing the teacher config, re-export with `python -m mppi.export_mppi`.

The Python sim server is still there: `WARM=1 bash scripts/mppi-server.sh`, then open `/?sim#triple`.

## Double demo

```bash
cd web && npm install && npm run dev
```

Open [http://localhost:5173/#double](http://localhost:5173/#double).

The double runs the triple's MPPI controller: a gated LQR anchor, 1.5 s residual plans, the same costs, and hold mode. The plant is `web/src/physics.js` (20 N, closed-form 3x3 solve), and `web/src/mppi/rollout-double.js` matches it to about 1e-14. It uses 2048 samples on Web Workers, about 5 ms per replan on an M2 Pro, and there is no WebGPU kernel for it.

- **Swing-up test:** in node from a hang (DD from near UU), 32/32 swing-ups and holds across the four goals at 2048 samples, and also at 512. The median time into the quiet box is 3.1–3.7 s, and 5.6 s for DD. Run it with `cd train && python -m mppi.test_web_mppi_double`.
- **Config:** per-goal LQR and costs come from `python -m mppi.export_mppi_double` (writes `web/public/mppi/double.json`).
- **Old policy:** the small MLP (`web/public/policy.json`) is no longer loaded.

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

Distilling the teacher into a network did not work. A state-only MLP cannot fit it, because the teacher's force depends on the plan it is following as well as the state. A net that also takes the plan fits the teacher's own trajectories. Once that net drives, though, the teacher's labels either scatter (~22 N between repeat runs) or depend on the teacher's hidden plan. The browser therefore runs the teacher itself.

## Layout

```text
train/mppi/                 teacher, costs, numba rollout, sim server, web export
train/mppi/configs/uuu.json 4096 samples, 1.5 s horizon
train/physics_triple.py     reference triple step
train/mppi/dynamics_fast.py batched step used by the teacher (matches the reference)
web/src/mppi/               in-browser teacher: rollout, sampler, workers, WebGPU, controller, bench
web/public/mppi/triple.json per-goal costs, LQR gains and full/lite profiles (export_mppi.py)
web/src/loop.js             physics and controller in the page (both plants)
web/src/remote-loop.js      ?sim: render the Python sim server
web/public/mppi/double.json double: per-goal costs and LQR gains (export_mppi_double.py)
web/public/policy.json      old double MLP (unused)
scripts/mppi-server.sh      start the Python triple sim
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
