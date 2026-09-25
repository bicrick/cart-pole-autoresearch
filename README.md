# cart-pole-autoresearch

GitHub: [`bicrick/cart-pole-autoresearch`](https://github.com/bicrick/cart-pole-autoresearch) · Live: [cart-pole-autoresearch.vercel.app](https://cart-pole-autoresearch.vercel.app/#triple)

Interactive cart–pendulum demos. Drag a link, shove the cart, and switch the target equilibrium while the controller recovers.

<p align="center">
  <img src="docs/demo.gif" alt="Browser demo" width="800">
</p>

| Plant | Targets | What runs it |
| --- | --- | --- |
| **Single** (1 link) | U, D | The n-link controller with one link. 1,024 samples. |
| **Triple** (3 links, the default) | 8 equilibria, DDD through UUU | MPPI in the browser. WebGPU where available, else Web Workers. No server. |
| **Double** (2 links) | UU, UD, DU, DD | The same controller, on Web Workers. No server. |
| **Quad** (4 links) | 16 equilibria, DDDD through UUUU | The same controller on a general n-link plant. WebGPU (32,768 samples) or workers. |

`θ = 0` is upright. The ship plant has no track walls: leaving `|x| > 2.4 m` falls off and respawns.

The top-left switcher cycles the plants in link order (single, double, triple, quad). The chevrons step, the squares jump straight to a plant, and `#single`, `#double`, `#triple`, `#quad` in the URL open one directly.

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
- `A` / `D` shove. `P` turns the controller off.
- The footer shows the target and whether the pendulum is seeking it or there. `?debug` adds the sample count, sim speed, replan time and fps, plus the share of frames that waited on a plan when it's above zero. `?mppi=1024` caps the samples.

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

## Quad demo

Open [http://localhost:5173/#quad](http://localhost:5173/#quad). The page opens on DDUU: UUUU is the one equilibrium this controller does not reach reliably.

The quad uses one n-link plant and one n-link rollout, not a hand-written 5×5 solve. `train/mppi/nlink/` is the Python reference (dynamics, per-goal LQR, costs, a numba CPU kernel and a numba.cuda kernel). The browser ports are `web/src/physics-nlink.js` (the page) and `web/src/mppi/rollout-nlink.js` (CPU workers). The WebGPU shader is generated per link count by `web/src/mppi/gpu-kernel-nlink.js`.

- **Config:** `python -m mppi.export_mppi_nlink` writes `web/public/mppi/quad.json`. Horizon 1.25 s (38 knots of 4 steps), force limit ±40 N. The full profile is 32,768 samples; lite is 4,096.
- **Speed:** on an M2 Pro, WebGPU replans 32,768 samples in about 13 ms, inside the 33 ms budget, so the page holds 120 fps at the full sample count. Without WebGPU, workers adapt to roughly 2,300 samples and still swing up to DDUU.
- **Parity:** `cd train && python -m mppi.nlink.test_web`. The JS kernel matches the Python f64 kernel to about 1e-15, and the page physics matches `nlink.plant.step` to about 1e-13. `?bench#quad` checks the GPU shader against the JS kernel (median relative error about 2e-7, same 32 cheapest samples).
- **What it can do.** On an L4, 32,768 samples and a 1.5 s horizon swing up and hold 12 of the 16 goals on 10/10 episodes from a hang, and 148 of 160 episodes overall. UUUU is the hard case: about 40% at 32,768 samples, 65% at 131,072, median about 12 s when it gets there. Nothing that reaches a goal falls out of it. Reports: `policies/mppi/quad/`.
- **Fewer samples** (1.25 s horizon, 10 episodes per goal, 25 s to swing up):

| Samples | Goals at 10/10 | Hardest |
| --- | --- | --- |
| 4,096 (lite) | 9 of 16 | UUUU 1/10, DUUU 2/10, UUUD 5/10 |
| 8,192 | 10 of 16 | UUUU 0/10, DUUU 4/10, UDUU 5/10 |
| 16,384 | 11 of 16 | UUUU 0/10, DUUU 2/10, UDUU and UUUD 7/10 |

At 4,096 samples, 9 of the 11 goals with at most two links up hold 10/10, and the other two hold 8/10 (DUDU) and 9/10 (DUUD). So the lite profile covers most of the quad, and the goals with three or four links up need the full 32k on WebGPU.

## Single demo

Open [http://localhost:5173/#single](http://localhost:5173/#single). It is the quad's n-link controller with one link (`web/public/mppi/single.json`, from `python -m mppi.export_mppi_nlink --n 1`): 20 N like the double, 1,024 samples, and it swings up from a hang in under 2 s. The camera keeps the whole track in frame for the single, so the cart's pumping stays visible on phones too. `python -m mppi.nlink.test_web --n 1` checks it against the Python kernel (about 1e-15).

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
web/src/mppi/               in-browser controller: rollouts, sampler, workers, WebGPU, controller, bench
web/public/mppi/triple.json per-goal costs, LQR gains and full/lite profiles (export_mppi.py)
web/public/mppi/double.json double: per-goal costs and LQR gains (export_mppi_double.py)
web/public/mppi/quad.json   quad: 16 goals, from export_mppi_nlink.py
web/public/mppi/single.json single: U and D, from export_mppi_nlink.py --n 1
web/src/physics-nlink.js    page plant for any link count (single and quad)
web/src/plant-switch.js     top-left plant carousel
web/src/loop.js             physics and controller in the page
web/src/remote-loop.js      ?sim: render the Python sim server
web/public/policy.json      old double MLP (unused)
train/mppi/nlink/           n-link plant, LQR, costs, CPU and CUDA rollouts, sweep
scripts/mppi-server.sh      start the Python triple sim
scripts/mppi-teacher-gates.sh
shared/constants-triple.json
shared/constants-quad.json
```

## Physics

- State: `[x, ẋ, θ1, θ̇1, …, θn, θ̇n]`, from θ1 on the single to θ4 on the quad.
- Semi-implicit Euler at `dt = 1/120` s. Force limit ±40 N on the triple and the quad, ±20 N on the single and the double.
- Rates clamp at ±30 m/s (cart) and ±50 rad/s (poles).
- Walls, when on, stop the cart only. The triple demo and the teacher both run with walls off.

## Earlier work

PPO, TQC, behavior cloning, energy pumping, and open-loop trajectory tracking are in the repo and in [`docs/`](docs/). They hold near a target and do not deliver a swing-up the hold can catch. [`docs/triple-training-wheels.md`](docs/triple-training-wheels.md) is the record of that path and of the MPPI result that replaced it.
