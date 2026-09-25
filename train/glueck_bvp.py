"""Glück-style swing-up feedforward: a short force program from the hang to rest.

Automatica 2013 ends a two-point boundary-value problem at the upright with
all rates zero, in about 4 seconds, then a local Riccati controller takes
over. This is that terminal condition on our plant, with force clipped at
forceLimit and the cart stopped by the walls. It is not a reward-shaped net.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from physics_triple import accelerations, load_constants, step


def hang_state() -> torch.Tensor:
    z = torch.zeros(8, dtype=torch.float64)
    z[2] = math.pi
    z[4] = math.pi
    z[6] = math.pi
    return z


def rollout(forces: np.ndarray, steps_per: int, constants: dict) -> torch.Tensor:
    """Piecewise-constant force. Returns the final 8-D state."""
    z = hang_state()
    for u in np.asarray(forces, dtype=np.float64).reshape(-1):
        force = torch.tensor(float(u), dtype=torch.float64)
        for _ in range(int(steps_per)):
            z = step(z, force, constants=constants)
    return z


def terminal_cost(z: torch.Tensor) -> tuple[float, dict]:
    th = z[[2, 4, 6]]
    err = torch.atan2(torch.sin(th), torch.cos(th))
    om = z[[3, 5, 7]]
    ang = float(err.abs().max())
    rate = float(om.abs().max())
    cost = float((err.square().sum() + om.square().sum() + z[0].square() + z[1].square()).item())
    info = {
        "cost": cost,
        "max_abs_theta": ang,
        "max_abs_omega": rate,
        "x": float(z[0]),
        "xd": float(z[1]),
    }
    return cost, info


def solve(seconds: float = 4.0, n_segments: int = 24, maxiter: int = 40) -> dict:
    """L-BFGS-B on segment forces. Success is the measured LQR box, not full rest."""
    from scipy.optimize import minimize

    constants = load_constants()
    constants["trackWalls"] = True
    constants["forceLimit"] = 40.0
    constants["trackLimit"] = 2.4
    dt = float(constants["dt"])
    n_steps = max(n_segments, int(round(seconds / dt)))
    steps_per = max(1, n_steps // n_segments)
    limit = 40.0

    def objective(u: np.ndarray) -> float:
        _cost, info = terminal_cost(rollout(u, steps_per, constants))
        # Prefer the LQR box over a slightly closer angle that is still fast.
        return info["max_abs_theta"] + 0.15 * info["max_abs_omega"] + 0.05 * abs(info["x"])

    starts = [np.zeros(n_segments)]
    rng = np.random.default_rng(0)
    for _ in range(3):
        starts.append(rng.uniform(-limit, limit, size=n_segments))
    # One stroke each way, held for a quarter of the horizon. A pump, not a whip.
    square = np.zeros(n_segments)
    q = max(1, n_segments // 4)
    square[:q] = limit
    square[q : 2 * q] = -limit
    starts.append(square)

    best = None
    for x0 in starts:
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=[(-limit, limit)] * n_segments,
            options={"maxiter": maxiter, "ftol": 1e-6},
        )
        z = rollout(result.x, steps_per, constants)
        cost, info = terminal_cost(z)
        info["objective"] = float(result.fun)
        info["nit"] = int(result.nit)
        info["seconds"] = steps_per * n_segments * dt
        info["in_lqr"] = info["max_abs_theta"] <= 0.05 and info["max_abs_omega"] <= 0.30
        if best is None or info["objective"] < best["objective"]:
            best = info
            best["forces"] = result.x.copy()
    return best


def rest_objective(info: dict) -> float:
    """1 inside the LQR box. A fast pass and a miss are both large."""
    return (
        (info["max_abs_theta"] / 0.05) ** 2
        + (info["max_abs_omega"] / 0.30) ** 2
        + (info["xd"] / 0.5) ** 2
    )


def search_rest(
    seconds: float = 8.0,
    n_segments: int = 12,
    maxiter: int = 4,
    track_limit: float = 2.4,
) -> dict:
    """Differential evolution toward Glück's rest terminal condition."""
    from scipy.optimize import differential_evolution

    constants = load_constants()
    constants["trackWalls"] = True
    constants["forceLimit"] = 40.0
    constants["trackLimit"] = float(track_limit)
    dt = float(constants["dt"])
    n_steps = max(n_segments, int(round(seconds / dt)))
    steps_per = max(1, n_steps // n_segments)
    limit = 40.0

    def objective(u: np.ndarray) -> float:
        _cost, info = terminal_cost(rollout(u, steps_per, constants))
        return rest_objective(info)

    result = differential_evolution(
        objective,
        bounds=[(-limit, limit)] * n_segments,
        maxiter=maxiter,
        popsize=3,
        seed=0,
        polish=False,
        workers=1,
    )
    _cost, info = terminal_cost(rollout(result.x, steps_per, constants))
    info["objective"] = float(result.fun)
    info["nfev"] = int(result.nfev)
    info["seconds"] = steps_per * n_segments * dt
    info["in_lqr"] = info["max_abs_theta"] <= 0.05 and info["max_abs_omega"] <= 0.30
    info["forces"] = result.x.copy()
    return info


def _step_diff(state: torch.Tensor, force: torch.Tensor, constants: dict) -> torch.Tensor:
    """Euler step without inplace writes, so a force program can be differentiated."""
    dt = float(constants["dt"])
    fmax = float(constants["forceLimit"])
    force = force.clamp(-fmax, fmax)
    xd = state[1].clamp(-30.0, 30.0)
    w1 = state[3].clamp(-50.0, 50.0)
    w2 = state[5].clamp(-50.0, 50.0)
    w3 = state[7].clamp(-50.0, 50.0)
    clamped = torch.stack((state[0], xd, state[2], w1, state[4], w2, state[6], w3))
    acc = accelerations(clamped, force, constants=constants)
    acc = acc.clamp(-1e4, 1e4)
    nxd = (xd + acc[0] * dt).clamp(-30.0, 30.0)
    nw1 = (w1 + acc[1] * dt).clamp(-50.0, 50.0)
    nw2 = (w2 + acc[2] * dt).clamp(-50.0, 50.0)
    nw3 = (w3 + acc[3] * dt).clamp(-50.0, 50.0)
    nx = state[0] + nxd * dt
    if bool(constants.get("trackWalls", False)):
        track = float(constants.get("trackLimit", 2.4))
        hit = (nx > track) | (nx < -track)
        nx = nx.clamp(-track, track)
        nxd = torch.where(hit, torch.zeros_like(nxd), nxd)
    return torch.stack((nx, nxd, state[2] + nw1 * dt, nw1, state[4] + nw2 * dt, nw2, state[6] + nw3 * dt, nw3))


def gradient_rest(seconds: float = 8.0, n_segments: int = 24, steps: int = 25) -> dict:
    """Adam on segment forces. Gradients go through the plant, not finite differences."""
    constants = load_constants()
    constants["trackWalls"] = True
    constants["forceLimit"] = 40.0
    constants["trackLimit"] = 2.4
    dt = float(constants["dt"])
    n_steps = max(n_segments, int(round(seconds / dt)))
    steps_per = max(1, n_steps // n_segments)
    raw = torch.zeros(n_segments, dtype=torch.float64)
    q = max(1, n_segments // 4)
    raw[:q] = 0.5
    raw[q : 2 * q] = -0.5
    raw.requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=0.15)
    last = None
    for _ in range(int(steps)):
        opt.zero_grad()
        force = 40.0 * torch.tanh(raw)
        z = hang_state()
        for i in range(n_segments):
            for _k in range(steps_per):
                z = _step_diff(z, force[i], constants)
        th = z[[2, 4, 6]]
        err = torch.atan2(torch.sin(th), torch.cos(th))
        om = z[[3, 5, 7]]
        loss = (err / 0.05).square().sum() + (om / 0.30).square().sum() + (z[1] / 0.5).square()
        loss.backward()
        opt.step()
        last = z.detach()
    info_z = last
    _cost, info = terminal_cost(info_z)
    info["objective"] = float(loss.detach())
    info["seconds"] = steps_per * n_segments * dt
    info["in_lqr"] = info["max_abs_theta"] <= 0.05 and info["max_abs_omega"] <= 0.30
    info["forces"] = (40.0 * torch.tanh(raw.detach())).numpy().copy()
    return info


def _defect(reached: torch.Tensor, knot: torch.Tensor) -> torch.Tensor:
    """State error at a knot. Angles are compared on the circle."""
    d = reached - knot
    dth = torch.atan2(torch.sin(d[[2, 4, 6]]), torch.cos(d[[2, 4, 6]]))
    return torch.stack((d[0], d[1], dth[0], d[3], dth[1], d[5], dth[2], d[7]))


def multiple_shooting_rest(seconds: float = 8.0, seg_seconds: float = 0.25, steps: int = 40) -> dict:
    """Short-horizon multiple shooting. An 8-second backward pass overflows."""
    constants = load_constants()
    constants["trackWalls"] = True
    constants["forceLimit"] = 40.0
    constants["trackLimit"] = 2.4
    dt = float(constants["dt"])
    steps_per = max(1, int(round(seg_seconds / dt)))
    n_segments = max(1, int(round(seconds / dt)) // steps_per)
    raw0 = torch.zeros(n_segments, dtype=torch.float64)
    q = max(1, n_segments // 4)
    raw0[:q] = 0.5
    raw0[q : 2 * q] = -0.5
    with torch.no_grad():
        knots0 = []
        z = hang_state()
        force0 = 40.0 * torch.tanh(raw0)
        for i in range(n_segments - 1):
            for _k in range(steps_per):
                z = _step_diff(z, force0[i], constants)
            knots0.append(z.clone())
    raw = raw0.clone().detach().requires_grad_(True)
    knots = torch.stack(knots0).detach().requires_grad_(True)
    opt = torch.optim.Adam(
        [{"params": [raw], "lr": 0.05}, {"params": [knots], "lr": 0.02}]
    )
    last_loss = None
    for step_i in range(int(steps)):
        opt.zero_grad()
        force = 40.0 * torch.tanh(raw)
        z = hang_state()
        defects = []
        for i in range(n_segments - 1):
            for _k in range(steps_per):
                z = _step_diff(z, force[i], constants)
            defects.append(_defect(z, knots[i]))
            z = knots[i]
        for _k in range(steps_per):
            z = _step_diff(z, force[-1], constants)
        th = z[[2, 4, 6]]
        err = torch.atan2(torch.sin(th), torch.cos(th))
        om = z[[3, 5, 7]]
        terminal = (err / 0.05).square().sum() + (om / 0.30).square().sum() + (z[1] / 0.5).square()
        defect_cost = torch.stack([d.square().sum() for d in defects]).sum()
        loss = terminal + 500.0 * defect_cost
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw, knots], 50.0)
        opt.step()
        last_loss = loss.detach()
        if step_i % 20 == 0 or step_i == int(steps) - 1:
            with torch.no_grad():
                jump = max(float(d.abs().max()) for d in defects)
            print(
                f"shoot_step={step_i} loss={float(last_loss):.1f} "
                f"terminal={float(terminal.detach()):.1f} max_defect={jump:.4f}",
                flush=True,
            )
    with torch.no_grad():
        force = 40.0 * torch.tanh(raw.detach())
        z = hang_state()
        max_defect = 0.0
        for i in range(n_segments - 1):
            for _k in range(steps_per):
                z = _step_diff(z, force[i], constants)
            max_defect = max(max_defect, float(_defect(z, knots[i]).abs().max()))
            z = knots[i]
        for _k in range(steps_per):
            z = _step_diff(z, force[-1], constants)
        # Open-loop replay from the hang. Knot jumps do not count as a swing.
        z_open = hang_state()
        for i in range(n_segments):
            for _k in range(steps_per):
                z_open = _step_diff(z_open, force[i], constants)
    _cost, info = terminal_cost(z_open)
    info["objective"] = float(last_loss)
    info["seconds"] = n_segments * steps_per * dt
    info["max_defect"] = max_defect
    info["in_lqr"] = info["max_abs_theta"] <= 0.05 and info["max_abs_omega"] <= 0.30
    info["forces"] = force.numpy().copy()
    return info


def solve_bvp_rest(
    seconds: float = 4.0,
    n_coef: int = 8,
    n_mesh: int = 30,
    tol: float = 0.05,
    goal: str = "UUU",
    guess_t: np.ndarray | None = None,
    guess_y: np.ndarray | None = None,
    guess_p: np.ndarray | None = None,
) -> dict:
    """Collocation Newton on the continuous plant. Force is a cosine series.

    Eight free coefficients match the eight extra boundary conditions
    (hang at t=0, upright and stopped at t=T). No walls and no force clip,
    so the peak force is the size of the maneuver.
    """
    from scipy.integrate import solve_bvp

    constants = load_constants()
    constants["trackWalls"] = False

    def force_of(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        ks = np.arange(n_coef, dtype=np.float64)
        phase = np.pi * np.asarray(t, dtype=np.float64).reshape(-1, 1) / seconds * ks.reshape(1, -1)
        return np.cos(phase) @ np.asarray(p, dtype=np.float64).reshape(-1)

    def fun(t, y, p):
        u = force_of(t, p)
        state = torch.tensor(np.ascontiguousarray(y.T), dtype=torch.float64)
        force = torch.tensor(u, dtype=torch.float64)
        with torch.no_grad():
            acc = accelerations(state, force, constants=constants).detach().cpu().numpy()
        dydt = np.zeros_like(y)
        dydt[0] = y[1]
        dydt[1] = acc[:, 0]
        dydt[2] = y[3]
        dydt[3] = acc[:, 1]
        dydt[4] = y[5]
        dydt[5] = acc[:, 2]
        dydt[6] = y[7]
        dydt[7] = acc[:, 3]
        return dydt

    from lqr_uuu import equilibrium_state

    hang = np.array([0.0, 0.0, math.pi, 0.0, math.pi, 0.0, math.pi, 0.0])
    target = equilibrium_state(goal)

    def bc(ya, yb, p):
        del p
        return np.concatenate((ya - hang, yb - target))

    if guess_t is not None and guess_y is not None:
        t = np.asarray(guess_t, dtype=np.float64)
        y = np.asarray(guess_y, dtype=np.float64)
        p0 = np.zeros(n_coef) if guess_p is None else np.asarray(guess_p, dtype=np.float64)
    else:
        t = np.linspace(0.0, seconds, n_mesh)
        y = np.zeros((8, n_mesh), dtype=np.float64)
        for idx, ang0 in ((2, hang[2]), (4, hang[4]), (6, hang[6])):
            ang1 = float(target[idx])
            y[idx] = np.linspace(ang0, ang1, n_mesh)
            y[idx + 1] = (ang1 - ang0) / seconds
        p0 = np.zeros(n_coef)
    sol = solve_bvp(fun, bc, t, y, p=p0, tol=tol, max_nodes=400, verbose=1)
    u = force_of(sol.x, sol.p)
    end = sol.y[:, -1]
    th = end[[2, 4, 6]] - target[[2, 4, 6]]
    err = np.arctan2(np.sin(th), np.cos(th))
    om = end[[3, 5, 7]]
    path_om = np.max(np.abs(sol.y[[3, 5, 7], :]))
    # Discrete replay. The collocation solution is continuous; the demo steps at dt.
    z = hang_state()
    dt = float(constants["dt"])
    n_steps = int(round(seconds / dt))
    replay_constants = dict(constants)
    replay_constants["forceLimit"] = 40.0
    peak_u = 0.0
    ref = sol.sol(np.linspace(0.0, seconds, n_steps + 1))
    max_step_defect = 0.0
    for i in range(n_steps):
        ui = float(force_of(np.array([(i + 0.5) * dt]), sol.p)[0])
        peak_u = max(peak_u, abs(ui))
        z_ref = torch.tensor(ref[:, i], dtype=torch.float64)
        z_next = step(z_ref, torch.tensor(ui, dtype=torch.float64), constants=replay_constants)
        defect = z_next.detach().cpu().numpy() - ref[:, i + 1]
        for idx in (2, 4, 6):
            defect[idx] = math.atan2(math.sin(defect[idx]), math.cos(defect[idx]))
        max_step_defect = max(max_step_defect, float(np.max(np.abs(defect))))
        z = step(z, torch.tensor(ui, dtype=torch.float64), constants=replay_constants)
    z_end = z.detach().cpu().numpy()
    th_r = z_end[[2, 4, 6]] - target[[2, 4, 6]]
    err_r = np.arctan2(np.sin(th_r), np.cos(th_r))
    om_r = z_end[[3, 5, 7]]
    solve_bvp_rest.last_sol = sol
    solve_bvp_rest.last_constants = constants
    return {
        "success": bool(sol.success),
        "status": int(sol.status),
        "message": str(sol.message),
        "seconds": seconds,
        "max_abs_force": float(np.max(np.abs(u))),
        "replay_peak_force": peak_u,
        "max_abs_theta": float(np.max(np.abs(err))),
        "max_abs_omega": float(np.max(np.abs(om))),
        "path_max_abs_omega": float(path_om),
        "path_max_abs_x": float(np.max(np.abs(sol.y[0]))),
        "x": float(end[0]),
        "xd": float(end[1]),
        "replay_max_abs_theta": float(np.max(np.abs(err_r))),
        "replay_max_abs_omega": float(np.max(np.abs(om_r))),
        "replay_x": float(z_end[0]),
        "replay_xd": float(z_end[1]),
        "max_step_defect": max_step_defect,
        "n_nodes": int(sol.x.size),
        "rms_bc": float(np.sqrt(np.mean(np.square(sol.rms_residuals)))) if np.size(sol.rms_residuals) else float("nan"),
        "coefficients": np.asarray(sol.p, dtype=np.float64).copy(),
    }


def _discrete_ab(z_ref: np.ndarray, u_ref: float, constants: dict) -> tuple[np.ndarray, np.ndarray]:
    """Discrete A, B of one Euler step at a reference state and force."""
    z0 = torch.tensor(np.asarray(z_ref, dtype=np.float64), dtype=torch.float64)
    u0 = float(u_ref)
    eps_z = 1e-5
    eps_u = 1e-3

    def nxt(z: torch.Tensor, u: float) -> np.ndarray:
        return _step_diff(z, torch.tensor(u, dtype=torch.float64), constants).detach().numpy()

    A = np.zeros((8, 8), dtype=np.float64)
    with torch.no_grad():
        for i in range(8):
            zp = z0.clone()
            zm = z0.clone()
            zp[i] += eps_z
            zm[i] -= eps_z
            A[:, i] = (nxt(zp, u0) - nxt(zm, u0)) / (2.0 * eps_z)
        B = ((nxt(z0, u0 + eps_u) - nxt(z0, u0 - eps_u)) / (2.0 * eps_u)).reshape(8, 1)
    return A, B


def _track_riccati(
    knot_v: torch.Tensor,
    forces: np.ndarray,
    constants: dict,
    steps_per: int,
) -> dict:
    """Time-varying Riccati along the stitched pieces. Playback does not reset."""
    from lqr_uuu import build_Q, state_error

    hang = hang_state()
    refs: list[np.ndarray] = []
    us: list[float] = []
    with torch.no_grad():
        for i in range(int(forces.shape[0])):
            z = hang.clone() if i == 0 else knot_v[i - 1].detach().clone()
            u = float(forces[i])
            for _k in range(steps_per):
                refs.append(z.detach().numpy().copy())
                us.append(u)
                z = _step_diff(z, torch.tensor(u, dtype=torch.float64), constants)
    n = len(refs)
    q = build_Q()
    r = np.array([[0.01]], dtype=np.float64)
    ab = [_discrete_ab(refs[k], us[k], constants) for k in range(n)]
    p = q.copy()
    gains: list[np.ndarray] = []
    for k in range(n - 1, -1, -1):
        A, B = ab[k]
        s = r + B.T @ p @ B
        gain = np.linalg.solve(s, B.T @ p @ A)
        p = q + A.T @ p @ A - A.T @ p @ B @ gain
        p = 0.5 * (p + p.T)
        gains.append(gain)
    gains.reverse()
    def play(z0: torch.Tensor, sim: dict | None = None) -> tuple[dict, int]:
        sim_c = constants if sim is None else sim
        z = z0.clone()
        saturated = 0
        max_x = 0.0
        with torch.no_grad():
            for k in range(n):
                err = state_error(z.detach().numpy(), refs[k])
                u = float(us[k] - (gains[k] @ err).ravel()[0])
                if abs(u) >= 39.9:
                    saturated += 1
                u = max(-40.0, min(40.0, u))
                z = _step_diff(z, torch.tensor(u, dtype=torch.float64), sim_c)
                max_x = max(max_x, abs(float(z[0])))
        _cost, info = terminal_cost(z.detach())
        info["in_lqr"] = info["max_abs_theta"] <= 0.05 and info["max_abs_omega"] <= 0.30
        info["in_quiet"] = info["max_abs_theta"] <= 0.03 and info["max_abs_omega"] <= 0.01
        info["final_state"] = z.detach().numpy().copy()
        info["max_abs_x"] = max_x
        return info, saturated

    info, saturated = play(hang_state())
    info["saturated_frac"] = saturated / max(n, 1)
    rng = np.random.default_rng(0)
    noisy = []
    for _ep in range(5):
        z0 = hang_state()
        z0[0] = float(rng.uniform(-0.05, 0.05))
        z0[1] = float(rng.uniform(-0.01, 0.01))
        for idx in (2, 4, 6):
            z0[idx] = math.pi + float(rng.uniform(-0.05, 0.05))
        for idx in (3, 5, 7):
            z0[idx] = float(rng.uniform(-0.01, 0.01))
        trial, _sat = play(z0)
        noisy.append(trial)
    info["noisy_lqr_rate"] = float(np.mean([1.0 if t["in_lqr"] else 0.0 for t in noisy]))
    info["noisy_quiet_rate"] = float(np.mean([1.0 if t["in_quiet"] else 0.0 for t in noisy]))
    info["noisy_worst_theta"] = float(max(t["max_abs_theta"] for t in noisy))
    info["noisy_worst_omega"] = float(max(t["max_abs_omega"] for t in noisy))
    info["arrival_states"] = [info["final_state"]] + [t["final_state"] for t in noisy]
    walled = dict(constants)
    walled["trackWalls"] = True
    walled["trackLimit"] = 2.4
    walled["forceLimit"] = 40.0
    wall_info, _wall_sat = play(hang_state(), walled)
    info["walls_in_quiet"] = wall_info["in_quiet"]
    info["walls_max_abs_x"] = wall_info["max_abs_x"]
    info["walls_max_abs_theta"] = wall_info["max_abs_theta"]
    info["walls_max_abs_omega"] = wall_info["max_abs_omega"]
    info["track_refs"] = refs
    info["track_forces"] = us
    info["track_gains"] = gains
    return info


def discrete_newton_rest(
    seconds: float = 4.0,
    n_segments: int = 16,
    iters: int = 12,
    goal: str = "UUU",
    step_clip: float = 1.0,
) -> dict:
    """Newton on the demo stepper, seeded by the continuous collocation curve.

    Each piece is short enough that its Jacobian stays finite. The open-loop
    replay from the hang is the trajectory the demo would actually step.
    A failed collocation is not a seed: the pieces then start on a straight
    angle interpolation with zero force.
    """
    from lqr_uuu import equilibrium_state

    info = solve_bvp_rest(seconds=seconds, tol=1e-3, goal=goal)
    sol = solve_bvp_rest.last_sol
    constants = dict(solve_bvp_rest.last_constants)
    constants["forceLimit"] = 40.0
    constants["trackWalls"] = False
    dt = float(constants["dt"])
    steps_per = max(1, int(round(seconds / dt)) // n_segments)
    target_np = equilibrium_state(goal)
    hang = hang_state()
    upright = torch.tensor(target_np, dtype=torch.float64)
    if info["success"]:
        coef = np.asarray(info["coefficients"], dtype=np.float64)
        ks = np.arange(coef.size, dtype=np.float64)

        def series_force(t: float) -> float:
            phase = np.pi * t / seconds * ks
            return float(np.cos(phase) @ coef)

        times = np.linspace(0.0, seconds, n_segments + 1)
        ref = sol.sol(times)
        raw = torch.tensor(
            [math.atanh(max(-0.999, min(0.999, series_force((i + 0.5) * seconds / n_segments) / 40.0))) for i in range(n_segments)],
            dtype=torch.float64,
        )
        knots = torch.tensor(ref[:, 1:-1].T, dtype=torch.float64)
    else:
        hang_np = hang.detach().numpy()
        if goal == "UDD":
            from energy_pump import EnergyPump

            pump = EnergyPump(constants=constants, target="inner", track_limit=2.4)
            z_roll = hang.clone()
            samples = [hang_np.copy()]
            seg_force: list[float] = []
            with torch.no_grad():
                for i in range(n_segments * steps_per):
                    u_i = pump.force(z_roll.detach().numpy())
                    seg_force.append(u_i)
                    z_roll = _step_diff(z_roll, torch.tensor(u_i, dtype=torch.float64), constants)
                    if (i + 1) % steps_per == 0:
                        samples.append(z_roll.detach().numpy().copy())
            raw = torch.tensor(
                [math.atanh(max(-0.999, min(0.999, float(np.mean(seg_force[i * steps_per:(i + 1) * steps_per])) / 40.0))) for i in range(n_segments)],
                dtype=torch.float64,
            )
            knots = torch.tensor(np.stack(samples[1:-1]), dtype=torch.float64)
        else:
            raw = torch.zeros(n_segments, dtype=torch.float64)
            rows = []
            for i in range(1, n_segments):
                a = i / n_segments
                z = (1.0 - a) * hang_np + a * target_np
                z[1] = 0.0
                z[[3, 5, 7]] = (target_np[[2, 4, 6]] - hang_np[[2, 4, 6]]) / seconds
                rows.append(z)
            knots = torch.tensor(np.stack(rows), dtype=torch.float64)

    def residuals(raw_v: torch.Tensor, knot_v: torch.Tensor) -> torch.Tensor:
        pieces = []
        for i in range(n_segments):
            z0 = hang if i == 0 else knot_v[i - 1]
            u = 40.0 * torch.tanh(raw_v[i])
            z = z0
            for _k in range(steps_per):
                z = _step_diff(z, u, constants)
            z1 = upright if i == n_segments - 1 else knot_v[i]
            pieces.append(_defect(z, z1))
        return torch.cat(pieces)

    def pack(raw_v: torch.Tensor, knot_v: torch.Tensor) -> np.ndarray:
        return np.concatenate((raw_v.detach().numpy(), knot_v.detach().numpy().reshape(-1)))

    def unpack(vec: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        raw_v = torch.tensor(vec[:n_segments], dtype=torch.float64)
        knot_v = torch.tensor(vec[n_segments:], dtype=torch.float64).reshape(n_segments - 1, 8)
        return raw_v, knot_v

    def one_residual(z0: torch.Tensor, raw_i: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        u = 40.0 * torch.tanh(raw_i[0])
        z = z0
        for _k in range(steps_per):
            z = _step_diff(z, u, constants)
        return _defect(z, z1)

    def jac_and_res(raw_v: torch.Tensor, knot_v: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        n_par = n_segments + (n_segments - 1) * 8
        jmat = np.zeros((n_segments * 8, n_par), dtype=np.float64)
        rows = []
        for i in range(n_segments):
            z0 = hang if i == 0 else knot_v[i - 1].detach()
            z1 = upright if i == n_segments - 1 else knot_v[i].detach()
            raw_i = raw_v[i].detach().reshape(1)
            block = slice(i * 8, (i + 1) * 8)
            jr = torch.autograd.functional.jacobian(lambda v, z0=z0, z1=z1: one_residual(z0, v, z1), raw_i)
            jmat[block, i] = jr.detach().reshape(8).numpy()
            if i > 0:
                jz0 = torch.autograd.functional.jacobian(
                    lambda v, raw_i=raw_i, z1=z1: one_residual(v, raw_i, z1), z0
                )
                col = n_segments + (i - 1) * 8
                jmat[block, col : col + 8] = jz0.detach().numpy()
            if i < n_segments - 1:
                jz1 = torch.autograd.functional.jacobian(
                    lambda v, z0=z0, raw_i=raw_i: one_residual(z0, raw_i, v), z1
                )
                col = n_segments + i * 8
                jmat[block, col : col + 8] = jz1.detach().numpy()
            rows.append(one_residual(z0, raw_i, z1).detach().numpy())
        return jmat, np.concatenate(rows)

    vec = pack(raw, knots)
    lam = 1e-2
    best = vec.copy()
    best_norm = float("inf")
    for it in range(int(iters)):
        raw_v, knot_v = unpack(vec)
        jmat, res = jac_and_res(raw_v, knot_v)
        norm = float(np.linalg.norm(res))
        print(f"newton_iter={it} residual={norm:.4f} lambda={lam:.1e}", flush=True)
        if norm < best_norm:
            best_norm = norm
            best = vec.copy()
            lam = max(lam * 0.3, 1e-6)
        else:
            lam = min(lam * 10.0, 1e6)
            vec = best.copy()
            raw_v, knot_v = unpack(vec)
            jmat, res = jac_and_res(raw_v, knot_v)
        gram = jmat.T @ jmat
        gram.flat[:: gram.shape[0] + 1] += lam
        step_vec = np.linalg.solve(gram, -(jmat.T @ res))
        step_norm = float(np.linalg.norm(step_vec))
        if step_norm > step_clip:
            step_vec *= step_clip / step_norm
        vec = vec + step_vec
    raw_v, knot_v = unpack(best)
    forces = (40.0 * torch.tanh(raw_v)).numpy()
    z = hang_state()
    jumped = 0.0
    for i in range(n_segments):
        u = torch.tensor(float(forces[i]), dtype=torch.float64)
        for _k in range(steps_per):
            z = _step_diff(z, u, constants)
        if i < n_segments - 1:
            jumped = max(jumped, float(_defect(z, knot_v[i]).abs().max()))
    _cost, out = terminal_cost(z.detach())
    final = z.detach().numpy()
    dth = final[[2, 4, 6]] - target_np[[2, 4, 6]]
    derr = np.arctan2(np.sin(dth), np.cos(dth))
    out["target_theta"] = float(np.max(np.abs(derr)))
    out["target_omega"] = float(np.max(np.abs(final[[3, 5, 7]])))
    out["max_knot_jump"] = jumped
    out["objective"] = best_norm
    out["seconds"] = n_segments * steps_per * dt
    out["max_abs_force"] = float(np.max(np.abs(forces)))
    out["in_lqr"] = out["max_abs_theta"] <= 0.05 and out["max_abs_omega"] <= 0.30
    out["forces"] = forces.copy()
    out["bvp_success"] = bool(info["success"])
    if goal != "UUU":
        tracked = _track_riccati(knot_v, forces, constants, steps_per)
        arrival = tracked["final_state"]
        dth = arrival[[2, 4, 6]] - target_np[[2, 4, 6]]
        derr = np.arctan2(np.sin(dth), np.cos(dth))
        out["track_target_theta"] = float(np.max(np.abs(derr)))
        out["track_target_omega"] = float(np.max(np.abs(arrival[[3, 5, 7]])))
        out["track_x"] = float(arrival[0])
        out["track_saturated_frac"] = tracked["saturated_frac"]
        out["track_max_abs_x"] = tracked.get("max_abs_x", tracked.get("walls_max_abs_x"))
        discrete_newton_rest.last_track = tracked
        return out
    tracked = _track_riccati(knot_v, forces, constants, steps_per)
    out["track_max_abs_theta"] = tracked["max_abs_theta"]
    out["track_max_abs_omega"] = tracked["max_abs_omega"]
    out["track_x"] = tracked["x"]
    out["track_xd"] = tracked["xd"]
    out["track_saturated_frac"] = tracked["saturated_frac"]
    out["track_in_lqr"] = tracked["in_lqr"]
    out["track_in_quiet"] = tracked["in_quiet"]
    out["noisy_lqr_rate"] = tracked["noisy_lqr_rate"]
    out["noisy_quiet_rate"] = tracked["noisy_quiet_rate"]
    out["noisy_worst_theta"] = tracked["noisy_worst_theta"]
    out["noisy_worst_omega"] = tracked["noisy_worst_omega"]
    out["walls_in_quiet"] = tracked["walls_in_quiet"]
    out["walls_max_abs_x"] = tracked["walls_max_abs_x"]
    out["walls_max_abs_theta"] = tracked["walls_max_abs_theta"]
    out["walls_max_abs_omega"] = tracked["walls_max_abs_omega"]
    discrete_newton_rest.last_track = tracked
    held = _latch_uuu_hold(tracked["arrival_states"])
    out["hold_survival"] = held["survival"]
    out["hold_at_goal"] = held["at_goal"]
    out["hold_align"] = held["align"]
    return out


def _latch_uuu_hold(states: list[np.ndarray], steps: int = 1000) -> dict:
    """Run the gated UUU hold from tracker arrivals. Does not write the zip."""
    from sb3_contrib import TQC

    from envs.triple_gym import TriplePendulumUUUEnv

    model = TQC.load("policies/tqc-m2-uuu-hold-w2-n004-lqrbc.zip", device="cpu")
    env = TriplePendulumUUUEnv(
        force_limit=40.0,
        max_steps=steps,
        track_walls=True,
        init_mode="near_target",
        init_noise=0.03,
        hang_frac=0.0,
        wide_frac=0.0,
        progress_w=0.0,
        angle_fall=True,
        seed=0,
        goal="UUU",
    )
    survival = []
    at_goal_rate = []
    align = []
    for s in states:
        env.reset(seed=0)
        st = torch.tensor(np.asarray(s, dtype=np.float32), dtype=torch.float32, device=env.device)
        env._state = st.clone()
        env._prev_state = st.clone()
        env._step_count = 0
        obs = env._obs_np()
        last: dict = {}
        for _ in range(steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _reward, term, trunc, last = env.step(act)
            if term or trunc:
                break
        survival.append(1.0 if last.get("survival_success") else 0.0)
        at_goal_rate.append(1.0 if last.get("at_goal_final", last.get("at_goal")) else 0.0)
        align.append(float(last.get("align", float("nan"))))
    return {
        "survival": float(np.mean(survival)),
        "at_goal": float(np.mean(at_goal_rate)),
        "align": float(np.nanmean(align)),
    }
