/**
 * Render loop for a server-owned sim (train/mppi/sim_server.py).
 *
 * The Python server steps the physics and runs the controller; this loop only
 * draws the latest state and sends inputs (goal, policy toggle, a/d force,
 * drag target) every frame. Same interface as startLoop: stop/getState/setState.
 */
export function startRemoteLoop({
  url,
  plant,
  input,
  constants,
  getGoal,
  getPolicyOn,
  getManualForce,
  getStarted,
  onFrame,
  onStatus,
}) {
  let state = { ...(plant.initialState || plant.hanging) };
  let force = 0;
  let info = { rt: 0, ms: 0, status: "connecting" };
  let ws = null;
  let open = false;
  let raf = 0;
  let stopped = false;

  function connect() {
    ws = new WebSocket(url);
    ws.onopen = () => {
      open = true;
      onStatus?.("sim server connected");
    };
    ws.onclose = () => {
      open = false;
      onStatus?.("sim server offline · run scripts/mppi-server.sh");
      if (!stopped) setTimeout(connect, 1500);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type !== "state") return;
      state = msg.state;
      force = msg.force;
      info = { rt: msg.rt, ms: msg.ms, status: msg.status, policy: msg.policy };
      onStatus?.(msg.status);
    };
  }

  function send(msg) {
    if (open && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  }

  function frame() {
    raf = requestAnimationFrame(frame);
    const started = typeof getStarted !== "function" || getStarted();
    const goalId = getGoal();
    const pointer = input.pointer;
    const grab =
      started && pointer.active && pointer.body
        ? { body: pointer.body, x: pointer.world.x, y: pointer.world.y }
        : null;
    send({
      started: started && document.hasFocus(),
      goal: goalId,
      policy: started && getPolicyOn(),
      manual: started ? getManualForce() : 0,
      grab,
    });
    onFrame({
      state,
      tips: plant.tipPositions(state, constants),
      pointer,
      force,
      goalId,
      policyOn: Boolean(info.policy),
      visual: null,
    });
  }

  connect();
  raf = requestAnimationFrame(frame);
  return {
    stop() {
      stopped = true;
      cancelAnimationFrame(raf);
      ws?.close();
    },
    getState() {
      return state;
    },
    setState(next) {
      if (typeof next === "string") send({ started: true, reset: next });
    },
    stats() {
      return info;
    },
  };
}
