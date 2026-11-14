"""Fit a MuJoCo position-servo model (kp, kv, joint frictionloss) per arm joint to traces
recorded on the real arm by `real_servo_id.py --record`.

Each trace is replayed in CPU MuJoCo from the recorded start pose with the recorded goal
sequence; the fitted joint's simulated trajectory is compared with the measured one.
Writes configs/servo_model_fit.json (used by the 'identified' actuator variant).

  python scripts/fit_servo_model.py logs/real/servo_id_<stamp>.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

from so101_reach.real.joint_map import JointMap
from so101_reach.sim import so101

OUT = so101.PROJECT_ROOT / "configs" / "servo_model_fit.json"


def load_traces(path: Path) -> dict[str, dict[str, list]]:
  d = np.load(path)
  keys = sorted({k.split("|")[0] for k in d.files if "|" in k})
  by_joint: dict[str, dict[str, list]] = {}
  for k in keys:
    j = k.split("/")[0]
    by_joint.setdefault(j, {})[k] = [d[f"{k}|t"], d[f"{k}|cmd"], d[f"{k}|pos"]]
  return by_joint


class Replayer:
  def __init__(self, jmap: JointMap):
    self.jmap = jmap
    self.m = so101.make_cpu_model()
    self.d = mujoco.MjData(self.m)
    m = self.m
    self.qadr = {n: m.jnt_qposadr[m.joint(n).id] for n in so101.ALL_JOINTS}
    self.dadr = {n: m.jnt_dofadr[m.joint(n).id] for n in so101.ALL_JOINTS}
    self.aid = {n: m.actuator(n).id for n in so101.ALL_JOINTS}
    self.base = {
      "gain": m.actuator_gainprm.copy(),
      "bias": m.actuator_biasprm.copy(),
      "fric": m.dof_frictionloss.copy(),
      "damp": m.dof_damping.copy(),
    }

  def set_params(self, params: dict[str, tuple[float, float, float]]) -> None:
    m = self.m
    m.actuator_gainprm[:] = self.base["gain"]
    m.actuator_biasprm[:] = self.base["bias"]
    m.dof_frictionloss[:] = self.base["fric"]
    for n, (kp, kv, fr) in params.items():
      a, dof = self.aid[n], self.dadr[n]
      m.actuator_gainprm[a, 0] = kp
      m.actuator_biasprm[a, 1] = -kp
      m.actuator_biasprm[a, 2] = -kv
      m.dof_frictionloss[dof] = fr

  def replay(self, joint: str, t: np.ndarray, cmd: np.ndarray, pos: np.ndarray) -> np.ndarray:
    m, d = self.m, self.d
    mujoco.mj_resetData(m, d)
    q0 = {n: self.jmap.to_rad(n, pos[0, i]) for i, n in enumerate(so101.ARM_JOINTS)}
    for n in so101.ARM_JOINTS:
      d.qpos[self.qadr[n]] = q0[n]
      d.ctrl[self.aid[n]] = q0[n]
    for n, v in so101.FIXED_JOINT_TARGETS.items():
      d.qpos[self.qadr[n]] = v
      d.ctrl[self.aid[n]] = v
    mujoco.mj_forward(m, d)
    # Let the arm settle on its holding goals (the real arm was holding before each trace).
    for _ in range(100):
      mujoco.mj_step(m, d)
    out = np.empty(len(t))
    k = 0
    tsim = 0.0
    j = so101.ARM_JOINTS.index(joint)
    offset = d.qpos[self.qadr[joint]] - q0[joint]  # Steady-state sag of the model at the start.
    for i, ti in enumerate(t):
      while tsim < ti:
        d.ctrl[self.aid[joint]] = self.jmap.to_rad(joint, cmd[max(k, 0)])
        mujoco.mj_step(m, d)
        tsim += m.opt.timestep
      k = i
      out[i] = d.qpos[self.qadr[joint]] - offset
    del j
    return out


def loss(r: Replayer, joint: str, traces, params) -> float:
  r.set_params({joint: params})
  err = []
  j = so101.ARM_JOINTS.index(joint)
  for t, cmd, pos in traces.values():
    real = np.array([r.jmap.to_rad(joint, p) for p in pos[:, j]])
    sim = r.replay(joint, t, cmd, pos)
    err.append(np.mean((sim - real) ** 2))
  return float(np.mean(err))


def summarize(r: Replayer, joint: str, traces, params) -> list[str]:
  r.set_params({joint: params})
  j = so101.ARM_JOINTS.index(joint)
  lines = []
  for key, (t, cmd, pos) in traces.items():
    real = np.array([r.jmap.to_rad(joint, p) for p in pos[:, j]])
    sim = r.replay(joint, t, cmd, pos)
    goal = r.jmap.to_rad(joint, cmd[-1])
    lines.append(
      f"    {key:28s} final err real {np.degrees(goal - real[-1]):+5.2f} deg sim {np.degrees(goal - sim[-1]):+5.2f} deg | "
      f"rms traj diff {np.degrees(np.sqrt(np.mean((sim - real) ** 2))):.2f} deg"
    )
  return lines


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("traces", type=Path)
  args = p.parse_args()
  r = Replayer(JointMap.load())
  by_joint = load_traces(args.traces)
  fit = {}
  for joint in so101.ARM_JOINTS:
    tr = by_joint[joint]
    base = loss(r, joint, tr, (998.22, 2.731, 0.052))
    best = None
    for x0 in ([np.log(30), np.log(1.0), np.log(0.1)], [np.log(100), np.log(3.0), np.log(0.3)], [np.log(10), np.log(0.5), np.log(0.05)]):
      res = minimize(lambda x: loss(r, joint, tr, tuple(np.exp(x))), x0, method="Nelder-Mead", options={"maxiter": 150, "xatol": 1e-3, "fatol": 1e-9})
      best = res if best is None or res.fun < best.fun else best
    kp, kv, fr = np.exp(best.x)
    fit[joint] = {"kp": float(kp), "kv": float(kv), "frictionloss": float(fr), "rms_deg": float(np.degrees(np.sqrt(best.fun))),
                  "menagerie_rms_deg": float(np.degrees(np.sqrt(base)))}
    print(f"{joint:13s} kp {kp:8.2f} kv {kv:6.3f} friction {fr:6.3f} | rms {fit[joint]['rms_deg']:.2f} deg (Menagerie model {fit[joint]['menagerie_rms_deg']:.2f} deg)")
    for ln in summarize(r, joint, tr, (kp, kv, fr))[:6]:
      print(ln)
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps({"source": str(args.traces), "joints": fit}, indent=2) + "\n")
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
