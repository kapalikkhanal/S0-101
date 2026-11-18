"""Batched, controller-agnostic reaching benchmark on independent evaluation targets.

Suite: ``num_targets`` targets drawn uniformly over the reachable volume from the
*evaluation* workspace bank (independent seed, never used for training) crossed with four
initial-configuration families (project home, NVIDIA home, folded rest, random feasible).
Every controller runs the same suite in the same physics for ``horizon_s`` seconds.

Per-episode metrics (errors in mm, times in s):
  final_err        error at the last step
  settled_err      mean error over the last ``settle_window_s``
  success@T        error < T for the whole last ``settle_window_s`` (settled, not a fly-by)
  reach_time@T     first time error < T
  settle_time@T    first time after which error stays < T until the end
  overshoot        max error after first entering 10 mm minus 10 mm (0 if never)
  collision        any ground / self contact with force > 1 N
  soft/hard limit  arm joint beyond 90 % soft limits / beyond hard limits (+1 mrad)
  rms_joint_acc, rms_tcp_jerk, rms_action_rate, path_ratio (TCP path / straight line),
  peak_tcp_speed
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from so101_reach.sim import so101
from so101_reach.sim.workspace import load_bank

THRESHOLDS_MM = (50.0, 20.0, 10.0, 5.0)
NVIDIA_HOME = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0, "wrist_flex": 1.57}
NVIDIA_GOAL_BOX = ((0.08, 0.28), (-0.18, 0.18), (0.05, 0.30))
INIT_FAMILIES = ("home", "nvidia_home", "rest", "random")


@dataclass
class EvalSuite:
  targets: np.ndarray  # (N, 3) base frame [m]
  init_q: np.ndarray  # (N, 4) arm joints [rad]
  family: np.ndarray  # (N,) index into INIT_FAMILIES
  seed: int

  @property
  def in_nvidia_box(self) -> np.ndarray:
    t = self.targets
    m = np.ones(len(t), dtype=bool)
    for k, (lo, hi) in enumerate(NVIDIA_GOAL_BOX):
      m &= (t[:, k] >= lo) & (t[:, k] <= hi)
    return m

  def save(self, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, targets=self.targets, init_q=self.init_q, family=self.family, seed=self.seed)

  @staticmethod
  def load(path: Path) -> EvalSuite:
    d = np.load(path)
    return EvalSuite(d["targets"], d["init_q"], d["family"], int(d["seed"]))

  def __len__(self) -> int:
    return len(self.targets)


def make_suite(num_targets: int = 500, seed: int = 2026) -> EvalSuite:
  bank = load_bank("eval")
  rng = np.random.default_rng(seed)
  tgt = bank.tcp_pos[rng.choice(len(bank), num_targets, replace=False)]
  arm = so101.ARM_JOINTS
  fixed = {
    "home": np.array([so101.HOME_JOINT_POS[n] for n in arm]),
    "nvidia_home": np.array([NVIDIA_HOME[n] for n in arm]),
    "rest": np.array([so101.REST_JOINT_POS[n] for n in arm]),
  }
  targets, init_q, family = [], [], []
  for f, name in enumerate(INIT_FAMILIES):
    if name == "random":
      q = bank.joint_pos[rng.choice(len(bank), num_targets, replace=False)]
    else:
      q = np.repeat(fixed[name][None], num_targets, axis=0)
    targets.append(tgt)
    init_q.append(q)
    family.append(np.full(num_targets, f))
  return EvalSuite(
    np.concatenate(targets).astype(np.float32),
    np.concatenate(init_q).astype(np.float32),
    np.concatenate(family).astype(np.int64),
    seed,
  )


@dataclass
class EpisodeTraces:
  dist: np.ndarray  # (T, N) [m]
  tcp: np.ndarray  # (T, N, 3)
  tcp_speed: np.ndarray  # (T, N)
  q: np.ndarray  # (T, N, 4)
  qacc: np.ndarray  # (T, N, 4)
  action: np.ndarray  # (T, N, 4) clipped
  contact: np.ndarray  # (T, N) bool
  dt: float


@dataclass
class Perturbation:
  """Physics/actuation perturbation applied identically to every controller (robustness)."""

  delay_steps: int = 0
  """Fixed action delay in 20 ms policy steps (incremental-target action term only)."""
  kp_scale: float = 1.0
  kv_scale: float = 1.0
  friction_scale: float = 1.0
  """Arm joint frictionloss scale. Applied by the eval env's reset event (make_eval_env_cfg
  friction_scale); recorded here so every report carries it."""
  deadband_ticks: float = 0.0
  """Servo position dead band [STS3215 ticks, 4096 / rev] (incremental-target action term only)."""

  @property
  def label(self) -> str:
    s = f"delay{self.delay_steps}_kp{self.kp_scale:g}_kv{self.kv_scale:g}"
    if self.friction_scale != 1.0:
      s += f"_fric{self.friction_scale:g}"
    if self.deadband_ticks:
      s += f"_db{self.deadband_ticks:g}"
    return s


TICK_RAD = 2 * np.pi / 4096


def reset_to_suite(
  env, suite: EvalSuite, action_name: str = "arm", command_name: str = "ee_target", perturb: Perturbation | None = None
) -> dict:
  """Place every env at its initial configuration and set its target. Returns observations."""
  dev = env.device
  perturb = perturb or Perturbation()
  cmd = env.command_manager.get_term(command_name)
  cmd.stage.kp_scale_range = (perturb.kp_scale, perturb.kp_scale)
  cmd.stage.kv_scale_range = (perturb.kv_scale, perturb.kv_scale)
  cmd.stage.max_action_delay = perturb.delay_steps
  env.reset()
  robot = env.scene["robot"]
  act = env.action_manager.get_term(action_name)
  arm_ids, _ = robot.find_joints(so101.ARM_JOINTS, preserve_order=True)
  fixed_names = tuple(act.cfg.fixed_targets.keys())
  fix_ids, _ = robot.find_joints(fixed_names, preserve_order=True)
  q = torch.tensor(suite.init_q, device=dev)
  qf = torch.tensor([act.cfg.fixed_targets[n] for n in fixed_names], device=dev).expand(len(q), -1)
  full = torch.cat([q, qf], dim=1)
  ids = torch.tensor(arm_ids + fix_ids, device=dev)
  robot.write_joint_state_to_sim(full, torch.zeros_like(full), joint_ids=ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  act.sync_target_to_measured()
  act._raw_actions.zero_()
  act._clipped.zero_()
  if hasattr(act, "_lag"):
    act._lag.fill_(perturb.delay_steps)
    act._delta_hist.zero_()
  elif perturb.delay_steps:
    raise ValueError("action delay is only supported by the incremental-target action term")
  if hasattr(act, "deadband"):
    act.deadband = perturb.deadband_ticks * TICK_RAD
  elif perturb.deadband_ticks:
    raise ValueError("servo dead band is only supported by the incremental-target action term")
  env.action_manager._action.zero_()
  env.action_manager._prev_action.zero_()
  cmd.set_target(torch.tensor(suite.targets, device=dev))
  env.sim.sense()
  return env.observation_manager.compute()


@torch.no_grad()
def run_suite(
  env,
  policy: Callable[[dict], torch.Tensor],
  suite: EvalSuite,
  horizon_s: float = 5.0,
  action_name: str = "arm",
  perturb: Perturbation | None = None,
) -> EpisodeTraces:
  assert env.num_envs == len(suite), "eval env must have one env per suite episode"
  obs = reset_to_suite(env, suite, action_name, perturb=perturb)
  cmd = env.command_manager.get_term("ee_target")
  act = env.action_manager.get_term(action_name)
  robot = env.scene["robot"]
  arm_ids, _ = robot.find_joints(so101.ARM_JOINTS, preserve_order=True)
  ground, selfc = env.scene["ground_contact"], env.scene["self_contact"]
  steps = int(round(horizon_s / env.step_dt))
  rec: dict[str, list] = {k: [] for k in ("dist", "tcp", "speed", "q", "qacc", "action", "contact")}
  for _ in range(steps):
    a = policy(obs)
    obs, *_ = env.step(a)
    rec["dist"].append(cmd.distance.clone())
    rec["tcp"].append(cmd.tcp_pos_b.clone())
    rec["speed"].append(torch.norm(cmd.tcp_lin_vel_b, dim=-1))
    rec["q"].append(robot.data.joint_pos[:, arm_ids].clone())
    rec["qacc"].append(robot.data.joint_acc[:, arm_ids].clone())
    rec["action"].append(act.clipped_action.clone())
    f = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for s in (ground, selfc):
      fh = s.data.force_history
      f |= (torch.norm(fh, dim=-1) > 1.0).flatten(1).any(dim=1)
    rec["contact"].append(f)
  st = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
  return EpisodeTraces(st["dist"], st["tcp"], st["speed"], st["q"], st["qacc"], st["action"], st["contact"], env.step_dt)


def _first_true(mask: np.ndarray) -> np.ndarray:
  """Index of first True along axis 0 (T if none)."""
  any_ = mask.any(axis=0)
  return np.where(any_, mask.argmax(axis=0), mask.shape[0])


def episode_metrics(tr: EpisodeTraces, settle_window_s: float = 0.5) -> dict[str, np.ndarray]:
  dt = tr.dt
  T = tr.dist.shape[0]
  w = max(1, int(round(settle_window_s / dt)))
  err_mm = tr.dist * 1e3
  out: dict[str, np.ndarray] = {
    "final_err": err_mm[-1],
    "settled_err": err_mm[-w:].mean(axis=0),
  }
  for th in THRESHOLDS_MM:
    below = err_mm < th
    out[f"success@{th:g}"] = below[-w:].all(axis=0)
    out[f"reach_time@{th:g}"] = np.where(below.any(axis=0), (_first_true(below) + 1) * dt, np.nan)
    # Settle time: first index after which `below` holds until the end.
    tail_ok = np.flip(np.cumprod(np.flip(below, 0), axis=0), 0).astype(bool)
    idx = _first_true(tail_ok)
    out[f"settle_time@{th:g}"] = np.where(idx < T, (idx + 1) * dt, np.nan)
  entered = err_mm < 10.0
  first = _first_true(entered)
  after = np.arange(T)[:, None] >= first[None, :]
  out["overshoot"] = np.where(first < T, np.maximum(np.where(after, err_mm, 0).max(axis=0) - 10.0, 0.0), np.nan)
  out["collision"] = tr.contact.any(axis=0)
  model = so101.make_cpu_model()
  hard = np.array([model.jnt_range[model.joint(n).id] for n in so101.ARM_JOINTS])
  mid, half = hard.mean(1), (hard[:, 1] - hard[:, 0]) / 2 * so101.SOFT_JOINT_LIMIT_FACTOR
  q = tr.q
  # Soft-limit excursion beyond max(soft limit, initial excursion): starting poses such as
  # the folded rest pose already lie outside the 90 % soft limits.
  lo_eff = np.minimum(mid - half, q[0])
  hi_eff = np.maximum(mid + half, q[0])
  out["soft_limit_excursion"] = ((q < lo_eff - 1e-3) | (q > hi_eff + 1e-3)).any(axis=(0, 2))
  out["hard_limit_violation"] = ((q < hard[:, 0] - 1e-3) | (q > hard[:, 1] + 1e-3)).any(axis=(0, 2))
  out["rms_joint_acc"] = np.sqrt((tr.qacc**2).mean(axis=(0, 2)))
  vel = np.diff(tr.tcp, axis=0) / dt
  jerk = np.diff(vel, n=2, axis=0) / dt**2
  out["rms_tcp_jerk"] = np.sqrt((np.linalg.norm(jerk, axis=-1) ** 2).mean(axis=0))
  out["rms_action_rate"] = np.sqrt((np.diff(tr.action, axis=0) ** 2).sum(-1).mean(axis=0))
  path = np.linalg.norm(np.diff(tr.tcp, axis=0), axis=-1).sum(axis=0)
  straight = np.linalg.norm(tr.tcp[-1] - tr.tcp[0], axis=-1)
  out["path_ratio"] = np.where(straight > 0.02, path / np.maximum(straight, 1e-9), np.nan)
  out["peak_tcp_speed"] = tr.tcp_speed.max(axis=0)
  return out


def summarize(m: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, float]:
  sel = (lambda x: x[mask]) if mask is not None else (lambda x: x)
  e = sel(m["settled_err"])
  s: dict[str, float] = {
    "episodes": int(len(e)),
    "mean_err_mm": float(np.mean(e)),
    "median_err_mm": float(np.median(e)),
    "p95_err_mm": float(np.percentile(e, 95)),
    "final_mean_err_mm": float(np.mean(sel(m["final_err"]))),
  }
  for th in THRESHOLDS_MM:
    s[f"success@{th:g}mm"] = float(np.mean(sel(m[f"success@{th:g}"])))
  for th in (10.0, 5.0):
    ok = sel(m[f"success@{th:g}"])
    s[f"mean_reach_time@{th:g}mm_s"] = float(np.nanmean(sel(m[f"reach_time@{th:g}"])[ok])) if ok.any() else math.nan
    s[f"mean_settle_time@{th:g}mm_s"] = float(np.nanmean(sel(m[f"settle_time@{th:g}"])[ok])) if ok.any() else math.nan
  s["collision_rate"] = float(np.mean(sel(m["collision"])))
  s["soft_limit_excursion_rate"] = float(np.mean(sel(m["soft_limit_excursion"])))
  s["hard_limit_violation_rate"] = float(np.mean(sel(m["hard_limit_violation"])))
  for k in ("overshoot", "rms_joint_acc", "rms_tcp_jerk", "rms_action_rate", "path_ratio", "peak_tcp_speed"):
    s[f"mean_{k}"] = float(np.nanmean(sel(m[k])))
  return s


@dataclass
class BenchmarkReport:
  controller: str
  overall: dict[str, float]
  by_init: dict[str, dict[str, float]]
  nvidia_goal_box: dict[str, float]
  latency: dict[str, float] = field(default_factory=dict)
  notes: dict[str, str] = field(default_factory=dict)

  def save(self, out_dir: Path, metrics: dict[str, np.ndarray] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(asdict(self), indent=2))
    if metrics is not None:
      np.savez_compressed(out_dir / "episodes.npz", **metrics)


def build_report(controller: str, suite: EvalSuite, m: dict[str, np.ndarray]) -> BenchmarkReport:
  by_init = {name: summarize(m, suite.family == i) for i, name in enumerate(INIT_FAMILIES)}
  return BenchmarkReport(controller, summarize(m), by_init, summarize(m, suite.in_nvidia_box))
