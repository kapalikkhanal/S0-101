"""Check dependencies/assets without opening devices or making model/API calls."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", action="store_true", help="also compile the CPU manipulation scene")
    p.add_argument("--policy-dir", type=Path, help="also run a short CPU ONNX reach")
    args = p.parse_args()
    import mujoco
    import numpy as np
    import torch
    from so101_reach.sim import so101
    expected_root = Path(__file__).resolve().parents[1]
    if so101.PROJECT_ROOT != expected_root:
        raise RuntimeError("A different checkout is installed. Run pip install -e . in this checkout.")
    print("Python packages:", {n: metadata.version(n) for n in ("mjlab", "mujoco", "torch", "onnxruntime")})
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("CPU simulation/inference available; PPO training requires a supported NVIDIA GPU.")
    m = so101.make_cpu_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert m.njnt == 6 and np.isfinite(d.qpos).all()
    print("Robot assets: OK; joints:", m.njnt)
    import mjlab.tasks  # task plugins are discovered on import
    from mjlab.tasks.registry import list_tasks
    assert {"So101-Reach", "So101-Reach-Real"}.issubset(list_tasks())
    print("Training task registration: OK")
    if args.agent:
        from so101_reach.agent.sim_world import SimWorld, scene
        world = SimWorld(scene("table"))
        assert world.jmap.joints and world.fcal
        world.close_renderers()
        print("CPU agent scene and simulation-only calibration: OK")
    if args.policy_dir:
        from so101_reach.real.controller import ReachController, SimPlant, rollout
        ctrl = ReachController(policy_dir=args.policy_dir)
        q0 = np.array([so101.HOME_JOINT_POS[n] for n in so101.ARM_JOINTS])
        tr = rollout(ctrl, SimPlant(q0, quantize=True), np.array([0.24, 0.0, 0.16]), 250)
        assert np.isfinite(tr["err"]).all()
        print("CPU ONNX rollout: OK; final error [mm] =", round(float(tr["err"][-1] * 1000), 2))
    print("Install check passed. No hardware or remote APIs were accessed.")


if __name__ == "__main__":
    main()
