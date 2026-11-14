"""Portable device settings. Export these variables before starting Python."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM_CONFIG_DIR = ROOT / "configs" / "sim"
POLICY_DIR = Path(os.environ.get("SO101_POLICY_DIR", str(ROOT / "checkpoints" / "so101_reach_policy"))).expanduser()
AGENT_POLICY_DIR = Path(os.environ.get("SO101_POLICY_DIR", str(ROOT / "checkpoints" / "so101_reach_policy_real"))).expanduser()
FOLLOWER_PORT = os.environ.get("SO101_FOLLOWER_PORT", "/dev/ttyACM0")
LEADER_PORT = os.environ.get("SO101_LEADER_PORT", "/dev/ttyACM1")
FOLLOWER_ID = os.environ.get("SO101_FOLLOWER_ID", "my_follower")
LEADER_ID = os.environ.get("SO101_LEADER_ID", "my_leader")
CALIB_ROOT = Path(os.environ.get("HF_LEROBOT_CALIBRATION", str(Path.home() / ".cache/huggingface/lerobot/calibration"))).expanduser()
FOLLOWER_CALIB = Path(os.environ.get("SO101_FOLLOWER_CALIB", str(CALIB_ROOT / "robots/so101_follower" / f"{FOLLOWER_ID}.json"))).expanduser()
LEADER_CALIB = Path(os.environ.get("SO101_LEADER_CALIB", str(CALIB_ROOT / "teleoperators/so101_leader" / f"{LEADER_ID}.json"))).expanduser()


def require_hardware_config(*, cameras=False, servo_fit=False):
    """Fail before engagement if per-robot configuration has not been supplied."""
    paths = [FOLLOWER_CALIB, ROOT / "configs/real_joint_map.json", ROOT / "configs/real_workspace.json"]
    if cameras:
        paths += [ROOT / "configs/head_camera_extrinsics.json", ROOT / "configs/wrist_camera_extrinsics.json"]
    if servo_fit:
        paths += [ROOT / "configs/servo_model_fit.json"]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Configure this robot first; see docs/HARDWARE.md. Missing: " + ", ".join(missing))
