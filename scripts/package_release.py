"""Audit and package source with an allowlist; never include local runtime artifacts."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP = {"README.md", "pyproject.toml", "constraints-tested.txt", ".gitignore", ".env.example"}
CONFIGS = {"real_joint_map.json", "sim_motor_calibration.json", "head_camera_extrinsics.json",
           "wrist_camera_extrinsics.json", "servo_model_fit.json"}
EXAMPLES = {"real_joint_map.json", "real_workspace.json", "attached_hardware.json"}
SCRIPTS = {
    "train.py", "build_workspace.py", "evaluate.py", "export_policy.py", "plot_training.py",
    "interactive.py", "mouse_follow.py", "record_video.py", "real_run.py", "real_inspect.py",
    "real_twin.py", "real_release.py", "real_mouse.py", "real_servo_id.py", "fit_servo_model.py",
    "calibrate_gripper.py", "calibrate_head_camera.py", "calibrate_wrist_camera.py",
    "calibrate_wrist_jaws.py", "real_wrist_capture.py", "real_readonly_check.py",
    "agent_pick.py", "ai_agent.py", "manipulate.py", "teach_grasp.py", "replay_grasp.py",
    "lerobot_run.py", "check_install.py", "package_release.py", "setup_env.sh",
    "setup_local_manipulation.sh", "setup_recording.sh", "hardware_common.sh",
    "setup_motors.sh", "calibrate_motors.sh", "teleop.sh", "record.sh", "replay.sh",
}
RELEASED_WEIGHTS = {
    "so101_reach_policy": {"policy.onnx", "policy.pt", "policy_contract.json", "model_5999.pt"},
    "so101_reach_policy_real": {"policy.onnx", "policy.pt", "policy_contract.json", "model_9997.pt"},
}
DOCS = {"TRAINING.md", "HARDWARE.md", "LOCAL_MODELS.md", "SHARING.md"}
SKIP_PARTS = {"__pycache__", ".git", ".venv", ".venv-lerobot", ".pytest_cache"}
PATTERNS = [
    ("API key", re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}")),
    ("cloud access key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("GitHub token", re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("personal home path", re.compile(rb"/(?:home|Users)/[A-Za-z0-9_.-]+/")),
    ("literal secret assignment", re.compile(rb'''(?im)^[ \t]*(?:OPENAI_API_KEY|OPENAI_KEY|HF_TOKEN|AWS_SECRET_ACCESS_KEY)[ \t]*=[ \t]*["']?[A-Za-z0-9_-]{12,}''')),
]


def selected(rel: Path) -> bool:
    parts = rel.parts
    if any(p in SKIP_PARTS or p.endswith('.egg-info') for p in parts):
        return False
    if len(parts) == 1:
        return rel.name in TOP
    if parts[0] == "checkpoints":
        return len(parts) == 3 and rel.name in RELEASED_WEIGHTS.get(parts[1], set())
    if parts[0] == "src":
        return len(parts) >= 3 and parts[1] == "so101_reach" and rel.suffix == ".py"
    if parts[0] == "scripts":
        return len(parts) == 2 and rel.name in SCRIPTS
    if parts[0] == "docs":
        return len(parts) == 2 and rel.name in DOCS
    if parts[0] == "configs":
        return (len(parts) == 2 and rel.name == "grasp_offsets.json") or (
            len(parts) == 3 and ((parts[1] == "sim" and rel.name in CONFIGS) or
                                 (parts[1] == "examples" and rel.name in EXAMPLES)))
    if parts[0] == "assets":
        return len(parts) >= 3 and parts[1] == "robotstudio_so101" and (
            rel.suffix in {".xml", ".stl", ".png"} or rel.name in {"LICENSE", "README.md", "CHANGELOG.md", "MENAGERIE_COMMIT"})
    return False


def collect(root: Path):
    files = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if path.is_symlink():
            if selected(rel) or rel.parts[0] in {"src", "assets", "scripts", "configs", "docs"}:
                raise ValueError(f"Symlink in source tree: {rel}")
            continue
        if path.is_file() and selected(rel):
            files.append(rel)
    missing = [p for p in TOP if not (root / p).is_file()]
    missing += [f"scripts/{p}" for p in SCRIPTS if not (root / "scripts" / p).is_file()]
    missing += [f"checkpoints/{folder}/{name}" for folder, names in RELEASED_WEIGHTS.items()
                for name in names if not (root / "checkpoints" / folder / name).is_file()]
    if missing:
        raise ValueError("Missing release source: " + ", ".join(sorted(missing)))
    return sorted(files)


def audit(root, files):
    problems = []
    for rel in files:
        data = (root / rel).read_bytes()
        # Meshes and pictures are binary source assets; check text everywhere else.
        if rel.suffix.lower() in {".stl", ".png"}:
            continue
        for label, regex in PATTERNS:
            if regex.search(data):
                problems.append(f"{rel}: {label}")
    if problems:
        raise ValueError("Release audit failed (values withheld):\n" + "\n".join(problems))


def manifest(root, files):
    return ''.join(f"{hashlib.sha256((root / rel).read_bytes()).hexdigest()}  {rel.as_posix()}\n" for rel in files)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--check", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    files = collect(root)
    audit(root, files)
    hashes = manifest(root, files)
    mp = root / "MANIFEST.sha256"
    if args.check:
        if not mp.exists() or mp.read_text() != hashes:
            raise SystemExit("Source manifest missing or stale. Rebuild with scripts/package_release.py.")
        print(f"Release source audit and SHA-256 manifest: OK ({len(files)} files)")
        return
    archive = (args.out or root.parent / "so101-reach-team.zip").resolve()
    if archive.is_relative_to(root):
        raise SystemExit("Place the archive outside the source tree.")
    archive.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(hashes)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in files + [Path("MANIFEST.sha256")]:
            z.write(root / rel, str(Path("so101-reach") / rel))
    print(f"Packaged {len(files) + 1} files: {archive} ({archive.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
