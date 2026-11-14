# Sharing this package

This directory is a fresh source tree. It contains no Git history from either source
project. Publish this directory, rather than adding the original experiment folder to
a public repository.

Included: runtime source, selected operational scripts, robot MJCF/meshes, fixed numerical
simulation fixtures, generic configuration examples, dependencies, and teammate guides.
Leader/follower and recording scripts derive from the companion `So-101_Arm` project;
`lerobot_run.py` preserves its retries, optional force limit, and terminal episode controls.
The command wrappers were simplified and adapted for configurable devices and local data.

Excluded: `.env.local`, all original Git metadata/history, environments, raw tests and
fixtures, results, handover reports, training logs, recordings, videos, intermediate checkpoints,
workspace/evaluation banks, calibration backups, and actual workstation device IDs.
The simulator retains a small numerical reference profile, separated under `configs/sim/`;
it has no physical-device serials and must not be installed as real-robot calibration.
The final simulation and real-arm policies are included under `checkpoints/`, with their
ONNX/TorchScript exports, matching contracts, and final PPO checkpoints. Teammates can
test without retraining; hardware still needs its own calibration.

## Create a new public repository

Run inside this clean directory:

```bash
python scripts/package_release.py --check
# Initialize here only, not in the original experiment directory.
git init
git add README.md pyproject.toml constraints-tested.txt .gitignore .env.example \
  MANIFEST.sha256 docs src scripts assets configs checkpoints
# Review filenames and content before committing:
git diff --cached --stat
git diff --cached
```

Then commit and push to a repository you choose. The package intentionally has no remote
or inherited history. `.gitignore` excludes future keys, actual hardware JSON configs,
intermediate weights, datasets, and experiment outputs. The two final released weight
directories are explicitly allowed by filename. A `.gitignore` does not remove files already
tracked in some other repository.

## Rebuild a source archive

```bash
python scripts/package_release.py
```

By default this creates `so101-reach-team.zip` beside the repository and includes a fresh
SHA-256 manifest. The archive uses an explicit set of source files/subtrees and rejects
symlinks, suspicious secret patterns, and workstation paths. It includes only the explicitly listed final policies and excludes other runtime outputs
even after you train or record locally. The manifest is over the archived files, excluding
the manifest itself. `--out /path/to/file.zip` chooses an output location.

`--check` audits the exportable source and verifies an existing manifest without writing
an archive. After editing source, rebuild the archive to refresh `MANIFEST.sha256`.
This scan is a useful check of the packaged files; it is not a general secrets audit of
all untracked local material. Keep real keys/calibrations in ignored locations.

The vendored robot model retains `assets/robotstudio_so101/LICENSE` and revision/provenance.
No license was selected for original project code on the owner's behalf; choose one
separately if granting public reuse rights is intended.
