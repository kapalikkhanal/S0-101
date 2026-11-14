# Training and export

The final simulation and real-arm policies are already included. Skip training if you
only want to test: run setup, generate the workspace, and use the README quick-start.
The viewer/evaluation commands fall back to the bundled final simulation checkpoint
when no local training runs exist. Local training outputs take precedence once available.

Run from the repository root after `bash scripts/setup_env.sh` and activating `.venv`.
The setup uses `constraints-tested.txt`, a selected version snapshot from the source
workstation, not a complete lockfile or a guarantee for every GPU. Check
`python scripts/check_install.py --agent` before launching a long run. If CUDA is false
on a training machine, install a CUDA-enabled PyTorch build suitable for its driver/GPU
and rerun the check. Training requires NVIDIA CUDA; real inference uses ONNX on CPU.

## 1. Generate targets

```bash
python scripts/build_workspace.py
```

This creates independent `data/workspace/train.npz` (seed 0) and `eval.npz` (seed 1).
Defaults sample 3 million / 1 million joint configurations, filter floor/self collision
and folded-back poses, then balance the target volume by voxel. Each retained target
has a feasible joint configuration. These are regenerated locally, not downloaded from
previous experiment results. A small installation check can use
`--num-samples 10000 --workers 1`; regenerate at default sizes for actual training.

Targets are position-only, in metres, in the robot base frame. Wrist roll and gripper
are fixed during this task. Edit `WorkspaceSpec` and the task limits if the robot/model
changes; rebuild both banks and retrain after changes to the observation/action contract.

## 2. Train PPO

```bash
python scripts/train.py --seed 42 --num-envs 4096 --iterations 6000
# Short pipeline check (not a useful trained policy):
python scripts/train.py --num-envs 64 --iterations 2 --no-eval
```

Outputs: `logs/rsl_rl/so101_reach/<run>/model_<iteration>.pt`, TensorBoard events,
parameters, and ONNX exports. Curriculum progresses from nearby targets to the broader
workspace, random initial poses, noise/delay/gain variation, then tighter settled accuracy.
A fresh run is not guaranteed to achieve the source project's historical performance.

```bash
tensorboard --logdir logs/rsl_rl/so101_reach
python scripts/plot_training.py
python scripts/train.py --resume-run '<run-directory-name>' \
  --resume-checkpoint model_600.pt --iterations 5400
```

`--resume-run` selects a run under the log root; provide the directory name, not an
absolute path. Checkpoints contain training state/curriculum. Load only trusted training
checkpoints. Inference `policy.pt` is TorchScript, not a resumable PPO checkpoint.

## 3. Evaluate and export

Training without `--no-eval` evaluates native PPO and two IK baselines and exports the
latest new checkpoint. The manual equivalents are:

```bash
python scripts/evaluate.py --controllers native ik ik_joint --tag team_run
python scripts/export_policy.py
python scripts/check_install.py --policy-dir checkpoints/so101_reach_policy
```

Use `--checkpoint logs/rsl_rl/so101_reach/<run>/model_<iteration>.pt` to choose explicitly.
Evaluation creates a held-out suite and writes reports under ignored `results/`.
The export compares ONNX/TorchScript inference to the trained actor. Coordinate commands use `checkpoints/so101_reach_policy`; AI agent/local manipulation
commands default to `checkpoints/so101_reach_policy_real`. Export `SO101_POLICY_DIR`
to select a custom export for both;
`ai_agent.py` and `real_run.py` also accept `--policy-dir`.

## 4. Simulate and enter points

```bash
python scripts/interactive.py                   # MjLab/Viser, normally CUDA
python scripts/interactive.py --viewer native   # desktop MuJoCo window
python scripts/mouse_follow.py
python scripts/real_run.py --mode sim --target 0.24 0.00 0.16 --num-targets 1
```

The last command uses CPU MuJoCo with the bundled or newly exported ONNX policy. Its default dynamics
are Menagerie's, matching default PPO training. `--sim-actuator identified` requires
`configs/servo_model_fit.json`, generated from your physical arm. Targets outside the
sampled workspace may be snapped to a feasible point. Always inspect the reported target.

## 5. Adapt dynamics for hardware

Calibrate the arm and joint map first, following `HARDWARE.md`. The default task uses
Menagerie's servo approximation. The companion `So101-Reach-Real` task uses a measured
fit from your own `configs/servo_model_fit.json`; do not copy simulation fixtures to a
physical robot as calibration.

```bash
# Small physical motions; start in a clear, supported installation.
python scripts/real_servo_id.py --record
python scripts/fit_servo_model.py --help
# Pass your recorded trace(s) as described by that help; inspect the generated fit.
python scripts/train.py --task So101-Reach-Real --num-envs 4096 --iterations 6000
```

You may resume a compatible PPO checkpoint with `--task So101-Reach-Real` and the resume
flags above to fine-tune the dynamics. Keep a separate export directory with
`python scripts/export_policy.py --checkpoint <checkpoint> --out checkpoints/my_real_policy`
and select it using `SO101_POLICY_DIR`. The training wrapper evaluates the selected task
with matching dynamics; when evaluating manually pass `--actuator-model identified`.

Validate with low-speed coordinate nudges before agent tasks. A language planner does
not remove model/calibration errors or provide a general collision guarantee.
