# Physical SO-101 setup

Use a secured SO-101 follower with STS3215 servos and the appropriate controller/power
supply; a leader is optional for autonomous reaching and required for teleoperation.
The camera integration expects a fixed D435i head RGB-D camera and a gripper-mounted
D405. The control rate is 50 Hz. New machines need their own calibration.

These scripts were carried over/adapted from the companion arm project. The release has
been checked without attached devices; the adapted hardware commands still require
physical validation. Run one controller/recording process at a time per servo bus.

## Device settings

```bash
cp .env.example .env.local    # only if you have not already created it
ls -l /dev/serial/by-id/
# Edit .env.local: follower/leader paths, IDs, camera serials, your own OpenAI key.
set -a
source .env.local
set +a
```

The settings file is shell syntax; loading it also supplies variables to Python. Agent
code can read an OpenAI key directly from `.env.local`, but hardware variables must be
exported before Python starts. Use stable `/dev/serial/by-id/...` paths where available.
Ensure your account has access to the serial devices. Connect RealSense cameras over
USB 3 and discover their serials in RealSense Viewer or with:

```bash
# In the main environment, after installing the real extra below:
python -c 'import pyrealsense2 as r; print([d.get_info(r.camera_info.serial_number) for d in r.context().query_devices()])'
```

## Motors, leader/follower, and recording

LeRobot 0.4.4 runs in its own environment to keep its dependency requirements separate
from MjLab. Install once:

```bash
bash scripts/setup_recording.sh
```

For new/unconfigured motors, follow LeRobot's guided connection order. Do not reassign
IDs on an already working assembled arm. Each bus uses motor IDs 1 through 6 for pan,
lift, elbow, wrist flex, wrist roll, gripper.

```bash
bash scripts/setup_motors.sh follower
bash scripts/setup_motors.sh leader
bash scripts/calibrate_motors.sh follower
bash scripts/calibrate_motors.sh leader
```

Calibration files normally appear under:

- `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<follower-id>.json`
- `~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/<leader-id>.json`

The Python controller uses `SO101_FOLLOWER_ID` / `SO101_LEADER_ID` to locate these files;
`SO101_FOLLOWER_CALIB` / `SO101_LEADER_CALIB` override their paths if you imported them.
Changing a calibration JSON does not program homing offsets into the servo: use LeRobot
to calibrate the actual motors, then use the matching file.

```bash
bash scripts/teleop.sh
bash scripts/record.sh --task "Pick up the red cube" --dataset red_cube --episodes 10
# Add episodes to an existing dataset explicitly:
bash scripts/record.sh --task "Pick up the red cube" --dataset red_cube --episodes 5 --resume
bash scripts/replay.sh red_cube 0
```

Record in an interactive terminal: **S** starts, **Space** saves, **Left arrow** discards,
**Esc** quits. The follower stays live while waiting between episodes. Replay commands
move the follower to the episode start pose and play recorded joint actions; keep the
path clear. Replay does not need a leader connected.

If both camera serials are exported, recording includes head/gripper video. Without
them it records joint data only. `SO101_RECORD_CAMERAS` can override the full LeRobot
camera configuration. `DISPLAY_DATA=true` enables its viewer. `SO101_GRIP_FORCE=15`
limits gripper force through the imported wrapper (1–50 percent). Datasets go to
`data/recordings/<name>` and `push_to_hub=false` is fixed in the recording script.
These demonstrations are for collection/replay or a separate imitation learning workflow;
`train.py` trains PPO in simulation and does not ingest them.

## Main environment and joint mapping

Return to the main environment for RL/agent control:

```bash
source .venv/bin/activate
uv pip install --python .venv/bin/python -c constraints-tested.txt -e '.[real]'
python scripts/real_inspect.py
cp configs/examples/real_joint_map.json configs/real_joint_map.json
cp configs/examples/real_workspace.json configs/real_workspace.json
python scripts/real_twin.py
```

`real_inspect.py` reads joint/device state. `real_twin.py` reads the follower without
writing torque or motion commands. Support the arm when moving it by hand. Verify each
joint's direction and angle against the twin. Edit the map's `sign`, `zero_tick`, and
`scale`: `q_rad = sign * scale * (tick - zero_tick) * 2*pi/4096`. Example values are
starting assumptions, not a calibration. Wrist-roll zero and gripper scale often need
special attention. With closed jaws, use the twin's tap check at several table points
to verify the TCP and model/table height.

Edit `configs/real_workspace.json` for your actual tabletop, walls, and fixtures. The
example is a small initial box; it is not derived from your installation. These limits
constrain TCP targets and are not a whole-arm swept-volume guarantee. For camera mounts
and cables not modeled by the MJCF, use
`cp configs/examples/attached_hardware.json configs/attached_hardware.json` and add measured
boxes in the appropriate model body frame for the manipulation monitor/self-filter.

For gripper geometry, measure real jaw gaps and use:

```bash
python scripts/calibrate_gripper.py --help
```

That tool can change the gripper position while measuring and saves fitted zero/scale
into the real joint map. Read its options before operating it. Run the servo
identification/fitting steps in `TRAINING.md` if using identified-servo simulation or
agent manipulation. `AgentArm` uses your fit for gravity compensation.

## Coordinate control, then cameras and agent

After checking all maps/bounds, use the included final real-arm policy (no retraining needed):

```bash
export SO101_POLICY_DIR="$PWD/checkpoints/so101_reach_policy_real"
python scripts/real_run.py --mode dry --no-viewer --num-targets 1
python scripts/real_run.py --mode live --nudge 0 0 0.01 --speed 0.1 --duration 2
# Use absolute coordinates only after verifying frame, table height and limits:
python scripts/real_run.py --mode live --target 0.24 0.00 0.16 --speed 0.1 --duration 4
```

Dry mode reads the real pose and predicts command streams; it does not move the arm.
Live mode enables torque and moves. The arm normally holds its pose on exit.

The camera agent needs measured camera extrinsics, not the fixtures in `configs/sim/`:

```bash
# Attach an AprilTag 16h5, ID 0, rigidly to the gripper for head calibration.
python scripts/calibrate_head_camera.py --manual
# Uses a stationary orange/reference object and the calibrated head camera.
python scripts/calibrate_wrist_camera.py --manual
python scripts/real_readonly_check.py --names "red cube" "white box"
python scripts/ai_agent.py --real --object "red cube" --place "white box" --speed 0.1
```

Manual camera calibration disables torque and asks you to pose the arm; support it.
Extrinsics are written to ignored `configs/head_camera_extrinsics.json` and
`configs/wrist_camera_extrinsics.json`. An optional jaw refinement is available in
`calibrate_wrist_jaws.py`; inspect `--help` before use. The readonly camera check needs
the Grounding DINO model cached. The real agent downloads it on first use if absent;
for an offline preparation step, use the model-cache command in `LOCAL_MODELS.md`.

With a real agent, check detections, table height, and projected jaws before allowing
motion. Keep direct access to the robot's power/stop control. To release torque after a
run, support the arm and run `python scripts/real_release.py`; use `--leader` for the
leader. Releasing torque makes an unsupported arm fall.
