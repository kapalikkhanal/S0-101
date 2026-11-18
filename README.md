# SO-101 Reach + AI Agent

Train a goal-conditioned PPO controller for the SO-101, reach coordinates in simulation
or on a calibrated follower arm, and use a language/vision agent to compose pick/place
motions. Leader/follower setup and local demonstration recording are included.

The reach policy learns many feasible targets in one training run. At inference it takes
a new target without retraining. “Any point” means a point inside the verified reachable
workspace, joint limits, and your installation's configured bounds. Reaching controls
position with four arm joints; wrist roll and gripper are held during reach training.
General pose control and guaranteed success on arbitrary objects are outside this task.

The OpenAI agent uses the Python SDK with function tools and images. Long travel uses
the exported reach policy; grasp/place also use geometry, IK, and feedback. It is not
end-to-end RL for manipulation. OpenAI execution sends scene images and task/state data
to the API and incurs usage charges. The local LLM alternatives are experimental:
they can work, but are inconsistent and not fully validated.

## Start from a clone

Use Linux with Python 3.12 and `uv`. PPO training and the MjLab interactive viewer need
a supported NVIDIA GPU/driver and a working CUDA PyTorch installation. CPU MuJoCo/ONNX
simulation and real-arm inference do not require a GPU. Rendering needs EGL or a desktop
OpenGL context. The final trained policies are included, so testing does not require
retraining. Secrets, demonstrations, workspace banks, historical tests, and experiment
results are excluded.

```bash
cd so101-reach
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_install.py --agent
python scripts/build_workspace.py
python scripts/check_install.py --policy-dir checkpoints/so101_reach_policy
python scripts/real_run.py --mode sim --target 0.24 0.00 0.16 --num-targets 1
```

No OpenAI key is needed for coordinate reaching. The CPU check and rollout use the
bundled weights immediately after setup and workspace generation. Training is optional:

```bash
python scripts/train.py --seed 42 --num-envs 4096 --iterations 6000
```

Reduce `--num-envs` if GPU memory is limited. See [training](docs/TRAINING.md) for
resume, manual export, evaluation, and hardware dynamics.

| Final weights | Use |
|---|---|
| `checkpoints/so101_reach_policy/` | Simulation reach policy; default coordinate controller/viewer |
| `checkpoints/so101_reach_policy_real/` | Identified-servo fine-tuned policy; default AI agent/local manipulation controller |

Both include ONNX, TorchScript, a matching policy contract, and the final PPO checkpoint
(`model_5999.pt` / `model_9997.pt`). Intermediate checkpoints and training logs are excluded.
The real-arm weights were fitted to the source arm's dynamics; teammates still need their
own hardware calibration and low-speed validation.

## Reach a coordinate

Using the included final simulation policy:

```bash
# MjLab browser viewer; enter x/y/z in metres at http://localhost:8080
python scripts/interactive.py
# CPU MuJoCo/ONNX rollout, no window
python scripts/real_run.py --mode sim --target 0.24 0.00 0.16 --num-targets 1
# Direct ONNX smoke check, no hardware
python scripts/check_install.py --policy-dir checkpoints/so101_reach_policy
```

Coordinates use the robot base frame: +x forward, +z up, z=0 at the mounting surface;
positive shoulder pan rotates toward -y. The policy TCP is the closed-jaw fingertip.
Agent tools use centimetres/degrees and a grip point between the jaws, as specified in
`src/so101_reach/agent/llm.py`. Do not mix those coordinate conventions.

## Run the AI agent

Load your own API key and settings; `.env.local` is ignored by Git:

```bash
cp .env.example .env.local
# Edit .env.local; set OPENAI_API_KEY and optionally SO101_OPENAI_MODEL.
set -a
source .env.local
set +a
python scripts/ai_agent.py --sim table --object "red cube" --place "white box"
python scripts/ai_agent.py --sim stack \
  --task "Stack the red cube on the white cube" --track "red cube" "white cube"
```

Simulation is the default. Choose `--model` for a vision/tool-capable model available to
your API account; the default can be overridden with `SO101_OPENAI_MODEL`.
`--mode pipeline` measures and travels in code; grasp/place first run locally for zero
tokens (parameters cached from earlier model successes in `data/agent_cache/skills.json`,
else a scripted default), and the model is called only when the local attempt fails its
hardware check - verified model recoveries are distilled back into the cache.
`--handoff llm-only` restores model-guided grasp/place for every phase (token baseline),
`--no-cache` disables the cache. `--mode free` lets the model choose each action. `--max-turns 30` bounds turns and
`--max-input-tokens 300000` bounds cumulative input tokens; these are not dollar caps.
Simulation uses ground-truth object detections plus rendered images, so success there
does not validate real-camera perception. Logs/images stay under ignored `logs/`.

### Imitation-learned grasps and Kalman object tracking

The local grasp rung can use a small behavior-cloned model between a cache hit and the
scripted default. It is trained only on grasps that held through a 5 cm lift in randomized
sim scenes; an optional LLM expert labels the scenes the scripted policy fails (DAgger):

```bash
python scripts/gen_grasp_data.py --episodes 200 --seed 0                  # no LLM, no tokens
python scripts/gen_grasp_data.py --episodes 50 --seed 1000 --expert local # optional expert pass
python scripts/train_bc_grasp.py                                          # -> data/il/bc_grasp.json
python scripts/eval_grasp_policy.py --episodes 40 --seed 100000           # held-out: heuristic vs BC
```

When `data/il/bc_grasp.json` exists the agent uses it automatically (`--no-bc` to disable).
`--tracker kf` adds per-object Kalman filters (outlier gating, prediction through arm
occlusion); the arm is then parked out of the head camera's view only when the tracker is
not confident. Evaluate it under simulated camera noise and occlusion:

```bash
python scripts/eval_tracker.py --episodes 20 --seed 200000                # truth / noisy / hold / KF
python scripts/ai_agent.py --sim table --object "red cube" --place "white box" \
  --sim-noise --tracker kf --sim-clear-view                               # parks saved in the summary
PYTHONPATH=src python3 -m pytest tests/test_tracker_bc.py -q              # numpy-only unit checks
```

For real operation, first complete [hardware setup and calibration](docs/HARDWARE.md).
Then, in the main environment with settings loaded, select the included real-arm policy:

```bash
# Coordinate control: --mode live explicitly selects hardware.
export SO101_POLICY_DIR="$PWD/checkpoints/so101_reach_policy_real"
python scripts/real_run.py --mode dry --no-viewer --num-targets 1
python scripts/real_run.py --mode live --nudge 0 0 0.01 --speed 0.1 --duration 2
# Agent control: --real explicitly selects hardware; review the terminal start prompt.
python scripts/ai_agent.py --real --object "red cube" --place "white box" --speed 0.1
```

The real commands do not require retraining; they require your own motor calibration, joint map, workspace bounds, and
(for the agent) camera extrinsics and servo fit. New packaging/configuration changes have
not been exercised on physical hardware. Simulation accuracy is not a hardware guarantee.
The agent holds the arm at exit; support it before `python scripts/real_release.py`.

## Leader/follower and recording

Use the separate LeRobot environment described in [hardware setup](docs/HARDWARE.md).
These scripts are adapted from the companion arm project, including its terminal episode
controls and servo communication retries:

```bash
bash scripts/setup_recording.sh
bash scripts/setup_motors.sh follower   # new/unconfigured motors only; follow prompts
bash scripts/setup_motors.sh leader
bash scripts/calibrate_motors.sh follower
bash scripts/calibrate_motors.sh leader
bash scripts/teleop.sh
bash scripts/record.sh --task "Pick up the red cube" --dataset red_cube --episodes 10
bash scripts/replay.sh red_cube 0
```

Recording: **S** starts, **Space** saves, **Left arrow** discards, **Esc** saves and quits.
Between episodes the follower remains live. Datasets stay under `data/recordings/`;
recording does not upload to Hugging Face. Demonstrations are optional and are not PPO
training inputs. For fully local models, see the short [local guide](docs/LOCAL_MODELS.md).

## Included source

| Path | Purpose |
|---|---|
| `src/so101_reach/sim`, `tasks` | Robot model, workspace generation, PPO task/curriculum |
| `src/so101_reach/real` | Bus, calibration, ONNX controller, cameras, teleop |
| `src/so101_reach/agent` | OpenAI / compatible-server agent and simulation world |
| `src/so101_reach/manipulation` | Experimental local planning, monitored manipulation skills |
| `checkpoints/` | Final simulation and real-arm weights for immediate testing |
| `scripts/` | Training, export, inference, calibration, recording commands |
| `assets/robotstudio_so101/` | Vendored MJCF/meshes with upstream license and revision |
| `configs/sim/` | Fixed numerical fixtures used only by the agent simulator |
| `configs/examples/` | Starting templates to edit for each physical installation |

Use an editable install from this repository: assets/configs live beside `src/` and are
not bundled in a standalone wheel. [Sharing notes](docs/SHARING.md) explain how to create
a clean public repository or a fresh archive. The vendored robot model retains its Apache
2.0 license; no new license has been assigned to the project's original code.
