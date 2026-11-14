# Fully local models (experimental)

The local paths can work, but results are inconsistent and not fully tested or validated.
They are not a dependable replacement for the OpenAI agent. Expect model/tool-call,
perception, grasp, and placement failures. Start in simulation.

## OpenAI-compatible local agent

Install and run a local vision/tool-capable model server, for example Ollama with
`qwen3-vl:8b-instruct` (or a compatible vLLM model). Download weights while online;
then run the model locally. With the included final policies and workspace built:

```bash
ollama pull qwen3-vl:8b-instruct
# Run `ollama serve` if the local service is not already running.
python scripts/ai_agent.py --sim table --object "red cube" --place "white box" \
  --base-url http://127.0.0.1:11434/v1 --model qwen3-vl:8b-instruct --effort none
```

The compatible-server path uses a dummy API key internally; no OpenAI key is required.
Use a loopback endpoint for local execution. Models must support images and compatible
function tools; text-only chat models are insufficient for this agent. GPU/VRAM needs
depend on the model and quantization. Real execution additionally requires all hardware
steps and the `real` extra; switch to `--real` only after validating the simulation path.

## Local planner plus programmed skills

An alternative asks Ollama to plan and executes monitored skills. This path defaults to
IK travel, rather than the reach policy; choose `--controller rl` to use RL travel.
The current implementation still constructs a reach controller in either mode, so it
requires the included exported policy as well. Simulation's default object measurements are ground
truth; they do not exercise real-camera perception.

```bash
bash scripts/setup_local_manipulation.sh   # installs real/manipulation extra and caches models
python scripts/manipulate.py "Pick the white cube and place it on the white box"
# A model-free plumbing check with a narrow rules parser:
python scripts/manipulate.py "Pick the white cube and place it on the white box" --planner rules
```

The setup downloads Ollama and Grounding DINO weights and may start a local Ollama
service if none is responding. `manipulate.py` sets Hugging Face/Transformers offline
mode; runtime real-camera perception needs its cache populated in advance. To cache just
the detector used by real perception:

```bash
python - <<'PY'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
name = 'IDEA-Research/grounding-dino-tiny'
AutoProcessor.from_pretrained(name)
AutoModelForZeroShotObjectDetection.from_pretrained(name)
PY
```

The rules planner is restricted to supported pick/place phrasing. Local language planning
is not evidence of robust object recognition or safe physical manipulation. Fine grasp
and placement remain sensitive to lighting, occlusion, jaw geometry, calibration, and
scene arrangement. Download all dependencies/models first if the target machine must run
offline, and validate that exact model/hardware combination before relying on it.
