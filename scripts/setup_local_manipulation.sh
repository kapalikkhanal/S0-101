#!/usr/bin/env bash
# One-time network setup. Normal manipulate.py execution stays local/offline.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v ollama >/dev/null; then
  echo 'Install Ollama from https://ollama.com/download, then rerun this script.' >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python 3.12
fi
if command -v uv >/dev/null; then
  uv pip install --python .venv/bin/python -c constraints-tested.txt -e '.[manipulation]'
else
  .venv/bin/python -m pip install -c constraints-tested.txt -e '.[manipulation]'
fi
mkdir -p logs/manipulation
if ! .venv/bin/python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2)' >/dev/null 2>&1; then
  OLLAMA_NO_CLOUD=1 nohup ollama serve > logs/manipulation/ollama-server.log 2>&1 &
  for attempt in $(seq 1 30); do
    if .venv/bin/python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2)' >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi
ollama pull qwen3-vl:8b-instruct
# Precache the existing open-vocabulary detector for offline real-camera operation.
.venv/bin/python - <<'PY'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
name='IDEA-Research/grounding-dino-tiny'
AutoProcessor.from_pretrained(name)
AutoModelForZeroShotObjectDetection.from_pretrained(name)
print('Grounding DINO cached.')
PY
.venv/bin/python - <<'PY'
from so101_reach.manipulation.planning import LocalPlanner,plan_dict
p=LocalPlanner()
print('Warm planner:',plan_dict(p.plan('Pick the white cube and place it on the white box',['white cube','white box'])))
print('Planner timings:',p.metrics)
PY
.venv/bin/python scripts/manipulate.py --diagnose
echo 'Ready. Run: .venv/bin/python scripts/manipulate.py "Pick the white cube and place it on the white box"'
