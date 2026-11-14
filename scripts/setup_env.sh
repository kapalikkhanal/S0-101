#!/usr/bin/env bash
# Run from any directory. uv must already be installed.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -c constraints-tested.txt -e '.[agent]'
.venv/bin/python scripts/check_install.py
