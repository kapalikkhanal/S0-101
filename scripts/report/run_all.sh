#!/usr/bin/env bash
# Run every report experiment in dependency order, then make the figures. Resumable: each test
# skips work already in results/report/. Logs: results/report/logs/<test>.log
#
#   bash scripts/report/run_all.sh            # full ("large") settings
#   SMALL=1 bash scripts/report/run_all.sh    # quick pass to check the pipeline end to end
#
# Phases (GPU users never overlap: MjLab reach benchmark / PPO training vs the Ollama VLM):
#   A  CPU: workspace bank, BC data + models (t07), reach sim2real (t04), non-LLM manipulation (t01/t02)
#      GPU meanwhile: reach benchmark (t03), then PPO training (t08)
#   B  GPU (Ollama): latency (t05), language (t06), LLM manipulation (t01/t02)
#   C  figures
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOG=results/report/logs
mkdir -p "$LOG"
export MUJOCO_GL=egl
if [[ "${SMALL:-0}" == 1 ]]; then
  BC="--episodes 160 --sizes 10 25 50 --eval 20"; S2R="--n 20"; M1="--n-fast 10 --n-llm 3"; M2="--n-fast 10 --n-llm 0"
  REACH="--num-targets 50 --only nominal"; PPO="--iterations 200 --num-envs 1024"; LAT="--n-llm 3"; LANG="--per-family 3"
else
  BC="--episodes 800 --eval 100"; S2R="--n 150"; M1="--n-fast 100 --n-llm 30"; M2="--n-fast 100 --n-llm 15"
  REACH="--num-targets 250"; PPO="--iterations 1500 --num-envs 2048"; LAT="--n-llm 15"; LANG="--per-family 20"
fi
WORKERS="${WORKERS:-8}"
step() { echo "[run_all] $(date +%H:%M:%S) $1"; }

[[ -f data/workspace/eval.npz ]] || { step "workspace bank"; $PY scripts/build_workspace.py > "$LOG/workspace.log" 2>&1; }

step "phase A: GPU reach benchmark + PPO training in the background"
( ollama stop qwen2.5vl-tools:7b >/dev/null 2>&1; ollama stop qwen2.5vl-tools:3b >/dev/null 2>&1
  $PY scripts/report/t03_reach_benchmark.py $REACH > "$LOG/t03.log" 2>&1
  $PY scripts/report/t08_ppo_training.py $PPO > "$LOG/t08.log" 2>&1 ) &
GPU_PID=$!

step "t07 BC data efficiency";   $PY scripts/report/t07_bc_data_efficiency.py $BC --workers "$WORKERS" > "$LOG/t07.log" 2>&1
step "t04 reach sim2real";       $PY scripts/report/t04_reach_sim2real.py $S2R > "$LOG/t04.log" 2>&1 &
S2R_PID=$!
step "t01 manipulation (non-LLM)"; $PY scripts/report/t01_manip_methods.py $M1 --methods rl_scripted rl_bc --workers "$WORKERS" > "$LOG/t01_fast.log" 2>&1
step "t02 noise sweep (non-LLM)";  $PY scripts/report/t02_noise_sweep.py $M2 --methods rl_scripted rl_bc --workers "$WORKERS" > "$LOG/t02_fast.log" 2>&1
wait $S2R_PID
step "waiting for the GPU jobs (t03, t08)"; wait $GPU_PID

step "phase B: VLA experiments (Ollama)"
step "t05 latency";   $PY scripts/report/t05_latency.py $LAT > "$LOG/t05.log" 2>&1
step "t06 language";  $PY scripts/report/t06_language_generalization.py $LANG > "$LOG/t06.log" 2>&1
step "t01 manipulation (VLA methods)"; $PY scripts/report/t01_manip_methods.py $M1 --methods hybrid vla vla_free > "$LOG/t01_llm.log" 2>&1
step "t02 noise sweep (VLA methods)";  $PY scripts/report/t02_noise_sweep.py $M2 --methods hybrid vla --levels 0.5 1.5 2.0 > "$LOG/t02_llm.log" 2>&1

step "phase C: figures"; $PY scripts/report/plot_all.py | tee "$LOG/plot.log"
step "done"
