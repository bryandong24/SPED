#!/bin/bash
# Tier-1 smooth-transition ablation: 6 methods x 3 examples on Causal Forcing.
# READY TO LAUNCH when a GPU frees up (currently all 8 are taken). Loads the model
# once per example and loops all methods (--method all).
#
# Usage:   bash run_cf_smooth.sh <GPU_ID>          # all 3 examples on one GPU
#          bash run_cf_smooth.sh <GPU_ID> dog      # single example
set -e
GPU="${1:?usage: run_cf_smooth.sh <GPU_ID> [example]}"
EX="${2:-}"
cd /data/SPED/gino/Causal-Forcing
PY="../Self-Forcing/.venv/bin/python"
run() { CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PY ../scripts/cf_smooth.py \
  --example "$1" --method all --total 40 --swap_frame 12 --transition_chunks 4 \
  --recache_W 9 --window 21 --sink 3 --post_window 3 --grow_to 15 --gain_kind minjerk \
  --out_dir ../out/cf_smooth 2>&1 | grep -E "^\[|^DONE"; }

if [ -n "$EX" ]; then run "$EX"; else for e in dog car jungle; do run "$e"; done; fi
echo "ALL SMOOTH DONE"
