#!/bin/bash
# Launch all 6 (example x group) sweeps in parallel across idle GPUs.
cd /data/SPED/gino/Causal-Forcing
PY="../Self-Forcing/.venv/bin/python"
run() { CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=. $PY ../scripts/cf_hybrid.py --example $2 --group $3 ; }

run 6 dog    delay  &
run 7 dog    sweep  &
run 2 car    delay  &
run 5 car    sweep  &
run 6 jungle delay  &
run 7 jungle sweep  &
wait
echo "ALL HYBRID DONE"
