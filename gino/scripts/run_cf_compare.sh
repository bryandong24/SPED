#!/bin/bash
# Hard-cut vs KV-recache across 3 diverse swap pairs on Causal Forcing.
set -e
cd /data/SPED/gino/Causal-Forcing
PY="../Self-Forcing/.venv/bin/python"
COMMON="--config_dummy"  # placeholder
run() { CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. $PY ../scripts/cf_recache.py \
  --total 40 --swap_frame 12 --local_attn_size 21 --sink_size 3 --seed 0 \
  --p1 "$2" --p2 "$3" --name "$1" 2>&1 | grep -E "^\[" ; }

# Example 1: dog meadow -> snowy night
D1A="A fluffy golden retriever sprinting through a sunlit meadow of orange wildflowers, warm golden daylight, lush green grass, cinematic, photorealistic, 4k"
D1B="A fluffy golden retriever running across a deep snowy field on a frigid winter night, full moon, falling snow, frosted pine trees, deep blue moonlight, cinematic, photorealistic, 4k"
# Example 2: sports car coast day -> neon city rain night
D2A="A red sports car driving along a sunny coastal highway by the blue ocean, bright clear daylight, palm trees, cinematic aerial view, photorealistic, 4k"
D2B="A red sports car driving down a rain-soaked neon city street at night, vibrant pink and blue neon reflections on wet asphalt, towering skyscrapers, cinematic, photorealistic, 4k"
# Example 3: drone over jungle -> red desert dunes
D3A="A drone shot flying low over a lush green tropical jungle canopy, misty waterfalls, bright daylight, vivid green foliage, cinematic, photorealistic, 4k"
D3B="A drone shot flying low over vast red sand desert dunes at golden hour, rippling sand, long shadows, arid wasteland, cinematic, photorealistic, 4k"

for mode in hardcut recache; do
  if [ "$mode" = "hardcut" ]; then export EX="--recache 0"; else export EX="--recache 9 --post_window 3 --grow_to 15"; fi
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. $PY ../scripts/cf_recache.py --total 40 --swap_frame 12 --local_attn_size 21 --sink_size 3 --seed 0 $EX --p1 "$D1A" --p2 "$D1B" --name "dog_${mode}"  2>&1 | grep -E "^\["
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. $PY ../scripts/cf_recache.py --total 40 --swap_frame 12 --local_attn_size 21 --sink_size 3 --seed 0 $EX --p1 "$D2A" --p2 "$D2B" --name "car_${mode}"  2>&1 | grep -E "^\["
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. $PY ../scripts/cf_recache.py --total 40 --swap_frame 12 --local_attn_size 21 --sink_size 3 --seed 0 $EX --p1 "$D3A" --p2 "$D3B" --name "jungle_${mode}" 2>&1 | grep -E "^\["
done
echo "ALL DONE"
