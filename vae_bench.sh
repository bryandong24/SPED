#!/usr/bin/env bash
cd "/mnt/data/SPED/controllable world model/minWM"
run () {
  local name=$1 gpus=$2 sp=$3 dec=$4
  echo "######## $name (sp=$sp decode=$dec) ########"
  CUDA_VISIBLE_DEVICES=$gpus NUM_GPUS=$sp SP_SIZE=$sp HY_SDPA_BACKEND=default MASTER_PORT=$((29700+sp)) \
    bash run_hy_action2v_sp.sh --vae_decode_mode $dec 2>&1 \
    | grep -iE "\[benchmark\]|\[timing\]|out of memory|Traceback|Error:" | sed "s/^/[$name] /"
  pkill -9 -f hy15_inference 2>/dev/null
  sleep 5
}
run SP4_tilepar 0,1,2,3 4 tile_parallel
run SP8_tilepar 0,1,2,3,4,5,6,7 8 tile_parallel
run SP8_leader  0,1,2,3,4,5,6,7 8 leader
echo "######## DONE ########"
