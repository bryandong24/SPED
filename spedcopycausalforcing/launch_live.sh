#!/usr/bin/env bash
# Launch the Causal Forcing++ (frame-wise 2-step) LIVE audio/text-steered demo.
# Video on the 1st visible GPU (cuda:0), ASR on the 2nd (cuda:1). Port 5013.
# Adjust CUDA_VISIBLE_DEVICES to two currently-free GPUs (check `nvidia-smi`).
set +e
cd /data/SPED/spedcopycausalforcing || exit 2
pkill -9 -f "spedcopycausalforcing/web_live_cf.py" 2>/dev/null
sleep 2
PY=/data/SPED/gino/Self-Forcing/.venv/bin/python
GPUS="${CUDA_VISIBLE_DEVICES:-5,6}"
PORT="${PORT:-5013}"
nohup env CUDA_VISIBLE_DEVICES="$GPUS" "$PY" web_live_cf.py --port "$PORT" --asr_gpu 1 \
  > /data/SPED/spedcopy_live.log 2>&1 &
echo "live server launched PID $! (GPUs=$GPUS [video=cuda:0, asr=cuda:1], port $PORT, log=/data/SPED/spedcopy_live.log)"
echo "open via:  ssh -L $PORT:localhost:$PORT <user>@<box>  then http://localhost:$PORT"
exit 0
