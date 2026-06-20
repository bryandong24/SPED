#!/usr/bin/env bash
# Launch the Causal Forcing++ (frame-wise 2-step) click-to-generate streaming UI. Port 5003.
# Uses one GPU. Adjust CUDA_VISIBLE_DEVICES to a currently-free GPU (check `nvidia-smi`).
set +e
cd /data/SPED/spedcopycausalforcing || exit 2
pkill -9 -f "spedcopycausalforcing/demo.py" 2>/dev/null
sleep 2
PY=/data/SPED/gino/Self-Forcing/.venv/bin/python
GPUS="${CUDA_VISIBLE_DEVICES:-7}"
PORT="${PORT:-5003}"
nohup env CUDA_VISIBLE_DEVICES="$GPUS" "$PY" demo.py --port "$PORT" \
  > /data/SPED/spedcopy_ui.log 2>&1 &
echo "UI server launched PID $! (GPU=$GPUS, port $PORT, log=/data/SPED/spedcopy_ui.log)"
echo "open via:  ssh -L $PORT:localhost:$PORT <user>@<box>  then http://localhost:$PORT"
exit 0
