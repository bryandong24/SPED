#!/usr/bin/env bash
# Launch the HunyuanVideo-1.5 live demo server on GPUs 1/2 (GPU0 is a teammate's job).
set +e
cd "/data/SPED/controllable world model/minWM" || exit 2
pkill -9 -f "Wan21/live/server.py" 2>/dev/null
sleep 2
export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared"
export GEMINI_DIR=/data/SPED/gemini
PY=/data/SPED/minwm_venv/bin/python
nohup "$PY" Wan21/live/server.py --backbone hy --port 5008 \
  --gen_device cuda:1 --vae_device cuda:2 \
  > /data/SPED/hy_live_server.log 2>&1 &
echo "server launched PID $! (gen=cuda:1 vae=cuda:2, log=/data/SPED/hy_live_server.log)"
exit 0
