#!/usr/bin/env bash
# Launch the Causal-Forcing++ (framewise-1step) live audio/text-steered demo.
# Video on GPU1, ASR on GPU2 (GPU0 is a teammate's job). Port 5010 (5009 is taken).
set +e
cd /data/SPED/framewise1step/audio_stream || exit 2
pkill -9 -f "framewise1step/audio_stream/web_live_cf.py" 2>/dev/null
sleep 2
PY=/data/SPED/gino/Self-Forcing/.venv/bin/python
# Port 5011 (5009/5010 are teammates' demos). Expose physical GPUs 3,4 (free) ->
# internal cuda:0 (video), cuda:1 (asr_gpu=1).
nohup env CUDA_VISIBLE_DEVICES=3,4 "$PY" -u web_live_cf.py --port 5011 --asr_gpu 1 \
  > /data/SPED/cf_live_server.log 2>&1 &
echo "live server launched PID $! (video=GPU3, asr=GPU4, port 5011)"
exit 0
