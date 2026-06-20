# minWM live — voice/text → Gemini → camera control

A live browser demo that steers the camera-conditioned **Wan Action2V (4-step DMD)** world
model by voice (push-to-talk) and text. Gemini converts each utterance/command into a
structured camera command; a deterministic controller turns that into camera motion; minWM
generates the steered continuation, streamed back to the browser.

## Files
- `streaming_worker.py` — `MinWMStreamingWorker` (**the server's worker**): persistent
  KV/PRoPE cache (no per-chunk recache), decode only the new chunk (VAE temporal cache),
  generator and VAE on separate GPUs. `start()` / `gen_step()` / `decode_step()`.
- `worker.py`   — `MinWMWorker` (Option A, used by `driver.py`): loads the pipeline+DMD
  checkpoint **once**; `bootstrap()` + `step()` continue via the native `initial_latent`
  path (recache-per-chunk). Simpler but ~3x slower; kept for headless tests/reference.
- `camera.py`   — `CameraController`: velocity state `{forward,strafe,turn,pitch,up,speed}`
  → per-frame motion dicts → viewmats/Ks, reusing `wan_utils.camera_trajectory`.
- `planner.py`  — `VoicePlanner`: text/audio → `WorldCommand` JSON via Gemini `ask_json`
  (keyword fallback offline). Imports the client from `/mnt/data/SPED/gemini`.
- `server.py`   — Flask+SocketIO server: `CommandBus` (latest-wins) + producer/emitter
  threads; events `start` / `utterance` (PCM) / `command` (text) / `stop`.
- `templates/live.html` — video canvas, hold-to-talk mic, text steer box, parsed-command panel.
- `driver.py`   — headless scripted driver (no browser) for quick worker/planner tests.

## Backbones
The server runs either backbone via `--backbone` (same camera/voice/keyboard controls):
- **`wan`** (default) — Wan2.1 T2V: seed is a **text prompt**; ~1.3B; ~1.6 s/chunk.
- **`hy`** — HunyuanVideo-1.5 Action2V: **I2V**, seed is an **image + caption** (pick from
  `assets/example.json`'s 18 entries in the UI); ~8B; ~6 s/chunk (much slower). Files:
  `hy_worker.py` (lifts `HY15/hy15_inference.run_inference_rollout` into start/gen_step/
  decode_step; persistent vision KV cache capped at `--cap` latent frames while `--sink`
  pins the seed anchor; incremental VAE decode; reuses `CameraController` with HY intrinsics).

## Run
```bash
cd "/mnt/data/SPED/controllable world model/minWM"
export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared"
# Wan (fast): generator on one GPU, VAE on another (overlapped) -> ~1.6 s/chunk
CUDA_VISIBLE_DEVICES=4,5 .venv/bin/python Wan21/live/server.py \
    --backbone wan --port 5019 --gen_device cuda:0 --vae_device cuda:1
# HY (image-seeded; needs ~35 GB free at start for the 16 GB transformer + encoders)
CUDA_VISIBLE_DEVICES=4 .venv/bin/python Wan21/live/server.py \
    --backbone hy --port 5019 --gen_device cuda:0 --vae_device cuda:0 --cap 24 --sink 1
# Wan, 1 GPU: ~2.4 s/chunk
CUDA_VISIBLE_DEVICES=4 .venv/bin/python Wan21/live/server.py --backbone wan --port 5019
```
From a laptop:  `ssh -L 5019:localhost:5019 <box>`  then open `http://localhost:5019`.
Click **Start scene**, then **hold** 🎤 to speak ("turn left and go forward", "stop",
"look up") or type a command. (Port 5008 is taken by another demo on this box — use 5019.)

Headless checks (no browser):
```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python Wan21/live/streaming_worker.py --chunks 20   # latency + rollout
CUDA_VISIBLE_DEVICES=4 .venv/bin/python Wan21/live/driver.py --steps 6                # Option-A worker
.venv/bin/python Wan21/live/planner.py "turn left and go forward"                     # parsed JSON
```

## Latency (single H100, 480×832, 4 latent frames/chunk = 1 s of video)
| config | gen | decode | wall/chunk | speedup |
|---|---|---|---|---|
| Option A (`driver.py`) | 1.4 s | 3.1 s (full window) | **5.1 s** | 1.0x |
| streaming, 1 GPU | 1.6 s | 0.8 s (new only) | **2.4 s** | 2.1x |
| streaming, 2 GPU (overlap) | 1.6 s | hidden | **~1.6 s** (~10 fps) | **3.2x** |

The emitter paces playback to the generation rate (backpressure) so it streams smoothly,
just below realtime (~10 fps). `gen` ramps over the first ~5 chunks as the 20-frame KV
window fills, then plateaus.

## torch.compile — tried, NOT used
`--compile` works (via the space-free python, below) but gives **no speedup** on this model
(gen 1628 ms vs 1620 ms — it's SDPA/attention-bound with dynamic cache control flow) and adds
multi-second recompile stalls. Left off by default.

Note: the repo path has a space ("controllable world model"), which breaks inductor's C++
link. The venv's real bytes were moved to `/mnt/data/SPED/minwm_venv` and `.venv` symlinks to
it; run via `/mnt/data/SPED/minwm_venv/bin/python … --compile` to get a space-free torch path
if you ever want to experiment with compile.

## Next (not yet built)
- Prompt/style steering: port `gino/scripts/swap_*.py` recache / split-recache onto the camera
  pipeline (`streaming_worker.set_prompt` already does a hard-cut swap).
- SPEED progressive-resolution acceleration (see `gino/findings/SPEEDUP_NOTES.md`).
