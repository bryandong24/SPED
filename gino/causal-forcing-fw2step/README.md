# Causal Forcing++ — Frame-wise 2-step (live + UI)

Lightweight workspace to run the **frame-wise 2-step** Causal Forcing++ model
(`num_frame_per_block=1`, `denoising_step_list=[1000,500]`, with a **4-step first chunk**
`[1000,750,500,250]` — the ASD first-frame trick), alongside the existing chunk-wise 4-step
setup. It is the frame-wise sibling of `gino/audio_stream/` (live) and
`gino/Causal-Forcing/demo.py` (UI).

## Design: lightweight, references the shared repo
This folder holds **only the glue**. All heavy code (model, pipeline, `wan/`, VAE), the
`wan_models/` symlink, and the `causal_forcing_dmd_framewise_2step.yaml` config live in the
shared **`../Causal-Forcing`** repo and are imported via `sys.path` + `os.chdir(CF_DIR)`.
No repo clone, no duplicated model code.

- `cf_streaming.py` — frame-wise 2-step streaming generator (`StreamingCF`), robust
  FSDP/EMA checkpoint loader, **first-chunk 4-step schedule applied in `step()`**, and
  `hardcut()` / `ramp_to()` prompt switching. Reused by both servers.
- `web_live_cf.py` — LIVE audio/text-steered server (mic → faster-whisper ASR →
  **Gemini** prompt refinement → PromptBus → continuous rollout; forward SLERP prompt ramp).
- `demo.py` — click-to-generate streaming UI (Flask + Socket.IO; matches `demo.html`).
- `live_pipeline.py` — `RollingBuffer` / `PromptDebouncer` (trimmed; no LongLive deps).
- `templates/` — `live.html` (audio steering) and `demo.html` (click-to-generate).

## Model / weights
- Checkpoint: `../Causal-Forcing/checkpoints/causal-forcing++/framewise-2step.pt`
  (download once: `hf download zhuhz22/Causal-Forcing causal-forcing++/framewise-2step.pt --local-dir checkpoints`).
- Base model (present, symlinked): `../Causal-Forcing/wan_models/Wan2.1-T2V-1.3B`.
- Python env: `/data/SPED/gino/Self-Forcing/.venv/bin/python`.

## Run
Pick free GPUs first with `nvidia-smi` (GPUs 0/1 are used by other demos).

```bash
# Headless smoke test (one GPU)
CUDA_VISIBLE_DEVICES=5 /data/SPED/gino/Self-Forcing/.venv/bin/python cf_streaming.py --seconds 6

# Live audio/text-steered server (video=cuda:0, asr=cuda:1), port 5013
CUDA_VISIBLE_DEVICES=5,6 ./launch_live.sh

# Click-to-generate UI, port 5003
CUDA_VISIBLE_DEVICES=7 ./launch_ui.sh
```

Reach a server from a laptop: `ssh -L <port>:localhost:<port> <user>@<box>` then open
`http://localhost:<port>`.

Note: `demo.html` still shows the upstream torch.compile / FP8 / TAEHV toggles; `demo.py`
ignores them (generation is backed by the frame-wise `StreamingCF` path). FPS/Duration work.
