# spedcopycausalforcing — Causal Forcing++ (frame-wise 2-step) + SPEED

A **self-contained fork** of `gino/causal-forcing-fw2step` with **SPEED** (Spectral
Progressive Diffusion, https://howardxiao.ca/speed/) wired into both the streaming and
batch inference paths. Training-free: no fine-tuning, just smarter scheduling.

## What this fork is
The original `gino/causal-forcing-fw2step` was *glue* that imported the model from the
shared `gino/Causal-Forcing` repo. SPEED needs model-level edits (RoPE), so this fork
**vendors** the model code (`wan/`, `pipeline/`, `utils/`, `configs/`) and applies SPEED
here, leaving gino's shared repo untouched. The big weights are **symlinked**, not copied:

- `checkpoints/` → `/data/SPED/gino/Causal-Forcing/checkpoints` (11 GB, reused)
- `wan_models/`  → `/data/SPED/gino/Causal-Forcing/wan_models`

## What SPEED does here
Diffusion fixes **low frequencies first**; high-frequency detail only emerges in the
late, low-noise steps. So the early high-noise step(s) of each chunk don't need full
spatial resolution. SPEED runs the **leading `speed_lowres_steps` step(s) at a reduced
latent resolution** (`speed_scale`), then **clean-upsamples the x0 estimate (bicubic)**
and finishes the remaining step(s) at full resolution.

For this **causal, KV-cached, streaming** model the two non-obvious pieces (ported from
`/data/SPED/speed-self-forcing`) are:
1. **RoPE alignment** (`wan/modules/causal_model.py::causal_rope_apply`, `full_hw`): low-res
   query/key tokens are placed on the *full-resolution* positional grid by striding the
   RoPE frequencies, so they stay consistent with the full-res cached context. An explicit
   `current_start_frame` is threaded through (`utils/wan_wrapper.py`,
   `pipeline/causal_inference.py`) because tokens-per-frame changes at low res.
2. **Cache stays full-res**: the clean-context cache-commit pass always runs at full
   resolution, so the autoregressive KV cache is never polluted by low-res tokens.

## Usage

### Streaming (the live/UI path — `cf_streaming.py`, `demo.py`, `web_live_cf.py`)
```python
pipe = load_cf_pipeline(use_speed=True, speed_scale=0.5, speed_lowres_steps=1)
```
Headless smoke test / A-B bench:
```bash
PY=/data/SPED/gino/Self-Forcing/.venv/bin/python
# single run with SPEED
CUDA_VISIBLE_DEVICES=0 $PY cf_streaming.py --speed --speed_scale 0.5 --seconds 6
# baseline-vs-SPEED A/B (FPS + latent MAE + writes two mp4s)
CUDA_VISIBLE_DEVICES=0 $PY bench_speed.py --seconds 4
```
The launch scripts (`launch_ui.sh`, `launch_live.sh`) are unchanged in interface; pass
SPEED through `load_cf_pipeline` in `demo.py`/`web_live_cf.py` (or edit the call) to serve
the accelerated model.

### Batch path (`inference.py`)
Use the ready-made SPEED config (same as the 2-step config but `use_speed: true`):
```bash
CUDA_VISIBLE_DEVICES=0 $PY inference.py \
  --config_path configs/causal_forcing_dmd_framewise_2step_speed.yaml \
  --output_folder output/fw2step_speed \
  --checkpoint_path checkpoints/causal-forcing++/framewise-2step.pt \
  --data_path prompts/demos.txt --use_ema
```

## Knobs
| knob | default | meaning |
|---|---|---|
| `use_speed` | `false` | master on/off |
| `speed_scale` | `0.5` | low-res latent scale in (0,1]; smaller = cheaper + more aggressive |
| `speed_lowres_steps` | `1` | leading steps run at low res; auto-clamped to `len(schedule)-1` so the final step is always full-res |

At `speed_scale=0.5` the latent grid drops from `60×104` to `28×52` (snapped even for
the 2× patchify).

## Measured (frame-wise 2-step, scale 0.5, 1 low-res step, GPU 0, ~4s clip)
| | FPS | wall-clock (denoise) |
|---|---|---|
| baseline | 22.2 | 2.74 s |
| SPEED | 24.3 | 2.52 s |
| **speedup** | | **1.09×** |

Output stays coherent (pixel mean/std 95/83 vs 101/86 baseline; no NaNs). See
`out/speed_vs_baseline.png`.

### Why the speedup is modest here (and how to get more)
This model is **already 2-step and frame-wise** (1 latent frame / 1560 tokens per chunk),
and SPEED only reduces the *query* tokens of the early step — the full-res cached KV window
and the mandatory full-res clean-context commit pass are unchanged. So per chunk only a
fraction of one of three forward passes is cheapened. SPEED's large wins (the paper reports
>2× on WAN 2.1 video, ~7× on FLUX) come from **many-step, full-resolution** sampling where
*all* tokens shrink quadratically. To push the gain here:
- lower `speed_scale` (e.g. `0.4` / `0.33`) — more savings, more quality risk;
- raise `speed_lowres_steps` on the higher-step variants (the 4-step framewise or chunkwise
  models, where the first chunk runs `[1000,750,500,250]` — up to 3 low-res steps);
- it most improves **time-to-first-chunk** (interactive latency), since the first chunk has
  the most steps.

## Files changed vs the original glue
- `wan/modules/causal_model.py` — `full_hw` RoPE striding + `current_start_frame` threading.
- `utils/wan_wrapper.py` — thread `full_hw` / `current_start_frame` to the model.
- `pipeline/causal_inference.py` — SPEED flags, low-res leading steps, `_speed_lowres_dims`,
  `_speed_upsample_x0`.
- `cf_streaming.py` — SPEED in `StreamingCF.step()`; `CF_DIR` now points at this self-contained repo.
- `configs/…_2step.yaml` (+ `…_2step_speed.yaml`) — SPEED knobs.
- `bench_speed.py` — A/B benchmark (new).
