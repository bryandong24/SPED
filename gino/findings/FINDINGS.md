# FINDINGS.md — real-time text interruption feasibility (running log)

Working dir: `/mnt/data/SPED/gino`. Substrate: Self-Forcing (Wan2.1-T2V-1.3B AR).
Status legend: ✅ done · 🟡 in progress · ⏳ blocked on env · ❓ open.

## Interim verdict (static analysis; empirical pending baseline run)

**Real-time text interruption looks FEASIBLE and the hardest sub-question (Q1)
is answered YES in code.** The architecture decouples text exactly as the brief
hypothesized. Two questions (Q2 transition quality, Q3 few-step headroom) need
the model running — env is being finalized (flash-attn compiling).

## The three deciding questions

1. **Is T5 text context per-chunk-overwritable mid-rollout? → ✅ YES, cleanly.**
   Text K/V is cached per block behind an `is_init` flag
   (`wan/modules/model.py:174-186`); flipping it to `False` + passing new
   `prompt_embeds` re-encodes only the text K/V, leaving the self-attn KV cache
   of past frames and the diffusion latents untouched. The reset path already
   exists in-repo (`causal_inference.py:124-126`). This is a *clean swap*, not
   surgery. Full evidence in `ARCH_MAP.md`.

2. **How OOD is a mid-rollout swap; do cheap tricks fix it? → 🟡 PoC written,
   pending run.** `scripts/text_swap_poc.py` implements hard-cut, cross-fade
   (2–4 chunk embedding interpolation), and post-swap KV eviction. Expectation
   from the architecture: morph rather than snap, because the self-attn KV cache
   carries old-world momentum that the new cross-attn must overcome.
   **Caveat found:** the distilled DMD loop runs ONE forward per step with **no
   classifier-free guidance** (`causal_inference.py:188-221`), so the brief's
   "raise CFG on the new prompt" lever has no native hook in the distilled model
   — it would need an added uncond pass. The two remaining levers (cross-fade,
   KV-window eviction) ARE available.

3. **Few-step headroom for coarse-to-fine? → 🟡 small in distilled model.**
   Distilled Self-Forcing = **4 steps/chunk**
   (`denoising_step_list:[1000,750,500,250]`). SPEED was tuned at 50 steps
   (`speed/configs.yaml`). Recommendation: prototype the speedup on the
   non-distilled WAN base first; target a moderately-distilled (8–12 step)
   variant for the AR speedup sweet spot. Details + what the fixed-token
   assumption breaks (RoPE + KV indexing keyed to `frame_seq_length=1560`) in
   `SPEEDUP_NOTES.md`.

## Task status

- ✅ **Model selection** — Self-Forcing confirmed #1; Rolling Forcing / Causal
  Forcing++ fallbacks (`MODEL_SURVEY.md`). Both project directions converge on
  the WAN 2.1-1.3B family (the local SPEED code already targets it).
- ✅ **Task 0 Setup — DONE, baseline runs.** repos cloned; weights downloaded
  (17G WAN + 5.3G `self_forcing_dmd.pt`); env on uv venv (torch 2.8.0+cu128).
  flash-attn solved via **prebuilt wheel** `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310`
  (<1s install; source build was the wrong path). **Baseline confirmed:** one
  prompt → `out/baseline/0-0_ema.mp4`, 81 px frames @ 480×832, real motion
  (frame Δ=11.4), generated in **6.73 s on one H100** (~12 FPS incl. VAE decode;
  in line with the paper's ~16 FPS). chunk=3 latent frames, 4 denoise steps.
- ✅ **Task 1 Architecture map** — `ARCH_MAP.md` with `file:line` for every
  claim. All brief "established facts" verified; one clarification (swap is via
  `is_init` flip, cleaner than "overwrite K/V").
- ✅ **Task 2/3 Text-swap PoC — RUN, characterized.** Transition tested:
  "dog in sunny meadow" → "dog in snowy winter night", swap at various points
  (scripts `swap_experiment.py`, `swap_experiment2.py`, `swap_firstframe.py`;
  outputs in `out/swap*`). Metric: warmth = mean(R−B) (sunny≈+100, night≈−50;
  luminance is useless because snow-at-night is bright).

  **Core result — a RESISTANCE GRADIENT set by accumulated history:**
  - swap before any frame (`instant`): **clean night**, zero resistance.
  - swap after **1** latent frame (1-frame blocks): **gradual morph** to
    winter over ~5s (warmth 99→−20). Gets there slowly, coherent.
  - swap after **2** frames: more resistance (ends +26, barely moves).
  - swap after a **3-frame chunk** / at the halfway chunk: **strong resist** —
    scene stays a sunny meadow; only the dog's breed morphs.
  Each retained old-prompt frame in the self-attn KV cache adds momentum the new
  cross-attn prompt must overcome. The swap mechanism itself is perfect (proven
  by `instant`); the difficulty is purely the cached-history conflict.

  **Aggressive levers FAIL on the distilled model:**
  - CFG-everywhere (scale 6/10): psychedelic blow-out (model never trained with
    guidance). Warmth went negative but it's blue *artifacts*, not night.
  - Full KV wipe (zero K/V): colored static post-swap — the model needs coherent
    history to attend to.
  So a clean mid-rollout SNAP is not achievable with these training-free tricks;
  consistent with the brief (mid-rollout swap is OOD; clean fine-tune = future work).
  Gentler levers still to try: post-swap-ONLY mild CFG (~1.5–2.5), coherence-
  preserving soft-evict (keep last 1–2 frames), longer cross-fade.

  **★ WORKING METHOD FOUND — post-swap KV-window shrink.** Two keys:
  1. **Long runway**: a plain swap DOES transition if given enough post-swap time
     — the rolling window (size 21) must flush old-prompt frames, which takes
     ~21 frames (~5s). Short clips couldn't; a 20s clip swapped at 2.5s reaches
     moonlit night (`out/swap_long/long_1f_plain.mp4`, `long_3f_plain.mp4`).
  2. **★ Window-shrink = FAST clean swap**: shrink the self-attn read window to
     ~3 frames right after the swap (`set_window` sets each block's
     `self_attn.max_attention_size`/`local_attn_size`). Old frames flush almost
     immediately so the new prompt dominates within ~1-2s, while 3 frames is
     still enough context to stay coherent (vs full-wipe → static). `win3`
     (`out/swap_win/win3.mp4`): sunny meadow snaps to a coherent moonlit snowy
     night and holds. warmth 79→−50 right after swap (ref window-21: 56→14, barely
     moves). win2 even tighter; restoring the window drifts back. Mechanism:
     transition time ≈ window flush time, so the window size is the swap-speed knob.
  Scripts: `swap_long.py` (--post_window/--ramp/--restore_after/--p1/--p2),
  `swap_experiment{,2,3}.py`, `swap_firstframe.py`. Efficiency: swap stays ~free
  (precomputed embeds; window-shrink is just attribute writes).
- ✅ **Task 4 Speedup integration** — `SPEEDUP_NOTES.md`: slot-in point, what
  the fixed-spatial-token assumption breaks, distilled-vs-base recommendation.
- ⏳ **Task 5 Measurement harness** — per-chunk latency + end-to-end FPS +
  interrupt→response instrumentation is built into `text_swap_poc.py`; numbers
  pending the run.

## Env reproducibility notes (gotchas hit)
- uv installs atomically: one un-buildable package (`nvidia-pyindex`,
  `nvidia-tensorrt`, `pycuda` — all TensorRT-only) aborts the whole
  `-r requirements.txt`. Filtered those out (`/tmp/sf_reqs_filtered.txt`); they
  are not needed for baseline inference.
- torch must be a **CUDA-12.x** build (system nvcc = 12.9). The default
  `uv pip install torch` pulled 2.12.1+cu130 → flash-attn build mismatch. Pinned
  `torch==2.8.0 torchvision==0.23.0 --index-url .../cu128`.
- flash-attn source build needs `g++` (build-essential) and `python3.10-dev`
  (Python.h) — installed via apt. Set `CUDA_HOME=/usr/local/cuda-12.9`.

## Open questions / next steps
- ❓ Measured baseline FPS on H100 and per-chunk ms (run baseline).
- ❓ Empirical morph behavior + chunks-to-effect; does cross-fade / eviction
  visibly help; interrupt→response latency.
- ❓ Does low-res-query → full-res-KV attention (SPEEDUP option a) degrade
  gracefully?
- Future work (out of scope this pass): fine-tune to make mid-rollout swaps
  in-distribution; per-resolution KV caches; distill the speedup into the AR model.
