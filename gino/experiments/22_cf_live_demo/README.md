# Phase 22 — Live audio-steered interaction demo on Causal Forcing (hard-cut)

Ports the LongLive live interactive demo (`audio_stream/web_live.py`) onto **Causal
Forcing**, using a plain **HARD CUT** for prompt switching (per request — no recache).

## Files (in `gino/audio_stream/`)
- `cf_streaming.py` — `StreamingCF` (driveable chunk loop) + `load_cf_pipeline()`. Hard-cut
  swap = re-encode new prompt + flip every `crossattn_cache["is_init"]=False`, leave the
  self-attn KV cache; rolling window 21 + sink 3 so it streams indefinitely. Per-chunk
  streaming decode via `vae.decode_to_pixel(den, use_cache=True)` (CF has no
  `decode_to_pixel_chunk`; `use_cache` routes to `cached_decode`). Has a headless smoke test.
- `web_live_cf.py` — Flask + socket.io live server. Browser mic → ASR (faster-whisper
  distil-large-v3, separate GPU) → PromptBus → continuous CF gen loop → JPEG frames back.
  Also type-to-steer (`set_prompt`, bypasses ASR). Prompt switch calls `gen.hardcut(p)`.
- `pentest_cf_live.py` — headless socket.io pentest client.

## Run
```
CUDA_VISIBLE_DEVICES=6,5 .venv/bin/python web_live_cf.py --port 5009 --asr_gpu 1
# from laptop:  ssh -L 5009:localhost:5009 spycoder@<box>  then http://localhost:5009
```

## Pentest results (thorough)
- **Throughput:** ~10–12 FPS streaming, first frame ~1.5–2 s after `start`.
- **Robustness:** stop-before-start, empty `set_prompt`, and double-start are all correctly
  ignored — no server crashes; clean `done` on stop.
- **Type-to-steer:** works great. coast → neon city → desert clearly rendered, red-car
  subject identity preserved across every swap (see `pentest_strip.png` pass 1).
- **ASR path:** faster-whisper transcribes accurately; steering commits drive the scene.
- **Hard-cut behavior:** transitions are GRADUAL (~6–8 s window-flush). Closely-spaced
  commands stay mid-morph; subject identity is preserved throughout (a feature of hard-cut +
  rolling window). Recache would switch faster but the user chose hard-cut.

## Bugs found & fixed during pentest
1. Pentest client `feed` thread emitted after `disconnect` → guarded with `sio.connected`.
2. `PromptDebouncer` default (`stable_ticks=2`, `jaccard=0.6`) suppressed same-subject scene
   changes from ASR (the 8 s rolling transcript never held one candidate stable for 2 ticks).
   Loosened in `web_live_cf.asr_loop` to `stable_ticks=1, jaccard=0.85` → ASR steering now
   commits readily (12 commits in a 28 s narration; warmth crashed to −47 as neon-night
   arrived). Type-to-steer was always unaffected.

Artifacts: `smoke_strip.png` (headless hard-cut transition), `pentest_strip.png` +
`pentest_stream.mp4` (live socket-driven run).
