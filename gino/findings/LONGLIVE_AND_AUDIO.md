# LongLive status + Real-time Audio-Steering feasibility

## LongLive: what's set up (2026-06-20)

- `gino/LongLive/` = full `NVlabs/LongLive` repo (train + inference code).
- `gino/LongLive-v1/` = runnable setup reusing our infra: `.venv`→Self-Forcing venv,
  `wan_models`→our Wan2.1 base, `longlive_models`→HF `Efficient-Large-Model/LongLive-1.3B`
  (8.2 GB, downloaded).
- LongLive-1.3B = **`longlive_base.pt` (DMD base) + `lora.pt` (LoRA rank 256)** on
  Wan2.1-T2V-1.3B. Config: `local_attn_size=12, sink_size=3, num_frame_per_block=3`,
  steps `[1000,750,500,250]`. It's the TRAINED version of our recache+window+sink work.
- **Confirmed running:** interactive multi-prompt inference produced a 957-frame (~60s)
  video w/ 5 switches → `LongLive-v1/videos/interactive/rank0-0-0_lora.mp4`. Demo prompts
  are same-subject ACTION changes (poker player), so it shows smooth same-scene switching,
  not a hard scene change.
- Run: `cd LongLive-v1 && bash inference.sh` (or `interactive_inference.sh`), configs in
  `LongLive-v1/configs/longlive_{inference,interactive_inference}.yaml`.

## Text-conditioning interface (the audio hook)

- Encoder = **umT5-xxl** (frozen, ~5.6–6B params). Prompt → `[seq_len≤512, 4096]`
  contextual token-embedding SEQUENCE → cross-attention per DiT block (4096→1536 proj).
- **Run ONCE per prompt; `context` reused as a frozen constant across all denoising
  steps and is fully cacheable/SWAPPABLE mid-rollout with zero per-step cost.** Injection
  point: `LongLive-v1/wan/text2video.py:172-178`, cross-attn `wan/modules/model.py:585`.
- Update granularity = per chunk (3 latent frames ≈ 0.75s video, ~50-100ms wall-clock).
  Mid-chunk swaps are blendy (few steps) — so effective control rate ≈ chunk rate.

## Real-time audio steering — RECOMMENDATION

**Cascade (training-free): streaming ASR → text → existing umT5 → hot-swap `context` per
chunk.** Native fit to the [L,4096] interface, zero video-side training, sub-second
interrupt→response. Streaming a GROWING partial transcript = a series of small prompt
deltas = a natural soft ramp (gentler than hard swaps). umT5 is bidirectional → re-encode
the current partial each chunk (cheap, off critical path), not token-streaming into it.

Models (all CC-BY-4.0 unless noted):
- **Kyutai STT-1B** — true streaming, ~500ms partial, ~8% WER. Best single-stream combo.
- **NVIDIA Nemotron-Speech-Streaming-0.6B** — ~80–160ms chunks, lowest latency.
- **Moshi (Kyutai, 7B)** — full-duplex, 12.5Hz "inner monologue" text stream (~200ms);
  transcribes its OWN speech, but delay-reversal → streaming ASR of user speech. Use if
  you want the system to also talk back / steer from its monologue.
- AVOID Whisper-family here (pseudo-streaming, ~2s+ first-word).

**Path (b) direct audio→[L,4096] umT5 = NOT recommended now.** No prior art maps audio to a
T5/umT5 token SEQUENCE; existing audio→diffusion adapters (AudioToken, GlueGen,
SonicDiffusion) target CLIP pooled space (512/768-d), not umT5. A GlueGen-style projector
retargeted to umT5 via caption-distillation is a ~few-GPU-day, 3–6 week research build —
only worth it for NON-VERBAL acoustic steering (timbre, ambient sound) that ASR text can't
capture.

Sources: Moshi arxiv 2410.00037 · Kyutai STT/DSM 2509.08753 · Nemotron streaming ASR (HF
blog) · AudioToken 2305.13050 · GlueGen 2303.10056. See also [[longlive-kv-recache]].

## IMPLEMENTED (2026-06-20): audio-steered video, Phases 1-2 working

Code in `gino/audio_stream/` (runs on Self-Forcing venv, from LongLive-v1 dir):
- `load_longlive.py` — loads LongLive-1.3B (base+LoRA), reusing the repo loader.
- `streaming_longlive.py` — **PromptBus** (thread-safe current-prompt) + **StreamingLongLive**
  (driveable chunk loop: polls bus between chunks, recaches via LongLive `_recache_after_switch`
  on change). Decouples from the repo's fixed `switch_frame_indices`. Phase-1 scripted test passes.
- `make_narration.py` — gTTS → test narration (3 scene commands + silence gaps).
- `run_audio.py` — faster-whisper ASR → utterances (grouped by silence gap) → audio-time→video-time
  1:1 schedule → drives StreamingLongLive → mp4 (+ narration muxed back on).

**Phase 1 (dynamic injection):** 18.2 FPS (faster than 16fps real-time), recache ~0.5s,
prompt switch at ARBITRARY chunks works (`out/audio_stream/scripted.mp4`).
**Phase 2 (audio→video):** narration "sunny meadow / snowy night / neon city" →
video steered through all 3 scenes in sync (switches at 12.0s, 23.2s; ~0.5s lag after the
spoken command), dog consistent. `out/audio_stream/audio_driven_with_audio.mp4`. 12.9 FPS.

**Dep gotcha fixed:** installing moshi/faster-whisper bumped transformers→5.12 + hf_hub,
breaking the video model's tokenizer. Pinned back: `transformers==4.49.0 tokenizers==0.21.0
huggingface_hub==0.30.2`. faster-whisper + LongLive both import under this set.

**Phase 3 (next, live):** per-chunk streaming decode + frame callback (live display), swap
faster-whisper → Kyutai STT-1B / Moshi for true low-latency streaming + mic, debounce/hysteresis,
word-level prompt growth (the natural ramp), prompt-expander stage (model likes detailed prompts).
