# MODEL_SURVEY.md — AR video-gen substrate selection

Goal: pick one model serving BOTH project directions — (1) coarse-to-fine
"spectral" diffusion speedup + eventual Mac/Metal port, (2) real-time text
interruption via between-chunk text-embedding swap. Direction 2 *requires*
retained text (T5/umT5) cross-attention with its own K/V cache, decoupled from
visual self-attention; chunk-wise + KV-cached + streaming AR.

## Verdict: **Self-Forcing** (#1), with Rolling Forcing / Causal Forcing++ as same-family fallbacks.

All three are Wan2.1-T2V-1.3B-based → a Self-Forcing-first codebase ports to
either with minimal churn. This also lines up with the local SPEED code
(`/mnt/data/SPED/speed`), which already targets WAN 2.1-1.3B for the speedup —
so both directions share VAE / umT5 / DiT block.

| # | Model | Base | Streaming AR + KV | Few-step | Text cross-attn | FPS / GPU | License | Why |
|---|---|---|---|---|---|---|---|---|
| **1** | **Self-Forcing** | Wan2.1-1.3B | yes, rolling KV | **4-step DMD**, 3-frame blocks | **PASS (umT5)**, separate from self-attn KV | ~16 FPS H100 / ~10 FPS 4090, 480p | Apache-2.0 | smallest real-time T2V that keeps T5; 4-step sampler is clearly the dominant cost → ideal for coarse-to-fine; most MPS-portable; already in repo |
| 2 | Rolling Forcing | Wan2.1-1.3B | yes + attention-sink anchor | few-step | PASS | ~16 FPS 1 GPU, multi-min | `TencentARC/RollingForcing` | **explicit native mid-rollout prompt switch** + long-video drift control; strongest on direction-2 if it's the harder axis |
| 3 | Causal Forcing++ | Wan2.1-1.3B | yes | **1–2 step** | PASS | real-time 4090 | `thu-ml/Causal-Forcing` | higher quality than SF; 1–2 step = even more sampler-dominant (better for direction 1); newest — verify weight maturity |
| 4 | Krea Realtime | Wan2.1-**14B** | yes (SF-based) | 4–6 step | PASS | 11 FPS @ B200 | Apache-2.0 | productized interactive prompt-switch, but 14B hurts MPS-friendliness → fallback only |
| 5 | CausVid | Wan-14B | yes | 4-step DMD | PASS ("dynamic prompting") | ~9 FPS | Apache-2.0 | the original Wan-causal distillation SF is built on; superseded for small/real-time |

### Rejected for direction 2 (no usable text channel / not streaming-AR)
- **Matrix-Game 2.0** — image+action only, text branch removed (hard FAIL), even
  though streaming is excellent (25 FPS, 3-step). Structural reference only.
- **FramePack** (i2v, not real-time), **Diffusion Forcing** (no text),
  **LTX-Video** (bidirectional, not chunk-causal — but the most MPS-mature DiT,
  note for the Mac port), **Pyramid Flow** (no KV cache; only documented MPS
  T2V), **Wan2.2 / Open-Sora 2.0** (bidirectional; streaming only via community
  causal distillation).
- **MAGI-1** (4.5B/24B) — clean native chunk-AR + per-chunk T5 prompting, but
  needs ~datacenter scale for real-time; keep as a heavier-model option.

### Bottom line
No single model beats Self-Forcing on the *combined* score. Two successors edge
it on one axis each — Rolling Forcing on native text-swap, Causal Forcing++ on
step-count/quality — and both are drop-in-adjacent. Proceed on Self-Forcing;
revisit Rolling Forcing if the text-swap (direction 2) proves to need a
natively-supported flush.

Key URLs: self-forcing.github.io · arxiv 2506.08009 · github.com/guandeh17/Self-Forcing ·
arxiv 2509.25161 (Rolling Forcing) · github.com/thu-ml/Causal-Forcing ·
huggingface.co/krea/krea-realtime-video · causvid.github.io
