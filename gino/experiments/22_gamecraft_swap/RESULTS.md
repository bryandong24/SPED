# 22 — Hunyuan-GameCraft: live demo + prompt-swap results

Substrate: Hunyuan-GameCraft-1.0 (HunyuanVideo 13B MM-DiT, i2v, autoregressive
"hybrid history condition", 8-step PCM distill). Start image `asset/village.png`.
All swaps: "sunny medieval village" → "ruined snowy night village".

## Latency (their speedup, reproduced) — distill + fp8 + DeepCache + sequence-parallel

| Config | s/chunk | gen-FPS | speedup |
|---|---|---|---|
| 1-GPU @ 704×1216 (baseline) | 109 | 0.31 | 1× |
| 4-GPU SP @ 384×672 | 10.0 | 3.4 | 11× |
| **4-GPU SP @ 256×448** | **4.9** | **6.95** | **22×** |

256×448 ≈ paper's reported 6.6 FPS. A chunk = 33 frames ≈ 1.3 s of video, so the
live demo runs ~3.6× from realtime. Token count: 33440 (720p) → 4480 (256×448).
Per-step ~12.4 s/it (1-GPU full-res) → ~3.3 s/it (4-GPU, low-res).

GCP A3-Ultra gotcha: the node sets `NCCL_NET=gIB`; the venv NCCL can't load that
plugin (`network gIB not found`). Fix for single-node SP: `NCCL_NET=Socket`,
`NCCL_IB_DISABLE=1`, unset the gib LD_LIBRARY_PATH/tuner config.

## Live demo
`realtime/web_play.py` + `realtime/templates/play.html`, launched by
`realtime/run_play.sh` (N-way SP under torchrun; rank 0 runs Flask-SocketIO and
broadcasts the per-chunk action/prompt/levers to all ranks). Browser: live video
(base64-JPEG over websocket, client buffers + plays at 25 fps), WASD = move,
arrow keys = look (with on-screen key highlight), prompt box = mid-rollout swap,
sliders = history-attenuation levers. Confirmed: continuous generation at
~7 gen-FPS on 4-GPU@256, controls + swap wired end-to-end.

## Swap result — drastic swaps RESIST (hypothesis H0 confirmed)

Plain mid-rollout prompt swap barely changes the scene: the i2v image-anchor +
the history latents (the previous frames are hard-clamped into the first half of
each chunk every denoise step, `pipeline:961`) keep the sunny village. Warmth
(mean R−B; sunny high, night low) *rose* after the swap (resistance), exactly the
i2v-anchor behaviour predicted — stronger than Self-Forcing, akin to Rolling
Forcing's attention-sink (SF arc phase 18).

## Iterate-to-improve — history-attenuation levers (deterministic A/B)

Added env/UI knobs at the channel-concat injection point (`pipeline:958-965`):
`GC_HIST_SCALE` (scale the clamped history latents), `GC_GT_SCALE` (gt reference
channel), `GC_MASK_SCALE` (1=history mask). Applied post-swap only. Fresh from the
village image, swap at action 3, 256×448 4-GPU. Pre-swap chunks identical (deterministic).

| config | post-swap warmth (chunk 4→8) | reads as |
|---|---|---|
| no lever | 31 → **47** (rises) | resists — stays warm village |
| gt=0, mask=0 | 30 → 39 (rises) | barely helps |
| **hist=0.3, gt=0, mask=0** | 28 → **25** (falls) | ✅ only config that cools toward night |

**Key finding:** the dominant anchor is the **hard-clamped history latents**
(`latents[:,:,:half]=last_latents`), not the gt/mask side channels. Only scaling
that clamp down (`GC_HIST_SCALE<1`) flips the swap from resisting to taking. With
a longer post-swap runway (live test, 9 chunks) hist=0.3 reached warmth ~16 (clear
cooling). Effect grows with runway (H3). Architectural contrast: GameCraft's
history is text-independent pixel latents, so the LongLive semantic-recache lever
does NOT apply — here the only handle is *reducing* history influence (the
GameCraft analog of Self-Forcing window-shrink), with the same resistance↔continuity
tradeoff (hist too low → discontinuity).

(Bounding runs hist=0.0 / hist=0.5 with longer runway: see sheet_h00*/h05* + below.)
</content>
