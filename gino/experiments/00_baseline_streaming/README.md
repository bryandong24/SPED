# 00 — Baseline & Streaming

Self-Forcing runs end-to-end and streams chunk-by-chunk.

| File | Prompt / setting | Note |
|---|---|---|
| `tokyo_woman_baseline.mp4` | "stylish woman on a Tokyo street" (detailed) | First baseline; 81 frames generated in **6.7s** on one H100 |
| `teddy_bear.mp4` | "teddy bear eating a dog" | Short prompt → lower motion (model prefers long prompts) |
| `dog_jumping_cat.mp4` | "dog jumping onto cat" | |
| `streaming_5s.mp4` | headless chunk-by-chunk streaming | ~14 FPS end-to-end; frames emitted per 3-frame chunk |
| `streaming_18s_rolling_window.mp4` | 24 chunks, 21-frame rolling KV window | Exceeds the 81-frame (5s) default; per-chunk latency plateaus at ~730ms as window fills; visible drift past ~5s |

**Key facts:** chunk = 3 latent frames, 4 denoise steps/chunk, 480×832, native 16 fps.
Default cache = global attention, 21 latent frames (= 81 px frames ≈ 5s); going
longer needs a finite `local_attn_size` (rolling window) so eviction kicks in.
