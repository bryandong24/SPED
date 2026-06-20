# 01 — Naive Swap Resists

Swap the text prompt mid-rollout (re-encode new prompt + flip every block's
cross-attn `is_init=False`). Transition: **sunny meadow → snowy winter night**,
5s clip (7 chunks), swap at chunk 3 (~halfway) unless noted.

| File | What | Result |
|---|---|---|
| `ref_sun.mp4` | all-sun reference | warm orange meadow ✓ |
| `ref_night.mp4` | all-night reference | moonlit blue snow ✓ (target look) |
| `hardcut.mp4` | hard prompt swap at chunk 3 | ❌ **resist** — stays a sunny meadow |
| `crossfade2.mp4` | lerp old→new embeds over 2 chunks | ❌ resist; only the dog breed morphs |
| `evict_soft.mp4` | drop old frames (keep sink+last) | ⚠️ dog morphs toward night breed, scene stays sunny |
| `instant.mp4` | swap **before** any frame (chunk 0) | ✅ **clean night** — no history to fight |
| `early_hardcut.mp4` | swap after the **first** 3-frame chunk | ❌ resist — even ~0.75s of sun locks the scene |

**Finding:** the swap is mechanically perfect (`instant` proves it). The blocker is
the **self-attn KV cache of past frames** — old-world momentum the new cross-attn
prompt can't overcome. Resistance appears with as little as one chunk of history.
See `*_sheet.png` (red top-stripe marks post-swap frames).
