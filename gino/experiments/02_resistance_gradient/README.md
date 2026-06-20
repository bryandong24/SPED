# 02 — The Resistance Gradient

How much does accumulated history matter, and does a longer runway let the morph
complete? Same sun→winter-night transition.

| File | What | Result |
|---|---|---|
| `swap_after1f.mp4` | 1-frame blocks, swap after **1** latent frame | **Gradual morph** to cool/winter over ~5s (warmth 99→−20). Coherent, slow. |
| `swap_after2f.mp4` | swap after **2** frames | More resistance (barely moves) |
| `long_3f_plain.mp4` | **20s** clip, 3-frame blocks, swap at 2.5s | ✅ Plain swap **works** given runway — slowly cools/darkens to night, dog preserved |
| `long_1f_plain.mp4` | **20s** clip, 1-frame blocks, swap at 2.5s | ✅ Reaches a clear **moonlit snowy night** (moon, snow) by the end |

**Finding:** resistance scales with retained history; **transition time ≈ rolling-
window flush time (~5s for window=21)**. The short clips in phase 01 simply didn't
have enough post-swap runway. A 20s clip swapped at 2.5s fully transitions — but
slowly (~5s) and a bit unstable mid-morph. This motivates the window-shrink (04) to
make it fast.
