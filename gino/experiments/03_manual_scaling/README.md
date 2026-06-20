# 03 — Manually Scaling Text vs Past-Frame Conditioning

Hypothesis: the swap resists because the **text (cross-attn)** signal is too weak
vs the **past-frame (self-attn)** signal. Scale each pathway directly in every DiT
block, post-swap only. α = cross-attn output scale (louder text), β = self-attn
output scale (quieter past-frame). Swap at chunk 3 (the hard case).

| File | Setting | Result |
|---|---|---|
| `text1p5.mp4` | α=1.5 | Coherent but barely changes — same meadow, slightly dimmer |
| `text2.mp4` | α=2 | More desaturated |
| `text3.mp4` / `text4.mp4` | α=3 / 4 | ❌ collapses to a **flat gray field** — dog/detail dissolve |
| `frame0p6.mp4` | β=0.6 (quieter past) | similar partial effect |
| `text2_frame0p6.mp4` | α=2, β=0.6 | gray mush |
| `cfg6_blowup.mp4` | classifier-free guidance ×6 | ❌ **psychedelic artifacts** (distilled model has no CFG) |
| `wipe_static.mp4` | zero the KV cache at swap | ❌ **colored static** (model needs coherent history) |

**Finding (important):** scaling slides the result *sunny → desaturated → gray
mush*, always on the **same spatial layout**. The self-attn cache locks the scene's
**structure/composition**, not just its color — amplified text can re-tint that
structure but can't re-author it into a different scene (moon, snow). Aggressive
levers (CFG, full wipe) destroy coherence. → motivates keeping the frames but
refreshing their representation (phase 05).
