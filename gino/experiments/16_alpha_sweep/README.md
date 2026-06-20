# 16 — Prompt (Cross-Attn) Scale Sweep on a Plain Hard Cut

Plain hard cut (no recache/window tricks), swap at 3s in a 15s clip, scaling ONLY
the cross-attn (prompt) output by alpha post-swap; self-attn left at 1.0.
`swap_long.py --text_scale α --frame_scale 1.0 --recache 0 --swap_frame 12`.

| α | warmth →end | result |
|---|---|---|
| 0.5 | 82→−122 | **artifact blow-out** (blue/purple psychedelic; weakening prompt → incoherence) |
| 1.0 | 71→11 | **resists** — warm meadow + mild sunset, no snowy night |
| 1.5 | 87→11 | washes toward gray |
| 2.0 | 93→5 | **flat gray mush** (structure locked, color over-pushed) |
| 3.0 | 88→11 | gray/neutral |
| 4.0 | 73→11 | gray/neutral |

**Finding:** scaling the prompt cross-attn on a plain hard cut CANNOT cleanly drive
the swap at any α — resist (α=1) → gray-mush (α>1) → artifacts (α<1). The self-attn
cache locks scene STRUCTURE; cross-attn only re-tints within it. Confirms phase 03
and motivates the recache family (which rebuilds the structure-bearing cache).
The warmth metric is misleading here: α=0.5's −122 is blue garbage, not night.

## Fine sweep around 1.0 (`fine_a*`)

α ∈ {0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4}, same hard cut @3s. All give a gradual
**day→dusk shift on the SAME meadow** (structure locked); α only modulates degree:
- α≈0.8–1.0: mild cooling, stays warm meadow
- **α≈1.2: sweet spot** — pleasant sunset→dusk gradient, consistent dog, not degraded
- α≈1.3–1.4: more desaturated, heading toward gray-mush
None re-author into snowy night. **Conclusion:** prompt-cross-attn scaling on a hard
cut can RE-LIGHT / re-time-of-day a scene but cannot change its STRUCTURE. α≈1.2 is
the pick for same-structure (lighting) transitions; structural changes need recache.
