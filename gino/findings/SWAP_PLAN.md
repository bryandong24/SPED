# Prompt-vector swapping — efficient implementation plan + pilots

Transition under test: **"a dog running in a sunny meadow" → "a dog running in a
snowy field on a winter night"**, swap ~halfway through a 5s (7-block, 81-frame)
rollout. 5s fits the default global KV cache (21 latent frames) — no rolling
window needed. Swap at chunk 3 (latent frame 9 ≈ 2.25s) so the response lands
near the middle.

## What "efficient" means here (two axes)

1. **Latency efficiency** — the swap itself must add ~0 to per-chunk time.
   - The ONLY non-trivial cost is the umT5-xxl encode of the new prompt.
     Mitigations: (a) **precompute** the target embedding for known swaps;
     (b) for interactive swaps, run the encode on a side CUDA stream so it
     overlaps the current chunk's denoise. Interrupt→response then =
     (hidden encode) + 1 chunk.
   - Cross-attn K/V recompute = 30 blocks × (k_proj,v_proj on 512×1536). Tiny.
     Triggered lazily by flipping `crossattn_cache[i]["is_init"]=False`; recompute
     happens on the FIRST denoise step of the next chunk, reused for the rest.
     So one reset per swap, not per step — already optimal.
   - Self-attn KV cache + latents untouched → zero cost there.

2. **Transition-quality efficiency** — cheap, training-free levers that make the
   new prompt "take" without artifacts, given the self-attn cache holds
   old-prompt frames (old-world momentum).

## Pilots (iterate)

- **V0 hardcut** — precomputed embed, swap conditional_dict + is_init reset at
  chunk 3. Baseline behavior: morph / resist / artifact? measure chunks-to-effect.
- **V1 crossfade-2 / crossfade-3** — lerp prompt_embeds old→new over N chunks.
  (Caveat: linear interp in umT5 space is not a semantic blend — pilot to see if
  it smooths or muddies.)
- **V2 evict-soft** — after swap, shrink the self-attn history (keep sink + last
  frame, drop the rest) so the new prompt re-anchors faster. Trade: motion
  continuity vs response speed.
- **V3 combo** — crossfade + soft evict, if V1/V2 each help.
- (Noted, not piloted first: re-add an uncond pass to raise CFG on the new
  prompt — the distilled model has no CFG, so this costs a 2nd forward. Only if
  the cheap levers fail.)

## Measurement

- Per-frame **mean luminance** (day bright → night dark) → when does it drop after
  the swap, and how steep. Also compare against a no-swap "all-sun" baseline and
  an "all-night" reference (same seed) to bound the achievable change.
- **Interrupt→response latency** = first pixel frame after the swap whose
  luminance departs >X% from the pre-swap mean.
- Eyeball a frame **contact sheet** per variant for artifacts/morph character.

## Efficiency deliverable

A swap API that: precomputes/【async-encodes】the target, flips is_init once,
optionally applies a chosen lever — adding no measurable per-chunk latency. The
"best" lever chosen from the pilots above.
