# SPEEDUP_NOTES.md — coarse-to-fine (SPEED) integration into Self-Forcing

How the local SPEED sampler (`/mnt/data/SPED/speed`, arxiv 2605.18736) drops into
Self-Forcing's per-chunk denoise loop, what the fixed-spatial-token assumption
breaks, and whether to target the distilled vs non-distilled model.

## Where it slots in

The per-chunk few-step denoise loop is `pipeline/causal_inference.py:188-221`:
4 steps over `denoising_step_list=[1000,750,500,250]`, each step a single
`generator(...)` forward at the **full** latent resolution (60×104).

SPEED's `generate()` (`speed/latent_video_gen.py:169-248`) replaces a flat
denoise loop with a **staged** one: run the early steps at a smaller spatial
scale (e.g. 0.5 → 30×52), then `spectral_expand_and_align()`
(`speed/utils.py:195-249`) upsamples the latent in DCT/DWT/FFT space (high freqs
filled with noise) and realigns the flow-matching timestep, then continue at the
next scale. Temporal dim is held fixed across stages — already video-aware.

Drop-in target: the body of `causal_inference.py:188-221`. Replace "4 steps at
full res" with "k steps at low res → spectral expand+align → remaining steps at
full res", reusing `speed/utils.py` verbatim (DCT path, no extra deps beyond
scipy/pywt).

## What the fixed-spatial-token assumption breaks (the crux)

In the **non-AR** WAN base (what SPEED currently drives), nothing breaks: each
forward recomputes `seq_len = ceil(H*W/4 * T)` per call
(`speed/latent_video_gen.py:106-116, 137`), there's no persistent cross-step
state, so changing H×W mid-trajectory is free.

In **Self-Forcing (AR)** two structures hard-code the per-frame spatial token
count `frame_seq_length = 1560` (= 60·104/4):

1. **Self-attn KV cache indexing.** Cache size, eviction math, and read window
   are all in units of `frame_seqlen` (`wan/modules/causal_model.py:194-235`;
   `causal_inference.py:283-288`, `frame_seq_length=1560` at `:34`). A chunk
   denoised at 0.5 scale has 390 tokens/frame, not 1560 — it cannot be written
   into, or attend against, a cache laid out for 1560. The past frames in the
   cache are full-res; a low-res query has a different token grid.

2. **RoPE positional indexing.** `causal_rope_apply(q, grid_sizes, freqs,
   start_frame=current_start//frame_seqlen)` (`causal_model.py:194-199`) derives
   frame position from `current_start // 1560`. At a different token-per-frame
   count this division mislabels positions, and the rope grid (H,W) differs from
   the cached keys' grid → cross-resolution attention is geometrically
   inconsistent.

**Consequence.** Coarse-to-fine *within* the AR self-attention is not a free
drop-in: the low-res sub-steps of a chunk would have to attend to a full-res
history. Options, cheapest first:
- **(a) Low-res only on the parts that don't hit the KV cache.** The diffusion
  refinement of the *current* chunk's own tokens can be multi-res while the
  cross-attn to text (512 tokens, resolution-independent) and the *write-back*
  to the KV cache happens once at full res after the final step (the clean-context
  pass `causal_inference.py:226-235` already re-runs at full res). Needs care: the
  intermediate low-res steps still read the full-res KV history.
- **(b) Per-resolution KV caches / re-derived rope.** Maintain the history at the
  chunk's working resolution, or downsample the cached K/V to match. More
  faithful, more surgery.
- **(c) Apply SPEED to the non-distilled WAN base path** and distill the speedup
  back — clean but heavier, and re-opens the train/test gap Self-Forcing closed.

## Distilled vs non-distilled — where's the headroom?

- **Distilled Self-Forcing: 4 steps/chunk.** Very little room — a 2-stage
  coarse-to-fine wants ≥1–2 steps at low res before transitioning; at 4 total
  steps the spectral transition cost (DCT round-trip per step boundary) may eat
  the savings, and quality margin is thin. SPEED's own WAN config uses **50
  steps** (`speed/configs.yaml: wan21.defaults.n_steps: 50`) — that's the regime
  it was tuned for.
- **Non-distilled WAN 2.1 base: 50 steps.** Lots of room; SPEED already shows
  2–8× there. But it is not AR/streaming and not real-time.

**Recommendation for this pass:** prototype the coarse-to-fine transition on the
**non-distilled WAN base** first (using the existing `speed/latent_video_gen.py`,
which already runs) to validate spectral expand+align on WAN latents and measure
the speedup envelope. Separately, treat the distilled AR model's 4-step budget as
the hard constraint and investigate a **2-step-low + 2-step-full** schedule with
option (a) above. Causal Forcing++ (1–2 step) has even less headroom — for the
speedup, a *moderately* distilled variant (8–12 steps) may be the sweet spot:
enough steps for a multi-res schedule, still far cheaper than 50.

## Concrete next experiments
1. Run `speed/latent_video_gen.py --scales 0.5 1.0` vs `--scales 1.0` on the WAN
   base; record wall-clock + quality. (Needs `WAN_PATH`/`WAN_CKPT` env → can
   reuse the weights already in `gino/Self-Forcing/wan_models`.)
2. Microbench the DCT expand+align cost vs one DiT forward at 480p to see if a
   transition is amortized within a 4-step budget.
3. Prototype option (a) in a Self-Forcing fork; check whether low-res-query →
   full-res-KV attention degrades gracefully or artifacts.
