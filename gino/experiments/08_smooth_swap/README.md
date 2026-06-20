# 08 — Smoothing the Swap MOMENT (interpolation ideas)

Problem (user): the post-swap video is consistent enough (06c grow-window), but
the **swap itself is an abrupt cut**. Goal: interpolate to make the transition
gradual. 15s, sun→night, swap at 7.5s. Metrics: `cut@swap` = pixel delta right at
the boundary (lower = smoother), `jitter` = post-swap frame-to-frame delta, warmth.

## Methods tried

| File(s) | idea | warmth →end | jitter | cut@swap |
|---|---|---|---|---|
| `int_pfade{3,6,9}_new` | **prompt crossfade**: ramp cross-attn ctx lerp(P_old→P_new) over N frames | weak (resists: 64→38) | 6–14 | 8–11 |
| `int_pfade6_slerp` | prompt crossfade, slerp | 65→18 | 9.1 | 8.6 |
| `int_pfade6_ortho` | prompt crossfade on ortho base | 76→31 (weak) | 11.4 | **6.3** |
| `int_vfade6_vswap` | **VALUE crossfade + K-preserve**: keep K_old (motion), blend V_old→V_new over 6 frames | **72→−42** | **5.26** | 7.7 |
| `int_vfade9_vswap` | value crossfade, 9 frames | **61→−47** | **5.50** | 8.1 |
| `dissolve_d{3,6,9,12}` | **dual-branch latent dissolve**: fork old/new branches, lerp output latents over N frames | weak (79→0) | — | 13–14 |
| `dissolve_d9_beach` | dissolve, beach→storm | 24→−3 | — | **4.8** |
| `dissolve_d6_car` | dissolve, car day→night | −9→−14 | — | 9.9 |

## Findings

- **★ Value-crossfade + K-preserve (`vfade`) is the winner.** Keep cached K_old
  (attention routing = motion), and smoothly blend the cached V_old→V_new over
  ~6–9 frames. Lowest jitter (5.3), full transition to night, and a gradual morph
  through intermediate frames instead of a cut. The clean "structure (K) fixed,
  content (V) interpolated" idea.
- **Prompt crossfade (`pfade`) resists** — ramping the prompt slows compliance
  (the night cache fights the still-sun-ish prompt); it smooths but barely
  transitions. slerp ≈ lerp here.
- **Dual-branch dissolve underperformed** for scene-changing swaps: the two
  branches diverge so much that the linear latent blend lands in a muddy middle,
  and `cut@swap` is actually *higher* than vfade. It only looks smooth when the
  two scenes are similar (beach→storm, cut@swap 4.8). For big scene changes,
  blending two very different latents ≠ a clean morph. (Bug-fixed version: each
  branch keeps its own pure latent, only the output blends.)

**Recommended for a smooth swap:** `swap_interp.py --base vswap --vfade 6 --grow_to 15`
(value-crossfade + K-preserve + grow-window). Caveat: training-free; still some
residual jitter from the 4-step distilled model.
