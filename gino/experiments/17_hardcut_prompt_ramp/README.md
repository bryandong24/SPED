# 17 — Principled Fix: Hard Cut + Prompt Ramp

Derived from an architectural diagnosis of WHY the two best methods each fail, then
fixing the better one (hard cut). 15s, swap at 3s, refined prompts.
`scripts/swap_interp.py --base hardcut --pfade N`.

## Architectural diagnosis

The self-attn KV cache is the model's visual memory. At a swap the cross-attn becomes
"night" instantly (clean), but the cache still holds **sun** frames. A smooth,
artifact-free swap needs the cache to be BOTH:
- **(A) consistent** with the cross-attn prompt (else the few-step denoiser gets
  contradictory targets → diffusion artifacts), and
- **(B) slow-changing** (else the displayed frame jumps → abrupt cut).

Each prior method satisfies only one:
- **Hard cut** keeps the real sun frames → (B) gradual, but **violates (A)**: the
  swap chunk must be consistent with sun neighbors (self-attn) AND express night
  (cross-attn); 4 distilled steps can't reconcile → **artifacts**, which then get
  written to the cache (clean-context pass) and **propagate**.
- **Recache** overwrites the cache to night → (A) consistent → smooth after, but
  **violates (B)**: the overwrite is total + instant → **abrupt snap**.

Root cause = one trade: conflict (hard cut → artifacts) vs instant homogenization
(recache → snap).

## The fix

Keep the **pure hard cut** (its graduality is the good part; no recache → no snap)
and remove the *conflict* by making the cross-attn target **lead the cache only
slightly**: ramp the prompt embedding sun→night over N frames (`--pfade N`). Each
frame's cross-attn is then only marginally ahead of what its self-attn cache shows,
so the denoiser never faces a big contradiction → fewer artifacts, while the cache
follows the prompt frame-by-frame → gradual morph. (Distinct from earlier `pfade`,
which sat on top of recache and fought the snap.)

| File | ramp | warmth →end | jitter |
|---|---|---|---|
| `hcp_pfade0` | none (pure hard cut, the artifacty baseline) | 80→16 | 7.32 |
| `hcp_pfade3` | 3 frames | 80→5 | 7.64 |
| `hcp_pfade6` | 6 | 89→15 | 8.14 |
| `hcp_pfade9` | 9 | 88→−7 | **5.15** |
| `hcp_pfade12` | 12 | 94→10 | 6.21 |
| `hcp_pfade18` | 18 | 101→2 | 8.29 |
| `hcp_pfade6_slerp` | 6, slerp | 88→−9 | 6.47 |

`CMP_cut_frames44-68.png` = dense consecutive-frame strip at the cut (rows:
pfade0 / pfade9 / pfade18) for inspecting per-frame artifacts.

## Status

Implemented + run; metrics (jitter/warmth) similar across ramp lengths. The
artifact reduction is a temporal/playback effect best judged by eye — pfade9 had
the lowest jitter. Recommended starting point: `--base hardcut --pfade 9`.
Complementary untested lever from the same diagnosis: give the swap chunk extra
denoise steps to resolve residual conflict. The trained upper bound for this whole
tradeoff is LongLive-1.3B (released; see project notes).
