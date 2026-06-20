# Phase 20 — Causal Forcing: does mid-rollout prompt switching change the video?

Substrate: **Causal Forcing** (thu-ml, arxiv 2602.02214, ICML 2026) — a Self-Forcing
*fork* on the same Wan2.1-T2V-1.3B base. `chunkwise/causal_forcing.pt` (5.3G), 4-step DMD,
3 latent frames/block. `wan/modules/model.py` is byte-identical to Self-Forcing, so the
text-swap hook is the same: re-encode the new prompt + flip every
`crossattn_cache["is_init"]=False`; leave the self-attn KV cache (`kv_cache1`) untouched.
Script: `gino/scripts/cf_swap_poc.py`. Setup: `gino/Causal-Forcing/` (reuses Self-Forcing
`.venv` + symlinked Wan2.1 base; no `--use_ema` — chunkwise ckpt has only a `generator` key).

Baseline: 81 px frames @ 480×832, **13.8 FPS** on one H100. Clean motion. ✅

## The question: switch the prompt midway — does CF change, or resist?

| Run | Length | Window | Swap | Result |
|---|---|---|---|---|
| `swap_at3.mp4` | 5s (21 latent, global attn) | global (cap 21f) | chunk 3 (~2.25s) | **RESISTS** — stays sunny meadow; only lighting drifts (warmth 65→38, bright 143→112) + a faint frosted pine at the very end. No snow/night. |
| `swap20s_at3.mp4` | 20s (81 latent, rolling) | **21 + sink 3** | chunk 3 (~2.25s) | **FULLY TRANSITIONS, gradually** — sunny meadow → moonlit snowy night (full moon, frosted pines, snow), dog identity preserved. Warmth 64→−47, bright 136→18. Takes ~6–8s to complete. |

## Conclusion (answers the core question)

**Yes — switching the prompt midway DOES let Causal Forcing change scenes, but only with
enough runway.** The transition speed ≈ the time for the rolling KV window to flush the
old-prompt frames (~6–8s at window=21). With a short runway (5s clip, global attention) the
self-attn KV-cache momentum locks the scene structure and CF only re-lights — it resists.

This is the **same behavior as Self-Forcing** (expected: identical architecture; CF only
changes the distillation). It confirms the gap that KV-recache (Phase 21, next) closes:
recache makes the swap **fast AND smooth** without needing 6–8s of window-flush runway.

Notes: the chunkwise model is trained for global attention on ≤21 latent frames; the 20s
run forces a rolling window (window=21, sink=3) it wasn't trained on — quality holds up well
but a purpose-built long model (`longvideo.pt` + Rolling Forcing pipeline) would be cleaner
for sustained long gen.

Artifacts: `swap_at3_grid.png` (5s before/after), `swap20s_timeline.png` (20s, ~2.5s apart).
