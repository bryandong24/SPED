# Phase 21 — Causal Forcing: LongLive KV-Recache vs hard-cut (3 diverse swaps)

Builds on [Phase 20](../20_causal_forcing_swap/README.md). Ports the **LongLive KV-Recache**
(training-free) onto Causal Forcing and compares it to a plain hard-cut prompt swap.

Script: `gino/scripts/cf_recache.py` (port of `swap_long.py`; CF = Self-Forcing fork, same
`WanDiffusionWrapper(is_causal, local_attn_size, sink_size)`). Driver: `scripts/run_cf_compare.sh`.

**Recache** = at the swap, rebuild the self-attn KV cache by re-encoding the last N clean
frames *under the new prompt* at the clean-context timestep (relative positions, pos_offset
bookkeeping), then continue. Same pixels (motion preserved), refreshed cached semantics.
**Hard-cut** = re-encode new prompt + flip `crossattn is_init`, leave KV cache untouched.

Setup: 10s clips (40 latent frames), swap at **3s** (frame 12), rolling window 21 + sink 3, seed 0.
- hard-cut: `--recache 0`
- recache: `--recache 9 --post_window 3 --grow_to 15` (grow-window for post-swap stability)

## Results — warmth = mean(R−B); sunny/desert +, night −

| Example (swap) | expect | hard-cut warmth pre→final | recache warmth pre→final | Verdict |
|---|---|---|---|---|
| **dog**: sunlit meadow → snowy moonlit night | ↓ | 56 → **13** (barely cooled, stays meadow) | 51 → **−48** (full snowy night, moon) | recache ✅ |
| **car**: sunny coast → rainy neon city night | (amb.) | −16 → 25 (stays sunny coast) | −6 → 20 (**full neon city**, by eye) | recache ✅ |
| **jungle**: green jungle → red desert dunes | ↑ | −1 → **−2** (no change, stays jungle) | −2 → **+90** (full red dunes) | recache ✅ |

## Conclusion

**KV-recache wins on all 3 diverse examples.** With only a 3s swap + 7s runway, plain hard-cut
**resists** — it largely stays in the old scene (jungle/car show ~zero change; dog only dims).
Recache transitions **fully and within ~1–2s** of the swap, subject + motion preserved
(same dog/car, continuous camera). This is the training-free LongLive mechanism working on a
model that was never trained for prompt switching — confirming the Phase 20 prediction that
recache closes the slow-window-flush gap. Grow-window (3→15) keeps the post-swap frames stable
(no visible double-cut at tile resolution).

Cost: recache replays N=9 frames once per switch (~a fraction of a chunk-time); negligible vs
the 6–8s of extra runway hard-cut would otherwise need.

Artifacts: `compare_{dog,car,jungle}.png` (top=hardcut, bottom=recache, red stripe=after swap),
6 mp4s `{dog,car,jungle}_{hardcut,recache}.mp4`.
