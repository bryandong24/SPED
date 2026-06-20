# 20 — Hunyuan-GameCraft: mid-rollout prompt swap (PLAN)

Substrate: **Hunyuan-GameCraft-1.0** (HunyuanVideo 13B MM-DiT, image+text+action,
autoregressive "hybrid history condition"). New substrate vs the Self-Forcing arc
(00–17) and Rolling Forcing (18–19). Goal: change the scene by swapping the text
prompt midway through the autoregressive rollout, and characterize what makes the
swap take / resist — reusing the playbook we already built.

## What is structurally DIFFERENT here (drives every hypothesis)

| | Self-Forcing (00–17) | GameCraft (this) |
|---|---|---|
| Task | T2V (no fixed image) | **I2V** — a fixed reference image starts the video |
| Text path | umT5 cross-attn, K/V cached behind `is_init` | LLaVA-Llama-3-8B + CLIP, **joint MM-DiT attention, NO text KV cache** |
| Swap mechanism | re-encode + flip `is_init` on 30 blocks | **just pass a different `prompt` arg per chunk** (re-encoded every chunk) — trivial |
| Scene "momentum" | self-attn KV cache of past frames (flushable via rolling window) | **`last_latents`** (continuity) + **`ref_latents`** (style anchor) — VAE image latents, **text-independent** |
| Chunk | 3 latent frames, 4 steps | **33 px frames ≈ 1.3 s**, 8 steps (distill) / 50 (std) |
| Loop | internal chunk loop in `inference()` | Python loop over actions, `sample_batch.py:242` |

**The big implication:** GameCraft's history is carried by **image latents fed as
extra input channels**, not by a text-conditioned KV cache. Verified mechanism
(`pipeline_hunyuan_video_game.py`):
- Per chunk the model denoises `latents_concat = concat([latents,
  gt_latents_concat, mask_concat], dim=1)` (**channel** concat, line 965) — i.e.
  HunyuanVideo inpainting-style conditioning. `gt_latents_concat` = the history
  reference latent; `mask_concat` = a **1=history / 0=predict** indicator (first
  half of frames = history, lines 913–933).
- `last_latents` (prev ~9 latent frames = 33 px / 4×) is written straight into the
  first half of the chunk's latents (line 961: `latents[:,:,:half] = last_latents`);
  the first chunk (image mode) only anchors frame 0 (line 959).
- `ref_latents` is the **first frame of the current chunk, updated every chunk**
  (`sample_batch.py:274`) — style drift IS allowed; it is not a rigid first-image
  pin (softer anchor than I feared).

Consequences for the swap:
- The swap itself is *mechanically trivial* (vary the prompt arg) — easier than SF.
- The **LongLive KV-recache lever does NOT transfer**: history is a VAE latent of
  *pixels* and the mask is text-independent, so there is nothing text-conditioned to
  "re-encode under the new prompt." Decode→VAE-reencode injects no new semantics.
  This is the honest architectural limit; the real levers are *reducing* history
  influence, not refreshing its meaning.
- Because it's **i2v + a per-chunk image-latent anchor + an explicit history mask**,
  expect **stronger resistance to drastic overhauls than Self-Forcing** — closer to
  Rolling Forcing's sink behavior (phase 18). Hypothesis H0.
- BUT the channel-concat injection exposes clean knobs (Levers A–C below) that are
  the GameCraft analog of our window-shrink/eviction — likely the key to forcing a
  swap.

## Hypotheses (from the playbook)

- **H0 — strong anchor:** drastic swaps (village→underwater) resist hard because the
  ref image + `ref_latents` anchor the scene. Same-structure steers (day→night,
  add snow/storm) take more easily. (Mirror of RF phase 18.)
- **H1 — ref freeze helps:** freezing `ref_latents` at the swap (`SWAP_RESET_REF=1`)
  reduces pull-back to the old scene. (Analog of removing the sink / window-shrink.)
- **H2 — detailed prompts help:** long cinematic prompts on both ends give cleaner
  commitment (SF phases 10/11). Cheap lever, test early.
- **H3 — runway matters:** more post-swap chunks ⇒ more complete morph; transition
  speed ≈ how fast history latents turn over (SF phase 02).
- **H4 — CFG matters:** the distill model runs cfg-scale 1.0 (no guidance) ⇒ weaker
  prompt adherence on the swap chunk (the SF "no-CFG" problem). The std model at
  cfg 2.0 (and higher) should adhere better. Test a CFG sweep on std.
- **H5 — waypoints/ramp smooth it** (SF phase 09/17): stepping the prompt through
  real intermediate scenes across chunks beats a hard cut for smoothness.

## Metrics (reuse SF harness, port to GameCraft outputs)

`analyze_swap.py` (to write): decode mp4 → per-frame
- **warmth** = mean(R−B) (day/sun +, night/cool −) — primary scene-change proxy
- **brightness** = mean luminance
- **jitter** = mean frame-to-frame pixel Δ (stability)
- **swap-moment Δ** = pixel Δ at the chunk boundary where the prompt switches
- **contact sheet** PNG (mark post-swap chunks), judged by eye
- bound the achievable change against two references: **all-old-prompt** and
  **all-new-prompt-from-start** (same seed + same start image) — like SF ref_sun/ref_night.

## Phase ladder (sub-phases under `22_gamecraft_swap/`; 20/21 taken by the Causal-Forcing port)

**22a baseline** (task #3): distill 8-step + std 50-step on `village.png`, action
`w s d a`. Confirm generation, record per-chunk wall-clock, peak VRAM, FPS. Decide
1-GPU vs sequence-parallel (GPUs 3/6/7 free) and resolution fallback if OOM.

**22b naive swap** (task #4): `sample_swap.py`, swap village-day → ruined-snowy-night
at action idx 3 of 6 (`w w s s d d`). Run distill (default) + **one std reference**
(the approved quality control). Question: does it change at all? where on the
resist↔snap axis? Expect (H0) partial resist.

**22c levers:** one knob at a time, distill unless noted —
- `SWAP_RESET_REF=1` (H1)
- detailed cinematic prompts both ends (H2)
- swap timing: idx 1 / 3 / 5 (H3, runway)
- CFG sweep on **std**: 2 / 4 / 6 (H4)

**22d same-structure vs drastic** (H0): a matrix of transitions —
- same-structure (expect easy): day→night, clear→storm, summer→snow on the village
- drastic (expect resist): village→underwater reef, village→desert dunes,
  village→neon cyberpunk city. Categorize what the anchor permits.

**22e smoothing** (H5): prompt ramp / semantic waypoints across chunks
(day→sunset→dusk→night), vs hard cut. Reuse SF phase-09/17 framing.

**22f history manipulation** (the key mechanism levers — now mapped). Patch
`pipeline_hunyuan_video_game.py` with env-driven knobs applied post-swap; this is
the GameCraft analog of window-shrink/eviction (SF phase 04). Reachable levers:
- **Lever A — history scale:** `latents[:,:,:half] = last_latents * HIST_SCALE`
  (line ~961). `HIST_SCALE<1` weakens the motion-history anchor.
- **Lever B — mask scale:** scale `mask_concat` before the concat (line ~965) —
  lower the "this is trusted history" signal so the new prompt can lead.
- **Lever C — gt zero/scale:** zero or down-scale the history portion of
  `gt_latents_concat` (lines ~922/962) — the cleanest "forget the old scene" knob.
- **Lever D — ref freeze:** `SWAP_RESET_REF` (already in `sample_swap.py`); expected
  smaller effect since `ref_latents` is only the current chunk's first frame.
- **NOT a lever — semantic recache:** history is text-independent pixel latents, so
  the LongLive re-encode-under-new-prompt trick injects no new meaning here. Record
  this as the architectural-contrast finding.
Sweep A/B/C alone + combined with detailed prompts; measure resist↔snap + jitter.

## Open unknowns to resolve at 22a (don't pre-optimize)

- **Memory/res:** 13B @ 704×1216 on one 80 GB H100 with `--use-fp8` — should fit;
  fallback = sequence-parallel over GPUs 3/6/7, or lower `--video-size`, or
  `--cpu-offload`. Verify allowed resolutions (HunyuanVideo wants multiples).
- **Latency:** 8-step 13B per 33-frame chunk on 1 GPU — if too slow to iterate,
  switch to SP across the 3 free GPUs (faster) for the sweep phases.
- **Distill no-CFG adherence (H4):** may cap how hard the distill model can swap;
  that's exactly why we keep the std reference.
- **Action coupling:** actions drive camera only (orthogonal to prompt) — confirm a
  swap doesn't interact with camera motion; keep the same action list across a
  comparison so only the prompt differs.

## Deliverables
- `experiments/20_gamecraft_swap/` … `25_*` phase folders (videos + sheets + README)
- `scripts/gamecraft_*` harnesses (swap, ramp, analyze)
- a findings write-up: feasibility verdict + which levers work on an i2v/image-
  anchored AR model vs the t2v KV-cache substrate, and how GameCraft compares to
  Self-Forcing / LongLive / Rolling Forcing for real-time text interruption.
</content>
