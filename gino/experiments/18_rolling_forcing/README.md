# 18 — Rolling Forcing: Baseline Hard Cut + Interventions

Tried prompt-swapping on **Rolling Forcing** (arxiv 2509.25161, TencentARC) — AR
long-video diffusion on Wan2.1-1.3B, built on the Self-Forcing codebase, with a
**rolling diffusion window + attention sink** for multi-minute consistency. Same
swap hook as Self-Forcing (swap conditional_dict + reset crossattn is_init).
Harness: `rf_swap.py`. 15s clips (63 latent frames), swap at 3s.

## Findings

**The baseline hard cut RESISTS drastic scene changes.** `baseline_hardcut_resists`
(sunny meadow → snowy winter night): stays a sunny meadow the whole clip.

**Not a bug** — verified:
- `diag_allnight` (prompt = night from the start) → renders night ✓
- `diag_swap_at_frame0` (swap at frame 0) → night ✓ (mechanism works)
- swaps at 3s / 7.5s → resist.

So RF genuinely resists mid-rollout overhauls: its **attention sink globally anchors
the initial frames** (the long-horizon-consistency mechanism) and **fights** scene
changes. Self-Forcing also resists, but RF resists harder.

**Cache interventions (clean-cache wipe / recache / prompt-ramp) DON'T break it** —
`intervention_kvreset_resists` still stays sunny. Reason: RF has a SECOND cache, the
`noisy_cache` (the in-flight rolling-window staircase), which carries old-scene
momentum even after the clean KV cache is reset.

## What works

- **★ `SUCCESS_calm2storm_samestructure`** — a SAME-STRUCTURE steer (calm ocean →
  stormy ocean) works with a **plain hard cut**: the ocean stays an ocean, the
  waves grow bigger/stormier. RF's "steering" is for weather/intensity/attribute
  changes, not structural overhauls.
- **★ `SUCCESS_fullreset_sun2night`** — for a drastic change, wiping **both** caches
  (clean + `noisy_cache`) at the swap breaks the resistance → reaches a moonlit
  winter night. Cost: a brief washed-out gap at the reset (both caches empty for a
  few frames). `--full_reset`.
- `day2rain_partial` — weather shifts partially (meadow stays).

## Takeaway vs Self-Forcing

RF trades steerability for stability: its sink + rolling window give great long-video
consistency but make drastic prompt swaps harder. Same-structure steering works
cleanly; structural overhauls need a full (clean+noisy) cache reset, which
reintroduces an abrupt gap. (Env note: RF reuses Self-Forcing's venv — a concurrent
fork polluted it; fixed with transformers 4.49.0 + huggingface_hub 0.28.1.)

## UPDATE — getting RF to transition correctly (recache-seeded full reset)

The plain `full_reset` reaches night but with a long washed-out gap (both caches
empty until they refill). Fix: **wipe both caches AND seed the clean cache with the
last N committed frames RE-ENCODED under the new prompt** (`--recache N --full_reset`).
The night-recache gives the wiped cache immediate night context so it commits fast.

| File | config | warmth →end | note |
|---|---|---|---|
| `SUCCESS_recache6_fullreset` | --recache 6 --full_reset | 14→**−53** | ★ bright moonlit night, brief reset gap |
| `SUCCESS_recache12_fullreset` | --recache 12 --full_reset | 14→−47 | darker snowy night |
| rf_fr_ramp4 | full_reset + ramp | 20→0 | ramp diluted it (keeps sun longer) — worse |
| rf_*_swap6 | earlier swap (1.5s) | →−9 | transitions, weaker |

**Verdict:** `--recache N --full_reset` makes RF transition correctly through a
drastic scene change. Wiping the `noisy_cache` (rolling staircase) is essential
(clean-cache-only resets resist); seeding with night-recache removes most of the
washed gap. A brief dark moment at the reset remains — RF's sink/rolling-window
fundamentally fights drastic mid-rollout swaps, so a clean reset has a cost. For
same-structure steers (calm→storm), the plain hard cut is still cleaner (no reset).
