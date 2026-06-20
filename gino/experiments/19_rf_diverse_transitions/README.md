# 19 — Rolling Forcing: Diverse Drastic Transitions (best method)

The best RF prompt-swap recipe (from 18) applied to 7 diverse DRASTIC scene
transitions. Method: **full reset + warm-seed** (`--full_reset --warm_seed`, no recache). At the swap (7.5s, middle) wipe both the clean KV cache and the noisy rolling-window staircase, then SEED the staircase with one pure-night frame re-noised to each level — this removes the grey cold-start gap. 15s clips, refined prompts. `rf_swap.py --full_reset --warm_seed --swap_frame 30`.
the clean KV cache and the noisy rolling-window staircase, and seed the clean cache
with the last 6 committed frames re-encoded under the new prompt. 15s clips, refined
prompts. `rf_swap.py --recache 6 --full_reset --swap_frame 30`.

| File | transition | warmth start→end | result |
|---|---|---|---|
| `div_dog_meadow2wolfnight` | sunlit meadow dog → snowy-night wolf | 14→−34 | ✓ moonlit snowy night |
| `div_car_coast2neoncity` | red car sunny coast → neon rainy city night | 16→−16 | ✓ vivid neon night, car kept |
| `div_beach2storm` | tropical sunset beach → stormy ocean + lightning | 15→−10 | ✓ darker stormy sea |
| `div_jungle2desert` | green jungle waterfall → orange desert dunes | 19→**171** | ✓ bright desert (a bit oversaturated) |
| `div_spring2volcano` | cherry-blossom spring park → erupting volcano lava | 17→32 | ✓ warm lava/volcano |
| `div_reef2mountain` | underwater coral reef → snowy mountain peak | 16→−13 | ✓ snowy peak |
| `div_city2forest` | busy city street day → misty pine forest at dawn | 17→56 | ✓ golden misty forest |

## Notes

- All 7 transition correctly to the new scene (warmth swings match each target:
  desert/volcano/forest warm-positive; night/storm/neon/mountain cool-negative).
- Every clip has a **brief dark/washed moment at the reset** (both caches wiped) —
  intrinsic to forcing a drastic swap on RF (its attention-sink + rolling-window
  fight scene changes). The night-recache seeding keeps that gap short.
- Subject/structure is re-authored under the new prompt (not preserved) — this is a
  scene CHANGE, not a same-subject morph. For same-structure steers (weather/
  intensity), use the plain hard cut instead (see 18, `calm2storm`).

Best RF recipe summary: **same-structure steer → plain hard cut; drastic overhaul →
`--recache 6 --full_reset`.**

## Grey-gap fix (warm-seed)

Earlier versions (recache-seeded full reset) had a grey/washed moment at the swap:
zeroing the rolling-window staircase = a cold start (blocks labeled "nearly clean"
held zeros → under-denoised grey frames until the staircase refilled). Fix: generate
one pure-night frame (few-step denoise vs the EMPTY cache, so it's unbiased by the old
scene) and seed each staircase slot with it re-noised to its level. Measured grey-gap
saturation (low = grey) went from 4–25 (old) to **60–121 (warm-seed)** across all 7,
with the transition preserved. Do NOT add --recache here: a recache-seeded warm-up
re-anchors the old scene's structure (subject stayed in the old world).
