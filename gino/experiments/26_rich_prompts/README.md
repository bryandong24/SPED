# Phase 26 — Rich (training-matched) prompts × forward-ramp, 15 s, 3 settings

Re-runs Phase 25 with **really rich, training-matched prompts** (multi-sentence: same subject both
ends with wardrobe/texture, scene elements, lighting, explicit camera framing + movement — matching
CF's native `prompts/demos.txt` style). New `*_rich` settings in `cf_smooth.py`. Method: **forward
conditioning ramp, no recache** (`fwd_ramp`), vs `hardcut` reference. 60 latent frames (~15 s),
swap @3 s, window 21/sink 3, NO grow, NO recache, min-jerk gain. Clips in `clips/`, grids
`compare_{dog,car,jungle}.png`.

## warmth pre→final (mean R−B; dog/car ↓ to night, jungle ↑ to red desert)
| setting | hardcut | fwd_ramp k=4 (~3 s) | fwd_ramp k=8 (~6 s) |
|---|---|---|---|
| dog_rich (meadow→snow night) | 40 → −27 | 48 → −6 | 46 → **−38** |
| car_rich (coast→neon night) | 17 → 16 | 16 → **−36** | 16 → 12 |
| jungle_rich (jungle→red desert) | 8 → 28 | −3 → 17 | 9 → 8 |

## Findings (honest)
- **Quality: clear win.** Rich prompts produce visibly richer, more composed scenes — mountains and
  depth in the meadow, glossy chrome car + palm-lined coast, detailed waterfalls/emerald canopy.
- **Rich prompts make each scene a STRONGER ATTRACTOR — cuts both ways:**
  - Helped **dog** (hardcut now commits to night 40→−27, vs plain-prompt hardcut that resisted
    55→19) and **car** (fwd_ramp k4 reaches neon −36).
  - **Hurt jungle's transition**: the lush detailed jungle holds on harder, so the desert
    transition is *weaker* than plain prompts (Phase 25 jungle k4 hit +75; here only +17). Two very
    rich scenes = more inertia on both sides.
- **Best per setting:** dog → fwd_ramp **k8** (smooth, deep night); car → fwd_ramp **k4** (neon);
  jungle → all weak (rich jungle too sticky — would need more runway / a stronger push).
- **Forward ramp (no recache) holds up** as the transition mechanism and stays self-consistent (no
  retroactive cache rewrite), confirming the Phase-25 idea; the right ramp horizon is still
  setting-dependent (k4 vs k8).

## Takeaway
Detailed prompts are a strong lever for **scene richness** and for *committing* to a target scene,
but they also deepen each scene's basin — so for hard transitions between two equally-rich scenes,
pair rich prompts with a longer runway or a stronger transition push (grow-window / recache /
larger k), not rich prompts alone.
