# 06 — KV-Recache on Diverse 15s Transitions

The best method (recache + window-5) applied to 3 diverse transitions. ~15s clips
(60 latent / 237 px frames), 3-frame blocks, prompt swapped at **~7.5s (halfway)**.
Config: `--recache 9 --post_window 5` (replay last 9 frames under the new prompt to
refresh KV semantics, then read window = 5 latent frames ≈ 1.25s for fast-but-smooth
compliance). Uses the fixed relative-position recache (robust for late/mid swaps).

| File | Transition | Result |
|---|---|---|
| `ex1_dog_meadow2wolf_night.mp4` | golden retriever in sunny meadow → wolf in snowy moonlit night | transitions to blue night; subject morphs (dog→wolf is the most visible identity change) |
| `ex2_car_coast2neoncity.mp4` | red sports car on sunny coast → same car in neon rainy city at night | **cleanest** — car stays consistent/smooth, environment flips day→night |
| `ex3_beach_calm2storm.mp4` | calm sunset beach → stormy dark ocean | smooth mood/weather shift, ocean+horizon structure preserved (no jump-cut) |

**Takeaway:** recache+window-5 preserves motion across the swap (no jump-cuts) with
a gradual, coherent scene change. Strongest when the prompts share scene structure
(ex3) or a persistent subject (ex2); looser when the subject identity itself changes
(ex1). Generated via `scripts/swap_long.py --recache 9 --post_window 5`.
