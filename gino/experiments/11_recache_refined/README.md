# 11 — Recache Trick + Refined Detailed Prompts

The best combo so far: KV-recache cache construction (07) + grow-window (06c) +
richly-detailed cinematic prompts (10). 15s, swap at 7.5s, recache 9, post_window 3,
grow_to 15. `scripts/swap_control.py --mode {new,null,vswap} --grow_to 15`.

| File | mode | warmth →end | jitter | note |
|---|---|---|---|---|
| `rcr_dog_null` | null | 85→**−57** | **6.32** | ★ lush sunlit meadow → proper moonlit snowy night; preserves the original dog |
| `rcr_dog_vswap` | vswap | 92→−50 | 7.28 | ★ commits harder to the night scene / snow |
| `rcr_dog_new` | new (plain recache) | 86→−35 | 6.95 | good, slightly less deep night |
| `rcr_car_null` | null | −2→−23 | 8.43 | coastal day → neon night, car consistent |
| `rcr_car_vswap` | vswap | −5→−13 | 11.48 | |
| `rcr_car_new` | new | 6→−10 | 8.27 | |

## Finding

**Refined prompts + recache + grow-window = the highest-quality result of the whole
arc.** The detailed prompts give much richer scenes on BOTH ends (lush wildflower
meadow ↔ full-moon snowy night with frosted pines), the model commits to a deeper
night (warmth −57 vs ~−39 with terse prompts), recache+grow keep motion consistent
and jitter low (~6.3), and the dog stays coherent. `null` preserves the original
subject identity; `vswap` commits harder to the new scene.

The swap *moment* is still a soft-but-visible transition (not a perfect dissolve),
but both halves are high quality and the handoff is the cleanest training-free
version we have. **Recommended showcase config:** `--mode null --recache 9
--post_window 3 --grow_to 15` with long detailed prompts.
