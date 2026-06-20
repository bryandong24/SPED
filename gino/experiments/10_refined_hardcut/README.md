# 10 — Base Method (Plain Hard Cut) + Refined Prompts

No cache tricks at all — just the plain prompt swap (re-encode new prompt + flip
cross-attn `is_init`) at 7.5s in a 15s clip (rolling window 21, sink 1). The twist:
make BOTH prompts long, detailed, cinematic (the model is trained on such prompts
and behaves better). `scripts/swap_long.py` with no --recache/--post_window/--grow_to.

| File | transition | warmth →end | note |
|---|---|---|---|
| `refined_dog_s7` | sunlit wildflower meadow → snowy moonlit winter night | 88→8 | gradual sky gradient sunset→blue, dog consistent ★ |
| `refined_dog_s42` | same, seed 42 | 83→19 | gradual warm→cool, consistent |
| `refined_dog_s123` | same, seed 123 | 91→25 | gradual, slightly less far |
| `refined_car_s7` | sunny coastal highway → neon rainy city night | 15→−42 | car consistent, day→neon/cyberpunk dusk ★ |
| `refined_car_s42` | same, seed 42 | 2→−30 | |

## Finding

The **base hard-cut works surprisingly well** when (a) the clip is long enough that
the rolling window flushes the old scene gradually (~5s runway after the swap) and
(b) **both prompts are richly detailed** — the model commits more cleanly and the
composition is better. The result is a *gradual* transition (sunset→blue evening /
day→neon dusk) with a consistent subject, with NONE of the cache surgery from
phases 04–09. Detailed prompts are a cheap, effective lever.

Caveat: still not a crisp "snap" and the swap moment is soft rather than perfectly
smooth, but for long clips this is the simplest decent result — and avoids the
resist/jitter/abrupt-cut failure modes of the fancier methods.
