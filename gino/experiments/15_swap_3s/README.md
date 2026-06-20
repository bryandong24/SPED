# 15 — Earlier Swap (at ~3s)

Split-encode recache with the swap moved to ~3s (latent frame 12) in a 15s clip,
so there is ~12s of post-swap runway -> a long gradual morph. Refined prompts.

| File | W / new | warmth →end | cuts | note |
|---|---|---|---|---|
| `s3_dog_w12n3` | 12 / 3 | 90→−30 | small/clustered | ★ meadow→moonlit night, gradual |
| `s3_dog_w18n4` | 18 / 4 | 91→−3 | Δ≈9 | smoothest, weaker |
| `s3_dog_full` | plain recache | 90→0 | resists | control |
| `s3_car_w12n3` | 12 / 3 | 7→**−70** | clustered@swap | ★ coast→neon night, strong |
| `s3_car_w15n3` | 15 / 3 | −9→−38 | Δ≈13 | |

**Finding:** the split-encode fix holds at an early 3s swap; with 12s of runway the
transition is a long gradual morph, subject consistent, no double-cut. Window 12,
new 3 is a good default at this swap point.
