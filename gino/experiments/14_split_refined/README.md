# 14 — Split-Encode Recache + Refined Prompts (smoothest result)

Phase-13 split-encode recache + detailed cinematic prompts. 15s, swap at 7.5s.
`scripts/swap_split.py --recache W --recache_new M`.

| File | W / new | warmth →end | cuts (px:Δ) | note |
|---|---|---|---|---|
| `ref_w15_new3` | 15 / 3 | 87→20 | all Δ≈10 | smooth, weaker transition |
| `ref_w15_new6` | 15 / 6 | 99→**−71** | all Δ≈**6–7** | ★ smoothest + full night |
| `ref_w18_new4` | 18 / 4 | 79→−15 | Δ≈10 | smooth, partial |

**Finding:** `--recache 15 --recache_new 6` + refined prompts = the cleanest swap
of the whole arc — per-frame jumps at the swap drop to Δ≈6-7 (vs Δ45+ for plain
recache) AND it reaches a deep night (warmth −71). The split-encode (keep most of
the window as un-recached old frames, re-encode only the last few under new) +
detailed prompts is the recommended training-free recipe.
