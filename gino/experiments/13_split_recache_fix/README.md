# 13 — Split-Encode Recache (fixes the double-cut)

Fix (user's idea) for the double-cut diagnosed in 12: instead of rebuilding the
WHOLE window from old-frames-re-encoded-under-new (one big bridge that evicts as a
block), rebuild the window as a **MIX**:

  [ (W−M) frames kept under the OLD prompt | M most-recent frames under the NEW prompt ]

so the window attends to both un-recached (old) and recached (new) frames, and the
old frames evict **one at a time** as native-new frames accumulate → no abrupt
bridge-drop. `scripts/swap_split.py --recache W --recache_new M --window W`.

| File | W / new | warmth →end | cuts (px:Δ) | verdict |
|---|---|---|---|---|
| `baseline_full` | 9 / 9 (=plain recache) | 79→4 | 117, **153, 156** | double-cut (control) |
| `split_w9_new3` | 9 / 3 | 97→−25 | 120, **153, 156** | still double-cuts (window too small) |
| `split_w12_new3` | 12 / 3 | 83→5 | 117/129/152 all Δ≈10 | smoother, weaker |
| `split_w15_new3` | 15 / 3 | 94→**−50** | cuts clustered 117–123, **no 2nd cut** | ★ single clean transition + full night |
| `split_w18_new3` | 18 / 3 | 82→−7 | all Δ≈10 | smoothest, transitions less |

## Finding

**The split-encode recache removes the second cut** — provided the window is large
enough (≥~12–15) that the un-recached old frames evict gradually rather than in a
block. `W=15, new=3` is the sweet spot: a single clean handoff AND a full
transition to night, no double-cut. Smaller windows (9) still double-cut; very
large (18) over-smooths and weakens the swap. Refined-prompt version in
`../14_*` / `out/swap_split_ref`. Recommended: `--recache 15 --recache_new 3
--window 15`.
