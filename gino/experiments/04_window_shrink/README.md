# 04 — Post-Swap KV-Window Shrink

Principled idea from phase 02: transition time ≈ window flush time → **shrink the
self-attn read window right after the swap** so old frames flush almost immediately.
12s clips, 3-frame blocks, swap at ~2.25s. (`set_window` rewrites each block's
`self_attn.max_attention_size`/`local_attn_size` live.)

| File | Setting | Result |
|---|---|---|
| `win21_ref.mp4` | no shrink (window 21) | barely transitions in 12s (warmth 56→14) |
| `win8.mp4` | window 8 | slow |
| `win5.mp4` | window 5 | transitions, moderate speed |
| `win3.mp4` | **window 3** | ✅ **fast snap** to moonlit night (~1-2s), holds (warmth 79→−64) |
| `win3_restore.mp4` | window 3 then restore to 21 | drifts back warmer (restore re-grows old context) |

### Generalization (window-3 on other transitions)
| File | Transition |
|---|---|
| `gen_meadow2beach.mp4` | meadow → tropical beach at sunset ✓ |
| `gen_meadow2fire.mp4` | meadow → forest wildfire ✓ |
| `gen_city_d2n.mp4` | city street day → neon night ✓ |
| `gen_meadow2moon.mp4` | meadow → astronaut on the moon ✓ |

**Finding:** window-shrink gives a **fast, generalizable** scene swap — BUT it
**breaks motion/identity** (the subject jump-cuts between inconsistent poses)
because shrinking to 3 frames throws away long-range motion context. Fast but not
smooth. → fixed in phase 05.
