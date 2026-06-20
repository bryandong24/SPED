# 06c — Grow-Window Stability Schedule

Stability fix from a control view: at the swap, window = small (fast flush of old
content via recache + tiny window), then **grow ~1 frame per new frame** up to a
cap. Anchored at the swap so the window only ever spans NEW-prompt content (recache
already purged raw old frames) → no inertia return. sun→night, 15s, swap at 7.5s.

| File | window schedule | warmth @0→100% |
|---|---|---|
| `fixed_win5` | constant 5 | 55→−25 (transitions but jittery) |
| `grow3to8` | 3→8 | 68→−45 (sharp transition) |
| `grow3to15` | 3→15 | 81→−24 |
| `grow3to21` | 3→21 | 67→−42 |

**Finding:** growing the window transitions *as strongly* as a fixed small window
but is far more stable (see 07 jitter numbers) — the early small window flushes the
old scene fast, the later large window stabilizes the new one. This is the
"critically-damped reference" idea: spread the control over the system's time
constant rather than a step. Combined with cache constructions in **07**.
Generated via `swap_long.py --grow_to N` / `swap_control.py --grow_to N`.
