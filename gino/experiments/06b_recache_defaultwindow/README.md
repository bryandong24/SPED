# 06b — Recache at the DEFAULT window (21) — stability↔compliance trade

Same 3 transitions as 06, but with recache and the **default window (21 frames,
the model's global cache)** — i.e. NO window-5 shrink. Shows the other end of the
trade-off.

| File | warmth @0→100% | vs window-5 (06) |
|---|---|---|
| `ex1_dog_meadow2wolf_night_win21` | 71→32 (barely moves) | window-5: 58→−39 (transitions) |
| `ex2_car_coast2neoncity_win21` | −11→−18 (little change) | window-5: 6→−24 |
| `ex3_beach_calm2storm_win21` | 6→−4 (minimal) | window-5: 21→−57 |

**Finding:** at the default window-21, recache is **more stable** frame-to-frame but
the swap is **much weaker** (strong old-prompt inertia — the new prompt fights a full
~5s of history). Window-5 transitions but jitters. There is no single window that is
both stable and fast-switching → motivates the attention-**sink** experiments (06c).
