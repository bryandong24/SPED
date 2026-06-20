# Phase 24 — CF baseline hard-cut (no tricks), 3 settings

Clean Causal-Forcing baseline: **plain hard-cut prompt swap, NO grow-window, NO recache** — for
the 3 standard pairs. This is the honest reference the smoothing methods (Phase 23) should beat.

Settings: chunkwise `causal_forcing.pt`, 40 latent frames (~10 s), swap @ frame 12 (~3 s),
`window=21, sink=3` (rolling window only so the 10 s clip fits — see note), `post_window=0,
grow_to=0` (no grow), no recache. `cf_smooth.py --method hardcut --grow_to 0 --post_window 0`.

## Result — hard-cut largely RESISTS in 10 s
warmth = mean(R−B); dog/jungle expect ↓ /↑ respectively to the B-scene; lower-magnitude = resists.

| pair | warmth pre→final | read |
|---|---|---|
| dog meadow→snowy night | 55 → **−7** | barely cools; stays meadow (cf. Phase 23 hardcut **−44** WITH grow-window) |
| car coast→neon night | −29 → −36 | stays coastal, minor change |
| jungle→red desert | −1 → **−6** | essentially no transition, stays jungle |

**Takeaway:** stock hard-cut on CF (swap @3 s, 10 s clip) **resists drastic scene changes** — the
self-attn KV momentum holds the old scene; only the rolling window slowly flushing it would change
the scene, but that needs much more runway. **The grow-window (window shrinks→grows post-swap) was
doing the heavy lifting** in Phase 23 — without it, even the "hardcut" rows there would resist like
this. So the right baseline to compare smoothing methods against is THIS (no-grow hard-cut), and
the grow-window itself is one of the levers, not a neutral default.

Note on settings: CF's chunkwise model has **no sink and global attention by default**
(`sink_size=0`, `local_attn_size=-1`, caps at 21 frames). We force `window=21, sink=3` purely so a
40-frame (10 s) clip fits a bounded rolling cache; otherwise this is the most-default hard-cut.

Artifacts: `baseline_hardcut.png` (3 rows, red stripe = post-swap), clips in `clips/`.
