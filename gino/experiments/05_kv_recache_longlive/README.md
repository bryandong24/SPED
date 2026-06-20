# 05 — KV-Recache (LongLive), training-free

Fix for the phase-04 jump-cuts. From **LongLive** (arxiv 2509.22622, Wan2.1-based,
same base as ours). At the swap: keep the already-generated frames, but **re-encode
the recent prefix through the model paired with the NEW prompt** and rebuild the
self-attn KV cache from it. Visual/motion state preserved; cached *semantics*
refreshed. Why it works: self-attn K/V at deeper blocks depend on earlier blocks'
cross-attn output, so re-encoding under a new prompt changes the cached K/V even
though the frames are identical. Implemented training-free in `swap_long.py:recache()`.

| File | Setting | Result |
|---|---|---|
| `rc_sun2night_w21.mp4` | recache, full window 21 | ✅ **smooth motion** (dog consistent) but weak compliance — scene only mildly cools (inertia) |
| `rc_recache_plus_win5.mp4` | recache **+ window 5** | ✅ **best trade** — smooth motion AND scene transitions to night |
| `rc_sun2night_ll.mp4` | recache + 9-window + 3-sink (LongLive cfg) | smooth, slower compliance (untrained) |
| `rc_beach.mp4` | recache, meadow→beach | generalizes |
| `rc_citynight.mp4` | recache, city day→night | generalizes |
| `rc_20s.mp4` | recache, 20s clip | longer-horizon check |

**Finding:** recache **preserves motion** where window-shrink broke it; **recache +
a moderately short window** flushes old semantics fast while retained frames keep
motion continuous — the best smoothness/compliance trade-off found.

**Caveat / positioning:** this is the training-free version of LongLive's inference
trick; LongLive *fine-tunes* the model to handle recache (cleaner results). Our
project's novelty is orthogonal — the coarse-to-fine **spectral speedup** for
lower interrupt latency, not the swap mechanism itself.
