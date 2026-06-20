# 09 — Waypoint Prompt-Stepping for a Smooth Swap

Idea (user): rather than interpolate the cache/output (still cuts), step the PROMPT
through intermediate waypoints, with **recache + a few frames of real rollout at
each waypoint**, so the video passes through genuinely-generated intermediate states.
15s, sun→winter-night, swap at 7.5s. `scripts/swap_waypoint.py`.

## Two variants

**A) Embedding-average waypoints** (`embavg_*`) — the literal "average of old & new
vector" idea: cond = lerp/slerp(P_old, P_new, beta) for beta in e.g. [0.5, 1.0].

| File | waypoints | result |
|---|---|---|
| `embavg_wp_avg_h2` | 0.5, 1.0 | partial; tends to resist (stays sunny) |
| `embavg_wp_avg_h3_slerp` | 0.5, 1.0 slerp, hold 3 | best of these — passes through a dusk-ish state into darker night |
| `embavg_wp_4step_h2` | 0.25/0.5/0.75/1.0 | **resists** — stays sunny |

→ **Linear/slerp averages of prompt embeddings are NOT semantically "halfway"**
(embedding space isn't linear-semantic); the averaged vector stays near "sunny" and
the recached sunny structure persists → resistance.

**B) Semantic waypoints** (`sem*`) — step through REAL intermediate prompts
(sunset → dusk → blue-hour → winter night), each an in-distribution scene.

| File | chain | warmth →end | note |
|---|---|---|---|
| `sem3_h2` | sunset→dusk→night, hold 2 | 71→−27 | **best** — gradual warm→dim→blue-night, dog consistent |
| `sem3_h3` | hold 3 | 75→1 | smoother but weaker |
| `sem4_h3` | 4 steps, hold 3 | 68→34 | **resists** — too gradual, never commits |

## The honest finding: smoothness ↔ compliance tension (training-free)

Across ALL swap-smoothing families tried (08 crossfades/dissolve, 09 waypoints):
- **More gradual** (more/longer waypoints, slower fades) → the transition **resists**
  or lands in a muddy middle; never cleanly reaches the new scene.
- **Cleanly reaches the new scene** (recache, window-shrink) → **abrupt cut**.

Semantic waypoints (B) are the best training-free compromise — real intermediate
scenes give a genuine progression — but a *perfectly* smooth AND fully-compliant
mid-rollout swap was not achieved training-free on the 4-step distilled model.
LongLive gets it by TRAINING (streaming long tuning teaches smooth recache
transitions); that is the likely real fix. Best training-free config:
`swap_waypoint.py --prompt_chain "sunset|dusk|night" --hold 2`.
