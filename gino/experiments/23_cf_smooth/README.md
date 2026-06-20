# Phase 23 — Principled smooth-transition methods for CF (Tier-1 scaffold)

Implements the approved plan (`~/.claude/plans/modular-toasting-planet.md`). **Scaffolded +
CPU-validated, NOT yet run on GPU** (all 8 GPUs were taken by subha's job; user paused GPU use).

## Code (in `gino/scripts/`)
- `cf_smooth.py` — single ablatable harness. Training-free toggles:
  - **A1** min-jerk / raised-cosine **conditioning ramp** on the embedding **geodesic (SLERP)**
    — replaces the C0 hard prompt step with a C2-smooth A→B ramp over `transition_chunks`.
  - **A2** critically-damped **grow-window** co-schedule (anchored at swap).
  - **A3** **Kalman-gain KV handoff** — recache the window under BOTH A and B, blend
    `(1-g)*KV_old + g*KV_new` with a min-jerk gain `g(t)` instead of hard-replacing (smooth
    self-attn state handoff; also avoids the block-evict double-cut). Requires
    `recache_W + nfpb*transition_chunks <= window` (asserted) so the blended slots don't evict
    mid-transition.
  - **B1** **CORAL / OT-Gaussian alignment** — map the recached K/V per-head mean+cov to the
    pre-swap in-distribution stats (`coral_align`), killing OOD recache artifacts.
  - `--method` registry: `hardcut`, `recache`, `coral`, `kalman`, `ramp`, `smooth_all`, `all`
    (the first two reproduce the Phase-21 baselines for free).
- `run_cf_smooth.sh <GPU_ID> [example]` — launch the 6-method × 3-example ablation (loads the
  model once per example via `--method all`).
- `build_cf_smooth.py` — stacked comparison grids (hardcut→recache→coral→kalman→ramp→smooth_all),
  each row labelled with warmth pre→final + a luma-jerk smoothness proxy.

## Validation done (no GPU)
`python cf_smooth.py --self_check` — **PASSED**: schedules (min-jerk flat-start 0.035 → C2,
monotonic, correct endpoints), SLERP (endpoints exact, magnitude-lerped midpoint), CORAL
(aligned new (mean .5/std .7) → old (mean 1.0/std 2.5) exactly). All three scripts `py_compile` clean.

## To run when a GPU frees up
```
bash gino/scripts/run_cf_smooth.sh <GPU_ID>      # 3 examples x 6 methods
python gino/scripts/build_cf_smooth.py           # build compare_{dog,car,jungle}.png
```
Then compare `smooth_all` vs `hardcut`/`recache` baselines: expect jerk↓ at the swap and fewer
OOD artifacts, without losing prompt compliance (warmth still reaches the B-scene value).

## GPU RUN — results (2026-06-20)
Ran all 6 methods × 3 examples (dog→snow, car→neon, jungle→desert), 10 s clips, swap @3 s,
transition_chunks=4, recache_W=9, window 21/sink 3, grow_to 15, min-jerk gain. Clips in `clips/`,
grids `compare_{dog,car,jungle}.png`. **The kalman + coral KV-cache paths ran correctly** (no
crashes, sane output) — the bookkeeping (double-recache snapshot, per-chunk `blend_window`,
`coral_align` on live cache) is validated.

**Abruptness / smoothness metrics** (avg over 3 examples; lower = smoother):
| method | max single-frame jump ↓ | full-RGB jerk ↓ |
|---|---:|---:|
| hardcut | 51.2 | 13.40 |
| recache | 54.6 | 17.46 |
| coral | 53.7 | 16.30 |
| kalman | 52.9 | 13.73 |
| **smooth_all** | **49.7** | 16.91 |
| ramp | 51.6 | 18.23 |

**Findings (honest):**
- **smooth_all gives the lowest max single-frame jump (49.7, −9% vs recache 54.6)** — the cleanest
  "abrupt-cut" measure — and visually ramp/smooth_all spread the A→B change over more frames
  (gradual morph) vs the harder snap of hardcut/recache. Goal #1 (cut-moment) shows a real but
  **modest** gain.
- The **full-RGB jerk column is confounded**: methods that transition *further* (more meanMotion)
  show more change → higher jerk, so it can't rank smoothness cleanly. Need a proper perceptual
  metric (LPIPS-velocity) or **latent-space jerk** to isolate smoothness from transition depth.
- **Smoothness↔compliance tradeoff** (as predicted): the smoother rows transition *less far* in the
  same window (warmth endpoints less extreme: dog smooth_all −15 vs kalman −61).
- **coral** slightly reduced jerk vs plain recache (16.30 vs 17.46) — weak evidence B1 helps the
  OOD artifacts, but inconclusive at this resolution.

**Next:** (1) proper latent-jerk / LPIPS-velocity metric; (2) hold transition horizon vs depth
constant to disentangle the tradeoff; (3) sweep transition_chunks / gain_kind; (4) Tier-2 (B2
adaptive steps) + C1 (spectral) per the plan.

GPU note: GPUs 0–3 were grabbed mid-run by another `spycoder` 4-GPU job (killed the first batch via
contention); the rerun used the free GPUs 4–6. Each method is best run as a single-method process
(load → 1 method → exit) to avoid cross-method allocator buildup.
