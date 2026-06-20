# Real-time Text Interruption for Autoregressive Video — Experiment Log

Substrate: **Self-Forcing** (Wan2.1-T2V-1.3B, chunk-wise autoregressive, KV-cached,
4-step distilled). Goal: swap the text prompt mid-generation to steer the video in
real time. All clips 480×832, 16 fps, single H100. Each phase folder has its own
README + the videos and frame contact-sheets (`*_sheet.png`).

## The story arc

| Phase | Question | Finding |
|---|---|---|
| **00 baseline_streaming** | Does Self-Forcing run / stream? | ✅ 81 frames in ~6.9s; chunk-by-chunk streaming; rolling window → 20s+ |
| **01 naive_swap_resists** | Just swap the prompt mid-rollout? | ❌ **Resists** — scene stays the old world (KV-cache momentum). Clean only with NO history (`instant`). |
| **02 resistance_gradient** | How much does history matter? | Resistance ∝ accumulated history. 1 frame → slow morph; long runway → eventually transitions. |
| **03 manual_scaling** | Scale text-vs-frame conditioning? | Text↑ slides sunny→gray *mush* (structure stays locked); CFG/​wipe → artifacts/static. KV locks **structure**, not just color. |
| **04 window_shrink** | Shrink the KV window post-swap? | ✅ **Fast** clean scene swap (~1-2s) + generalizes (beach/fire/city)... ❌ but **jump-cuts** — motion/identity breaks. |
| **05 kv_recache_longlive** | Re-encode the prefix under the new prompt? | ✅ **Smooth motion** preserved + scene transitions. recache + short window = best trade. (= LongLive's method, here training-free.) |
| **06 recache_15s_diverse** | Best method on diverse 15s clips? | ✅ recache + window-5, swap at 7.5s: dog→wolf-night, car coast→neon-city, beach→storm. Smooth motion, coherent transitions. |
| **06b recache_defaultwindow** | recache at default window 21? | stable but weak swap (inertia) — the other end of the stability↔compliance trade. |
| **06c grow_window** | grow the window post-swap? | ✅ small→large window: strong transition + much more stable ("critically-damped" control). |
| **07 cache_control** | control-theory / mechinterp cache builds? | ✅ **best results**: K-preserve/V-swap or null-recache + grow-window → strong swap, **3× lower jitter** than plain recache. |
| **08 smooth_swap** | interpolate to smooth the swap MOMENT? | crossfades/dissolve: each still cuts perceptually or resists. cut@swap metric proved unreliable — judge by eye. |
| **09 waypoint** | step through intermediate prompts + rollout? | embedding-averages resist (not semantically halfway); **real semantic waypoints** (sunset→dusk→night) = best training-free compromise. Core finding: **smoothness↔compliance tension; clean+smooth swap likely needs training (LongLive).** |
| **10 refined_hardcut** | plain hard cut + long detailed prompts? | ✅ **simplest decent result** — base swap in a 15s clip with richly-detailed prompts gives a gradual, consistent transition with no cache tricks. Detailed prompts are a cheap, effective lever. |
| **11 recache_refined** | recache trick + detailed prompts? | ✅ **highest quality of the whole arc** — recache + grow-window + detailed prompts: rich meadow ↔ moonlit snowy night, deepest transition (warmth −57), low jitter, consistent dog. null preserves subject, vswap commits to scene. |
| **12 double_cut_diagnosis** | why do recache models double-cut? | ✅ **root cause traced**: recache injects an old-structure bridge; cut2 = when the window evicts it (cut2 tracks window size, not recache size). |
| **13 split_recache_fix** | fix the double-cut? | ✅ **split-encode recache**: recache few frames under new + keep rest of window as un-recached old → gradual eviction, no 2nd cut. W=15/new=3 sweet spot. |
| **14 split_refined** | + detailed prompts? | ✅ **smoothest swap of the arc** — W=15/new=6 + refined prompts: Δ≈6-7 jumps, full night. |
| **15 swap_3s** | earlier swap (3s)? | ✅ holds at 3s; 12s runway → long gradual morph, no double-cut. |
| **16 alpha_sweep** | scale prompt cross-attn on a hard cut? | re-lights/changes time-of-day (α≈1.2 = nice sunset→dusk) but can't change STRUCTURE (no snow/moon); α<1 artifacts, α>1.3 gray-mush. Confirms structure-locking. |
| **17 hardcut_prompt_ramp** | principled fix for hard-cut artifacts? | architectural diagnosis: cache must be consistent (no artifacts) AND slow (no snap); hard cut breaks consistency, recache breaks slowness. Fix = **pure hard cut + prompt ramp** (prompt leads cache slightly → less conflict, keeps gradual morph). pfade9 default. |
| **18 rolling_forcing** | prompt swap on Rolling Forcing? | RF resists drastic swaps HARDER (attention-sink anchors scene; not a bug — swap@0 works). Same-structure steer (calm→storm) works w/ plain hard cut; drastic sun→night needs full clean+noisy cache reset (brief gap). |
| **19 rf_diverse_transitions** | best RF method on many diverse drastic pairs? | ✅ recache-seeded full reset transitions all 7 (meadow→wolf-night, car→neon-city, beach→storm, jungle→desert, spring→volcano, reef→peak, city→forest); brief reset gap intrinsic to RF. |

## Headline takeaways

1. **The swap mechanism is clean** (text K/V is a separate cache); the hard part is
   the *self-attn KV cache of past frames*, which carries old-world momentum.
2. **Transition speed ≈ how fast the rolling window flushes old frames.** The window
   size is the swap-speed knob.
3. **Two failure axes**: too much retained history → *resist*; too little → *jump-cut*.
   The fix that resolves both is **KV-Recache** (keep the frames, recompute their
   K/V under the new prompt) — LongLive (arxiv 2509.22622), reproduced training-free.
4. **Positioning**: prompt-switching itself is solved by LongLive (trained). Our
   project's novelty is the orthogonal **coarse-to-fine spectral speedup** (SPEED)
   that makes the streaming loop fast enough for *instant* interrupts.

See `../findings/` for the full write-ups (ARCH_MAP, SPEEDUP_NOTES, FINDINGS,
MODEL_SURVEY, SWAP_PLAN). Scripts in `../scripts/`.
