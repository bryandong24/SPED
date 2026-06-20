# 07 — Control-Theoretic / Mechinterp Cache Constructions

Framing: prompt swapping is a **control problem** — state = KV cache (has momentum
= inertia), control = prompt/cache edits, jump-cuts/flicker = transients from an
abrupt (step) input. The prompt enters ONLY via cross-attn, accumulating in the
residual stream → both self-attn K and V. Three principled ways to rebuild the
cache "faithful to old structure, old prompt removed", each combined with the
grow-window schedule (06c). 15s, sun→night, swap at 7.5s.

Metric: **post-swap jitter** = mean frame-to-frame pixel delta (lower = more
stable, the thing we were trying to fix) + warmth (sun + / night −).

| File | construction | warmth @0→100% | jitter ↓ |
|---|---|---|---|
| `ctrl_new_w5` | plain recache, window 5 (the original method) | 66→−33 | **20.2** (worst) |
| `ctrl_ortho_w5` | orthogonal-projection neutralize old-prompt dir | 81→−15 | 20.0 (crude, weak) |
| `ctrl_kswap_w5` | ablation K_new + V_old | 75→−17 | 15.3 |
| `ctrl_null_w5` | null-prompt recache (neutral structure), window 5 | 56→−44 | 13.0 |
| `ctrl_vswap_w5` | **K-preserve / V-swap**, window 5 | 63→−31 | 13.1 |
| `ctrl_vswap_grow` | **K-preserve/V-swap + grow-window** | 62→**−51** | **6.96** ★ |
| `ctrl_null_grow` | **null-recache + grow-window** | 67→−39 | **5.70** ★ |

## Findings

- **Plain recache is the jitteriest** (20.2) — quantitatively confirms the
  flicker/jump-cut problem. The fancy constructions + grow-window cut it ~3×.
- **K-preserve / V-swap works** (`vswap`): keep K (attention routing = motion/
  structure), swap V (content). Strongest scene transition with low jitter.
  Mechinterp hypothesis (K=where you attend, V=what you read) validated.
- **null-recache** (prompt-neutral structure + let AR inject new semantics) is the
  **most stable** and tends to **preserve the original subject's identity** (same
  dog, moved to night) vs vswap which commits harder to the new scene (snow/husky).
- **Orthogonal projection underperformed** here — the per-(token,head) rank-1
  neutralization was too crude / distorted K,V. A low-rank (SVD across positions)
  or value-only projection is the obvious refinement, not yet tried.
- **The grow-window schedule (06c) is the dominant stability lever**; the cache
  construction (vswap vs null) then trades scene-commitment vs identity-preservation.

**Recommended config:** `vswap` (full scene change) or `null` (keep subject) +
grow-window 3→15. Caveat: first-order approximations of a nonlinear system, training-
free; LongLive-style fine-tuning would push further. `scripts/swap_control.py`.
