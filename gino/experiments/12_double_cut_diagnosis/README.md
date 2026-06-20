# 12 — Root-Cause of the Recache Double-Cut

Observed failure (recache models only): sun → *cut* → winter-scene-A → *cut* →
winter-scene-B → stable. Two cuts, not one. Diagnosed by measuring per-frame pixel
deltas around the swap and varying one knob at a time.

## Diagnostic (cut locations, pixel : Δ)

| config | cuts | cut 2 at |
|---|---|---|
| window 3 | 116, **128** | swap **+3** latent |
| window 6 | 117, **141/144** | swap **+6** latent |
| window 9 (no grow) | **152/156** | swap **+9** latent |
| full window 21 | all Δ≈12, no 2nd cut | — (inertia instead) |
| recache 3 vs 21 | both cut2 @128 | unchanged by recache size |

**Cut 2 tracks the WINDOW size, not the recache size.** That is the proof.

## Mechanism

Recache rebuilds the cache from recent *old-prompt* frames **re-encoded under the
new prompt** → those cached frames carry the **old scene's STRUCTURE** with new
semantics. So:
1. **Cut 1 (swap):** new frames attend to them → "recolored old scene" (scene A).
2. **Cut 2 (swap + window):** the small window **evicts** those old-structure
   frames as a block; the window then holds only native-new frames (native-new
   structure) → re-anchor → scene B.
3. Stabilizes.

`full window 21` keeps the bridge forever → no cut 2, but inertia (weak swap).
Only recache injects this bridge → only recache double-cuts. Fix in **13**.
