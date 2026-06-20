"""Follow-up SPEED profiling: WHERE the time goes, and does the win reach 10%.

D) PER-PHASE TIMING - wrap generator() (non-invasively) and accumulate GPU time per
   call class: low-res denoise / full-res denoise / clean-context commit. Shows whether
   the low-res forward is actually cheaper at steady state (full KV window).
E) LONG-VIDEO SCAN  - baseline vs SPEED across 32/48/64/96 frames; does speedup climb
   toward 10% as the rolling KV window stays full longer?

    CUDA_VISIBLE_DEVICES=0 /data/SPED/gino/Self-Forcing/.venv/bin/python profile_speed2.py
"""
import os, sys, json, time, statistics
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf_streaming import StreamingCF, load_cf_pipeline, CF_DIR

PROMPT = ("A fluffy golden retriever sprinting through a sunlit meadow of orange "
          "wildflowers, warm golden daylight, cinematic, photorealistic, 4k")
print("loading pipeline...", flush=True)
pipe = load_cf_pipeline()
DEV = next(pipe.generator.parameters()).device
CTX_T = int(pipe.args.context_noise)


# ---- non-invasive timing wrapper around the generator forward ----
class PhaseTimer:
    def __init__(self, gen_module):
        self.m = gen_module
        self.orig = gen_module.forward
        self.events = []   # (tag, start_evt, end_evt)
        self.on = False
    def __enter__(self):
        def wrapped(*a, **k):
            if not self.on:
                return self.orig(*a, **k)
            ts = k.get("timestep")
            tsval = int(ts.flatten()[0].item()) if ts is not None else -1
            full_hw = k.get("full_hw", None)
            tag = "low" if full_hw is not None else ("ctx" if tsval == CTX_T else "full")
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = self.orig(*a, **k)
            e.record()
            self.events.append((tag, s, e))
            return out
        self.m.forward = wrapped
        return self
    def __exit__(self, *x):
        self.m.forward = self.orig
    def summary(self):
        torch.cuda.synchronize()
        agg = {}
        for tag, s, e in self.events:
            ms = s.elapsed_time(e)
            d = agg.setdefault(tag, [0.0, 0])
            d[0] += ms; d[1] += 1
        return {t: {"total_ms": round(v[0], 1), "calls": v[1],
                    "ms_per_call": round(v[0] / max(v[1], 1), 2)} for t, v in agg.items()}


def rollout(use_speed, scale, lowres, frames, seed=0):
    pipe.use_speed = use_speed; pipe.speed_scale = scale; pipe.speed_lowres_steps = lowres
    gen = StreamingCF(pipe, seed=seed); gen.start(PROMPT, total_frames=frames)
    pipe.vae.model.clear_cache()
    for _ in range(frames): gen.step()


def timed(use_speed, scale, lowres, frames, trials=3):
    rollout(use_speed, scale, lowres, frames)  # warmup
    ds = []
    for _ in range(trials):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        rollout(use_speed, scale, lowres, frames)
        torch.cuda.synchronize(); ds.append(time.perf_counter() - t0)
    return statistics.mean(ds), (statistics.pstdev(ds) if len(ds) > 1 else 0.0)


results = {}

# ---------- D) PER-PHASE TIMING @ 48 frames ----------
print(f"\n{'='*78}\nD) PER-PHASE GPU TIMING @ 48 frames (full rolling window)\n{'='*78}", flush=True)
phase = {}
for name, us, sc, lr in [("baseline", False, 1.0, 1), ("s0.50_l1", True, 0.50, 1)]:
    rollout(us, sc, lr, 48)  # warmup
    pt = PhaseTimer(pipe.generator)
    with pt:
        pt.on = True
        rollout(us, sc, lr, 48)
        pt.on = False
    s = pt.summary()
    phase[name] = s
    tot = sum(v["total_ms"] for v in s.values())
    parts = "  ".join(f"{t}: {v['total_ms']:.0f}ms/{v['calls']}={v['ms_per_call']:.1f}ms" for t, v in sorted(s.items()))
    print(f"  {name:10s} total_gpu={tot:.0f}ms | {parts}", flush=True)
results["per_phase_48f"] = phase

# ---------- E) LONG-VIDEO SCAN ----------
print(f"\n{'='*78}\nE) LONG-VIDEO SCAN (baseline vs s0.50_l1, 3 trials)\n{'='*78}", flush=True)
scan = {}
for F in [32, 48, 64, 96]:
    bm, bs = timed(False, 1.0, 1, F)
    sm, ss = timed(True, 0.50, 1, F)
    sp = bm / sm
    scan[F] = {"base_s": round(bm, 3), "base_std": round(bs, 3),
               "speed_s": round(sm, 3), "speed_std": round(ss, 3),
               "base_fps": round(F / bm, 2), "speed_fps": round(F / sm, 2),
               "speedup": round(sp, 4), "pct": round((sp - 1) * 100, 1)}
    print(f"  {F:3d} frames: base {bm:.3f}s (±{bs:.3f}) | speed {sm:.3f}s (±{ss:.3f}) "
          f"| {(sp-1)*100:+.1f}%  ({sp:.3f}x)", flush=True)
results["long_scan"] = scan

out = os.path.join(CF_DIR, "out", "profile_results2.json")
with open(out, "w") as f: json.dump(results, f, indent=2)
print(f"\nwrote {out}\nPROFILE2 DONE", flush=True)
