"""SPEED profiling for the CHUNKWISE 4-step streaming model (gino's live model).

Chunkwise: num_frame_per_block=3, denoising_step_list=[1000,750,500,250]. Each forward
carries 3x the tokens of the frame-wise model (more compute-bound) and there are 4 steps,
so SPEED can run up to 3 leading steps at low res -> a much larger share of forwards
cheapened than the 2-step frame-wise model.

  A) CONFIG SWEEP @ 48f  - baseline vs SPEED (scale x lowres_steps), warmup+trials,
                           speedup, TTFB(lock), and quality: no-reference sharpness
                           (does SPEED blur?) + divergence-from-baseline (PSNR/SSIM/relL2,
                           confounded by stochastic AR trajectory -- reported as context).
  B) PER-PHASE @ 48f     - GPU time per call class (low / full / ctx) at full window.
  C) LENGTH SCAN         - baseline vs best config across 21/48/96 frames.

    CUDA_VISIBLE_DEVICES=0 /data/SPED/gino/Self-Forcing/.venv/bin/python profile_chunkwise.py
"""
import os, sys, json, time, statistics
import numpy as np, torch
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf_streaming import StreamingCF, load_cf_pipeline, CF_DIR

FRAMES = 48          # multiple of 3 (chunkwise block size)
TRIALS = 2
SEED = 0
PROMPT = ("A fluffy golden retriever sprinting through a sunlit meadow of orange "
          "wildflowers, warm golden daylight, cinematic, photorealistic, 4k")

print("loading CHUNKWISE 4-step pipeline...", flush=True)
pipe = load_cf_pipeline(model="chunkwise")
DEV = next(pipe.generator.parameters()).device
CTX_T = int(pipe.args.context_noise)
print(f"  nfpb={pipe.num_frame_per_block}  steps={list(map(int,pipe.denoising_step_list))}", flush=True)


def rollout(use_speed, scale, lowres, frames, seed=SEED, want_latents=False):
    pipe.use_speed = use_speed; pipe.speed_scale = scale; pipe.speed_lowres_steps = lowres
    gen = StreamingCF(pipe, seed=seed)
    frames -= frames % gen.nfpb
    gen.start(PROMPT, total_frames=frames); pipe.vae.model.clear_cache()
    n_blocks = frames // gen.nfpb
    lat = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    d = gen.step()
    torch.cuda.synchronize(); ttfb = time.perf_counter() - t0
    if want_latents: lat.append(d.float().cpu())
    for _ in range(1, n_blocks):
        d = gen.step()
        if want_latents: lat.append(d.float().cpu())
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    return dt, ttfb, frames, (torch.cat(lat, dim=1) if want_latents else None)


def timed(use_speed, scale, lowres, frames):
    rollout(use_speed, scale, lowres, frames)  # warmup
    ds, ts = [], []
    lat = None
    for k in range(TRIALS):
        d, t, n, l = rollout(use_speed, scale, lowres, frames, want_latents=(k == 0))
        ds.append(d); ts.append(t)
        if k == 0: lat = l
    return dict(s=statistics.mean(ds), std=statistics.pstdev(ds) if len(ds) > 1 else 0.0,
                ttfb=statistics.mean(ts), fps=frames / statistics.mean(ds)), lat


def decode(latents):
    gen = StreamingCF(pipe, seed=SEED); pipe.vae.model.clear_cache()
    out = []
    nf = latents.shape[1]
    for i in range(0, nf, pipe.num_frame_per_block):
        d = latents[:, i:i+pipe.num_frame_per_block].to(DEV).to(torch.bfloat16)
        out.append(gen.decode_chunk(d))
    return np.concatenate(out, axis=0)


def sharpness(vid_u8):
    """No-reference sharpness: mean gradient magnitude on luma (higher = sharper)."""
    g = vid_u8.astype(np.float32).mean(axis=3)  # [F,H,W]
    gx = np.abs(np.diff(g, axis=2)).mean()
    gy = np.abs(np.diff(g, axis=1)).mean()
    return float(gx + gy)


def divergence(ref, test):
    a, b = ref.astype(np.float32), test.astype(np.float32)
    mse = ((a - b) ** 2).mean()
    psnr = 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)
    ss = float(np.mean([ssim(ref[i], test[i], channel_axis=2, data_range=255) for i in range(ref.shape[0])]))
    return float(psnr), ss


results = {"meta": {"model": "chunkwise", "frames": FRAMES, "trials": TRIALS,
                    "nfpb": pipe.num_frame_per_block,
                    "steps": list(map(int, pipe.denoising_step_list))}}

# ---------------- A) CONFIG SWEEP ----------------
print(f"\n{'='*86}\nA) CHUNKWISE CONFIG SWEEP @ {FRAMES}f, {TRIALS} trials\n{'='*86}", flush=True)
base, base_lat = timed(False, 1.0, 1, FRAMES)
base_vid = decode(base_lat); base_sharp = sharpness(base_vid)
print(f"  baseline   : {base['s']:.3f}s (±{base['std']:.3f})  {base['fps']:.2f} FPS  "
      f"TTFB={base['ttfb']*1e3:.0f}ms  sharp={base_sharp:.2f}", flush=True)
sweep = {"baseline": {**base, "speedup": 1.0, "sharp": base_sharp}}
CONFIGS = [
    ("s0.50_l1", 0.50, 1), ("s0.50_l2", 0.50, 2), ("s0.50_l3", 0.50, 3),
    ("s0.40_l2", 0.40, 2), ("s0.33_l2", 0.33, 2), ("s0.33_l3", 0.33, 3),
]
for name, sc, lr in CONFIGS:
    st, lat = timed(True, sc, lr, FRAMES)
    vid = decode(lat); sh = sharpness(vid)
    psnr, ss = divergence(base_vid, vid)
    rel = float((lat - base_lat).norm() / base_lat.norm())
    low = pipe._speed_lowres_dims(60, 104)
    sp = base['s'] / st['s']
    sweep[name] = {**st, "speedup": sp, "pct": (sp-1)*100, "low_dims": list(low),
                   "sharp": sh, "sharp_ratio": sh/base_sharp, "psnr": psnr, "ssim": ss, "rel_l2": rel}
    print(f"  {name:10s}: {st['s']:.3f}s (±{st['std']:.3f})  {st['fps']:.2f} FPS  "
          f"{(sp-1)*100:+.1f}% ({sp:.3f}x)  TTFB={st['ttfb']*1e3:.0f}ms  low={tuple(low)}  "
          f"sharp={sh:.2f}({sh/base_sharp*100:.0f}%) PSNR={psnr:.1f} SSIM={ss:.2f} relL2={rel:.2f}", flush=True)
results["sweep"] = sweep

# pick best by speedup among configs that keep sharpness >= 90% of baseline
ok = [(n, v) for n, v in sweep.items() if n != "baseline" and v["sharp_ratio"] >= 0.90]
best_name = max(ok, key=lambda kv: kv[1]["speedup"])[0] if ok else "s0.50_l2"
bsc, blr = dict([(n, (s, l)) for n, s, l in CONFIGS])[best_name]
print(f"\n  -> best (>=90% sharpness): {best_name}  ({sweep[best_name]['pct']:+.1f}%)", flush=True)

# ---------------- B) PER-PHASE ----------------
print(f"\n{'='*86}\nB) PER-PHASE GPU TIMING @ {FRAMES}f (baseline vs {best_name})\n{'='*86}", flush=True)
class PhaseTimer:
    def __init__(self, m): self.m=m; self.orig=m.forward; self.events=[]; self.on=False
    def __enter__(self):
        def w(*a, **k):
            if not self.on: return self.orig(*a, **k)
            ts=k.get("timestep"); tv=int(ts.flatten()[0].item()) if ts is not None else -1
            tag="low" if k.get("full_hw") is not None else ("ctx" if tv==CTX_T else "full")
            s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            s.record(); o=self.orig(*a,**k); e.record(); self.events.append((tag,s,e)); return o
        self.m.forward=w; return self
    def __exit__(self,*x): self.m.forward=self.orig
    def summ(self):
        torch.cuda.synchronize(); agg={}
        for t,s,e in self.events:
            d=agg.setdefault(t,[0.0,0]); d[0]+=s.elapsed_time(e); d[1]+=1
        return {t:{"total_ms":round(v[0],0),"calls":v[1],"ms_per_call":round(v[0]/max(v[1],1),1)} for t,v in agg.items()}
phase={}
for name,us,sc,lr in [("baseline",False,1.0,1),(best_name,True,bsc,blr)]:
    rollout(us,sc,lr,FRAMES)
    pt=PhaseTimer(pipe.generator)
    with pt:
        pt.on=True; rollout(us,sc,lr,FRAMES); pt.on=False
    s=pt.summ(); phase[name]=s
    tot=sum(v["total_ms"] for v in s.values())
    print(f"  {name:10s} gpu_total={tot:.0f}ms | "+"  ".join(f"{t}:{v['total_ms']:.0f}ms/{v['calls']}={v['ms_per_call']}ms" for t,v in sorted(s.items())), flush=True)
results["per_phase"]=phase

# ---------------- C) LENGTH SCAN ----------------
print(f"\n{'='*86}\nC) LENGTH SCAN (baseline vs {best_name})\n{'='*86}", flush=True)
scan={}
for F in [21, 48, 96]:
    b,_=timed(False,1.0,1,F); s,_=timed(True,bsc,blr,F)
    sp=b['s']/s['s']
    scan[F]={"base_s":round(b['s'],3),"speed_s":round(s['s'],3),"base_fps":round(b['fps'],2),
             "speed_fps":round(s['fps'],2),"speedup":round(sp,4),"pct":round((sp-1)*100,1)}
    print(f"  {F:3d}f: base {b['s']:.3f}s ({b['fps']:.1f} FPS) | speed {s['s']:.3f}s ({s['fps']:.1f} FPS) | {(sp-1)*100:+.1f}% ({sp:.3f}x)", flush=True)
results["length_scan"]=scan

out=os.path.join(CF_DIR,"out","profile_chunkwise.json")
with open(out,"w") as f: json.dump(results,f,indent=2)
print(f"\nwrote {out}\nCHUNKWISE PROFILE DONE", flush=True)
