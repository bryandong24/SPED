"""Rigorous SPEED profiling for the CF++ framewise-2step STREAMING path.

Loads the model once and runs:
  A) CONFIG SWEEP  - baseline vs SPEED at several (speed_scale, speed_lowres_steps),
                     timing with warmup + repeats (mean/std/min), TTFC, and fidelity
                     metrics vs baseline (PSNR / SSIM / pixel-MAE / latent rel-L2).
  B) LENGTH SCAN   - baseline vs best SPEED config across clip lengths (does the win
                     grow/shrink as the rolling KV window fills?).
  C) PER-CALL MICRO- isolated single generator() forward, full-res vs each low-res
                     scale, with a warm KV cache (explains the end-to-end ceiling).

Timing excludes VAE decode (decode is done separately, only for quality).
All wall-clock uses torch.cuda.synchronize() + perf_counter.

    CUDA_VISIBLE_DEVICES=0 /data/SPED/gino/Self-Forcing/.venv/bin/python profile_speed.py
"""
import os, sys, json, time, argparse, statistics
import numpy as np, torch
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf_streaming import StreamingCF, load_cf_pipeline, CF_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=32, help="latent frames for the main sweep (~4/sec)")
ap.add_argument("--trials", type=int, default=3, help="timed trials per config (after 1 warmup)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(CF_DIR, "out", "profile_results.json"))
args = ap.parse_args()

PROMPT = ("A fluffy golden retriever sprinting through a sunlit meadow of orange "
          "wildflowers, warm golden daylight, cinematic, photorealistic, 4k")

print("loading CF++ framewise-2step pipeline...", flush=True)
pipe = load_cf_pipeline()
DEV = next(pipe.generator.parameters()).device


def rollout(use_speed, scale, lowres, frames, seed=args.seed, want_latents=False):
    """One full streaming rollout. Returns (denoise_s, ttfc_s, n_frames, latents|None)."""
    pipe.use_speed = use_speed
    pipe.speed_scale = scale
    pipe.speed_lowres_steps = lowres
    gen = StreamingCF(pipe, seed=seed)
    total = frames - (frames % gen.nfpb)
    gen.start(PROMPT, total_frames=total)
    pipe.vae.model.clear_cache()
    n_chunks = total // gen.nfpb
    lat = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    # first chunk (TTFC)
    d = gen.step()
    torch.cuda.synchronize(); ttfc = time.perf_counter() - t0
    if want_latents: lat.append(d.float().cpu())
    for _ in range(1, n_chunks):
        d = gen.step()
        if want_latents: lat.append(d.float().cpu())
    torch.cuda.synchronize(); denoise = time.perf_counter() - t0
    latents = torch.cat(lat, dim=1) if want_latents else None
    return denoise, ttfc, total, latents


def decode_latents(latents):
    """Decode [1,F,C,H,W] latents to uint8 frames [F,H,W,3] via streaming VAE."""
    gen = StreamingCF(pipe, seed=args.seed)
    pipe.vae.model.clear_cache()
    frames = []
    for f in range(latents.shape[1]):
        d = latents[:, f:f+1].to(DEV).to(torch.bfloat16)
        frames.append(gen.decode_chunk(d))
    return np.concatenate(frames, axis=0)


def fidelity(ref_u8, test_u8):
    """PSNR / SSIM / pixel-MAE between two uint8 videos [F,H,W,3]."""
    a = ref_u8.astype(np.float32); b = test_u8.astype(np.float32)
    mae = np.abs(a - b).mean()
    mse = ((a - b) ** 2).mean()
    psnr = 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)
    ss = float(np.mean([
        ssim(ref_u8[i], test_u8[i], channel_axis=2, data_range=255)
        for i in range(ref_u8.shape[0])]))
    return dict(psnr=float(psnr), ssim=ss, pixel_mae=float(mae))


def timed(use_speed, scale, lowres, frames):
    """Warmup once, then `args.trials` timed rollouts. Returns stats + latents(1st trial)."""
    rollout(use_speed, scale, lowres, frames, want_latents=False)  # warmup (compile/autotune)
    ds, ts = [], []
    lat = None
    for k in range(args.trials):
        d, t, n, l = rollout(use_speed, scale, lowres, frames, want_latents=(k == 0))
        ds.append(d); ts.append(t)
        if k == 0: lat = l
    return dict(
        denoise_mean=statistics.mean(ds), denoise_std=(statistics.pstdev(ds) if len(ds) > 1 else 0.0),
        denoise_min=min(ds), ttfc_mean=statistics.mean(ts),
        fps=frames / statistics.mean(ds), n=frames), lat


results = {"meta": {"frames": args.frames, "trials": args.trials, "device": str(DEV),
                    "schedule": list(map(int, pipe.denoising_step_list)),
                    "first_chunk": list(map(int, pipe.denoising_step_list_first_chunk))}}

# ----------------- A) CONFIG SWEEP -----------------
print(f"\n{'='*78}\nA) CONFIG SWEEP @ {args.frames} frames, {args.trials} trials\n{'='*78}", flush=True)
SWEEP = [
    ("baseline",     False, 1.0,  1),
    ("s0.50_l1",     True,  0.50, 1),
    ("s0.40_l1",     True,  0.40, 1),
    ("s0.33_l1",     True,  0.33, 1),
    ("s0.25_l1",     True,  0.25, 1),
    ("s0.50_l2",     True,  0.50, 2),
    ("s0.50_l3",     True,  0.50, 3),
    ("s0.33_l3",     True,  0.33, 3),
]
sweep = {}
base_stats, base_lat = timed(False, 1.0, 1, args.frames)
base_vid = decode_latents(base_lat)
sweep["baseline"] = {**base_stats, "speedup": 1.0,
                     "low_dims": None, **{"psnr": 99.0, "ssim": 1.0, "pixel_mae": 0.0}}
print(f"  baseline: {base_stats['denoise_mean']:.3f}s "
      f"(±{base_stats['denoise_std']:.3f}) {base_stats['fps']:.2f} FPS  TTFC={base_stats['ttfc_mean']*1e3:.0f}ms", flush=True)

for name, us, sc, lr in SWEEP[1:]:
    st, lat = timed(us, sc, lr, args.frames)
    vid = decode_latents(lat)
    fid = fidelity(base_vid, vid)
    rel_l2 = float((lat - base_lat).norm() / base_lat.norm())
    low = pipe._speed_lowres_dims(60, 104)
    sp = base_stats["denoise_mean"] / st["denoise_mean"]
    sweep[name] = {**st, "speedup": sp, "low_dims": list(low), "latent_rel_l2": rel_l2, **fid}
    print(f"  {name:10s}: {st['denoise_mean']:.3f}s (±{st['denoise_std']:.3f}) "
          f"{st['fps']:.2f} FPS  speedup={sp:.3f}x  TTFC={st['ttfc_mean']*1e3:.0f}ms  "
          f"low={tuple(low)}  PSNR={fid['psnr']:.1f}dB SSIM={fid['ssim']:.3f} relL2={rel_l2:.3f}", flush=True)
results["config_sweep"] = sweep

# ----------------- B) LENGTH SCAN -----------------
print(f"\n{'='*78}\nB) LENGTH SCAN (baseline vs s0.50_l1)\n{'='*78}", flush=True)
lengths = [16, 32, 48]
scan = {}
for F in lengths:
    b, _ = timed(False, 1.0, 1, F)
    s, _ = timed(True, 0.50, 1, F)
    sp = b["denoise_mean"] / s["denoise_mean"]
    scan[F] = {"base_s": b["denoise_mean"], "speed_s": s["denoise_mean"],
               "base_fps": b["fps"], "speed_fps": s["fps"], "speedup": sp}
    print(f"  {F:2d} frames: base {b['denoise_mean']:.3f}s ({b['fps']:.1f} FPS) | "
          f"speed {s['denoise_mean']:.3f}s ({s['fps']:.1f} FPS) | speedup {sp:.3f}x", flush=True)
results["length_scan"] = scan

# ----------------- C) PER-CALL MICRO -----------------
print(f"\n{'='*78}\nC) PER-CALL MICRO (single forward, warm KV cache @ frame 10)\n{'='*78}", flush=True)
# warm a cache by rolling 10 full-res frames, then time one forward at full vs low res.
def micro(scale):
    gen = StreamingCF(pipe, seed=args.seed)
    pipe.use_speed = False
    gen.start(PROMPT, total_frames=48)
    for _ in range(10): gen.step()                       # fill window
    cs = gen.cur_frame * gen.fsl
    ph, pw = pipe.generator.model.patch_size[1], pipe.generator.model.patch_size[2]
    full_hw = (gen.H // ph, gen.W // pw)
    if scale >= 1.0:
        x = torch.randn([1, 1, gen.C, gen.H, gen.W], device=DEV, dtype=torch.bfloat16); fhw = None
    else:
        lh, lw = pipe._speed_lowres_dims(gen.H, gen.W)
        x = torch.randn([1, 1, gen.C, lh, lw], device=DEV, dtype=torch.bfloat16); fhw = full_hw
    ts = torch.ones([1, 1], device=DEV, dtype=torch.int64) * 1000
    call = lambda: pipe.generator(noisy_image_or_video=x, conditional_dict=gen.cond, timestep=ts,
                                  kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                                  current_start=cs, current_start_frame=gen.cur_frame, full_hw=fhw)
    for _ in range(5): call()                            # warmup compile
    torch.cuda.synchronize(); t0 = time.perf_counter()
    N = 30
    for _ in range(N): call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N * 1e3          # ms/call

mic = {}
full_ms = micro(1.0)
mic["full_1.00"] = full_ms
print(f"  full-res (60x104, 1560 tok): {full_ms:.2f} ms/call", flush=True)
for sc in [0.50, 0.40, 0.33, 0.25]:
    m = micro(sc)
    lh, lw = pipe._speed_lowres_dims(60, 104) if sc == pipe.speed_scale else (None, None)
    pipe.speed_scale = sc
    lh, lw = pipe._speed_lowres_dims(60, 104)
    toks = (lh // 2) * (lw // 2)
    mic[f"low_{sc:.2f}"] = {"ms": m, "low_dims": [lh, lw], "tokens": int(toks), "callspeedup": full_ms / m}
    print(f"  low-res {sc:.2f} ({lh}x{lw}, {toks} tok): {m:.2f} ms/call  -> {full_ms/m:.2f}x per call", flush=True)
results["per_call_micro"] = mic

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {args.out}\nPROFILE DONE", flush=True)
