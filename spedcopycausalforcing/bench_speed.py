"""A/B bench for SPEED in the CF++ framewise-2step STREAMING path.

Loads the pipeline once, then generates the same clip with SPEED off and on,
reporting wall-clock, FPS, and mean abs latent difference (a rough quality proxy).
Writes one mp4 per mode for visual inspection.

    CUDA_VISIBLE_DEVICES=0 /data/SPED/gino/Self-Forcing/.venv/bin/python bench_speed.py --seconds 4
"""
import os, sys, time, argparse
import numpy as np, imageio, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf_streaming import StreamingCF, load_cf_pipeline, CF_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=4.0)
ap.add_argument("--speed_scale", type=float, default=0.5)
ap.add_argument("--speed_lowres_steps", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

PROMPT = ("A fluffy golden retriever sprinting through a sunlit meadow of orange "
          "wildflowers, warm golden daylight, cinematic, photorealistic, 4k")

print("loading CF++ framewise-2step pipeline...")
pipe = load_cf_pipeline()  # SPEED toggled per-run below


def run(use_speed):
    pipe.use_speed = use_speed
    pipe.speed_scale = args.speed_scale
    pipe.speed_lowres_steps = args.speed_lowres_steps
    gen = StreamingCF(pipe, seed=args.seed)
    total = max(gen.nfpb, int(round(args.seconds * 4)))
    total -= total % gen.nfpb
    gen.start(PROMPT, total_frames=total)
    pipe.vae.model.clear_cache()
    n_chunks = total // gen.nfpb
    # warm up one chunk (kernels/compile) is implicit; time the full rollout
    torch.cuda.synchronize(); t0 = time.time()
    lat = []
    for c in range(n_chunks):
        den = gen.step()
        lat.append(den.float().cpu())
    torch.cuda.synchronize(); dt = time.time() - t0
    # decode for visual check
    frames = []
    pipe.vae.model.clear_cache()
    # re-decode from stored latents
    for d in lat:
        frames.append(gen.decode_chunk(d.to(gen.device).to(torch.bfloat16)))
    frames = np.concatenate(frames, axis=0)
    return dt, frames, torch.cat(lat, dim=1)


print("\n=== BASELINE (SPEED off) ===")
dt0, f0, l0 = run(False)
print(f"  {f0.shape[0]} frames in {dt0:.2f}s = {f0.shape[0]/dt0:.2f} FPS")

print("\n=== SPEED on ===")
dt1, f1, l1 = run(True)
print(f"  {f1.shape[0]} frames in {dt1:.2f}s = {f1.shape[0]/dt1:.2f} FPS")

mae = (l0 - l1).abs().mean().item()
out0 = os.path.join(CF_DIR, "out", "bench_baseline.mp4")
out1 = os.path.join(CF_DIR, "out", "bench_speed.mp4")
os.makedirs(os.path.dirname(out0), exist_ok=True)
imageio.mimwrite(out0, f0, fps=16, codec="libx264", macro_block_size=1)
imageio.mimwrite(out1, f1, fps=16, codec="libx264", macro_block_size=1)

print("\n=== SUMMARY ===")
print(f"  speedup (denoise wall-clock): {dt0/dt1:.2f}x   ({dt0:.2f}s -> {dt1:.2f}s)")
print(f"  latent MAE (baseline vs speed): {mae:.4f}")
print(f"  low-res dims @ scale {args.speed_scale}: {pipe._speed_lowres_dims(60,104)} (full 60x104)")
print(f"  wrote {out0}\n        {out1}")
print("BENCH DONE")
