"""Before/after benchmark for SPEED on Self-Forcing.

Runs the same prompt+seed twice -- baseline and SPEED-enabled -- reports fps for
each, and saves both videos for visual comparison.
"""
import argparse, time, torch
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange
from pipeline import CausalInferencePipeline
from utils.misc import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/self_forcing_dmd.pt")
ap.add_argument("--config", default="configs/self_forcing_dmd.yaml")
ap.add_argument("--frames", type=int, default=21)
ap.add_argument("--scale", type=float, default=0.5)
ap.add_argument("--lowres_steps", type=int, default=2)
ap.add_argument("--outdir", default="results_speed_cmp")
args = ap.parse_args()

import os
os.makedirs(args.outdir, exist_ok=True)
device = "cuda"
torch.set_grad_enabled(False)

config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                         OmegaConf.load(args.config))
pipe = CausalInferencePipeline(config, device=device)
sd = torch.load(args.ckpt, map_location="cpu")
pipe.generator.load_state_dict(sd["generator" if "generator" in sd else "generator_ema"])
pipe = pipe.to(dtype=torch.bfloat16)
pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
print(f"local_attn_size={pipe.local_attn_size}  num_frame_per_block={pipe.num_frame_per_block}  "
      f"denoising_steps={len(pipe.denoising_step_list)}")

prompts = ["a corgi running on a sunny beach, waves crashing, cinematic"]


def run(tag, use_speed):
    pipe.use_speed = use_speed
    pipe.speed_scale = args.scale
    pipe.speed_lowres_steps = args.lowres_steps
    set_seed(0)
    noise = torch.randn([1, args.frames, 16, 60, 104], device=device, dtype=torch.bfloat16)
    # warmup (compiles + fills caches once)
    set_seed(0)
    _ = pipe.inference(noise=noise, text_prompts=prompts, return_latents=False)
    pipe.vae.model.clear_cache()
    torch.cuda.synchronize(); t = time.time()
    set_seed(0)
    video = pipe.inference(noise=noise, text_prompts=prompts, return_latents=False, profile=True)
    torch.cuda.synchronize(); dt = time.time() - t
    pipe.vae.model.clear_cache()
    nframes = video.shape[1]
    fps = nframes / dt
    print(f"\n=== {tag}: frames={nframes} time={dt:.2f}s fps={fps:.2f} ===\n")
    vid = (255.0 * rearrange(video, 'b t c h w -> b t h w c')).clamp(0, 255).to(torch.uint8).cpu()[0]
    path = os.path.join(args.outdir, f"{tag}.mp4")
    write_video(path, vid, fps=16)
    print(f"saved {path}")
    return fps, dt


print("######## BASELINE ########")
base_fps, base_t = run("baseline", use_speed=False)
print("######## SPEED ########")
speed_fps, speed_t = run("speed", use_speed=True)

print("\n================ SUMMARY ================")
print(f"baseline: {base_fps:.2f} fps ({base_t:.2f}s)")
print(f"speed   : {speed_fps:.2f} fps ({speed_t:.2f}s)")
print(f"speedup : {base_t / speed_t:.2f}x")
