"""Quick baseline throughput benchmark for the Self-Forcing causal pipeline.

Loads self_forcing_dmd, runs one warmup generation (triggers flex-attention
max-autotune compile), then a timed profiled generation. Reports fps =
generated video frames / diffusion time.
"""
import argparse, time, torch
from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline
from utils.misc import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/self_forcing_dmd.pt")
ap.add_argument("--config", default="configs/self_forcing_dmd.yaml")
ap.add_argument("--frames", type=int, default=21)  # latent frames
args = ap.parse_args()

device = "cuda"
set_seed(0)
torch.set_grad_enabled(False)

config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                         OmegaConf.load(args.config))
pipe = CausalInferencePipeline(config, device=device)
sd = torch.load(args.ckpt, map_location="cpu")
gen_key = "generator" if "generator" in sd else "generator_ema"
print(f"loading weights from key '{gen_key}'")
pipe.generator.load_state_dict(sd[gen_key])
pipe = pipe.to(dtype=torch.bfloat16)
pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)

prompts = ["a corgi running on a sunny beach, waves crashing, cinematic"]
noise = torch.randn([1, args.frames, 16, 60, 104], device=device, dtype=torch.bfloat16)

print("=== warmup (compiles flex-attention, may take minutes) ===", flush=True)
t = time.time()
v, _ = pipe.inference(noise=noise, text_prompts=prompts, return_latents=True)
pipe.vae.model.clear_cache()
torch.cuda.synchronize()
print(f"warmup done in {time.time()-t:.1f}s", flush=True)

print("=== timed run (profile) ===", flush=True)
torch.cuda.synchronize(); t = time.time()
v, lat = pipe.inference(noise=noise, text_prompts=prompts, return_latents=True, profile=True)
torch.cuda.synchronize(); dt = time.time() - t
pipe.vae.model.clear_cache()

nframes = v.shape[1]
print(f"\n=== RESULT ===")
print(f"video {tuple(v.shape)} latents {tuple(lat.shape)}")
print(f"generated_video_frames={nframes}  total_time={dt:.2f}s  fps={nframes/dt:.2f}")
