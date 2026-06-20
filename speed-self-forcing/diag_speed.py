import argparse, torch, cv2, numpy as np
from omegaconf import OmegaConf
from einops import rearrange
from pipeline import CausalInferencePipeline
from utils.misc import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.75, 0.5])
ap.add_argument("--lowres_steps", type=int, default=2)
args = ap.parse_args()

device = "cuda"; torch.set_grad_enabled(False)
config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                         OmegaConf.load("configs/self_forcing_dmd.yaml"))
pipe = CausalInferencePipeline(config, device=device)
sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
pipe.generator.load_state_dict(sd["generator" if "generator" in sd else "generator_ema"])
pipe = pipe.to(dtype=torch.bfloat16)
pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
prompts = ["a corgi running on a sunny beach, waves crashing, cinematic"]

def gen(use_speed, scale):
    pipe.use_speed = use_speed; pipe.speed_scale = scale; pipe.speed_lowres_steps = args.lowres_steps
    set_seed(0)
    noise = torch.randn([1, 21, 16, 60, 104], device=device, dtype=torch.bfloat16)
    set_seed(0)
    v = pipe.inference(noise=noise, text_prompts=prompts, return_latents=False)
    pipe.vae.model.clear_cache()
    return (255.0*rearrange(v,'b t c h w -> b t h w c')).clamp(0,255).to(torch.uint8).cpu()[0].numpy()

base = gen(False, 1.0)
cv2.imwrite("/tmp/diag_baseline.png", cv2.cvtColor(base[40], cv2.COLOR_RGB2BGR))
for s in args.scales:
    out = gen(True, s)
    d = np.abs(out.astype(np.float32)-base.astype(np.float32)).mean()
    cv2.imwrite(f"/tmp/diag_speed_{s}.png", cv2.cvtColor(out[40], cv2.COLOR_RGB2BGR))
    print(f"scale={s}: MAE vs baseline = {d:.2f}  (low dims {pipe._speed_lowres_dims(60,104)})", flush=True)
print("DIAG DONE")
