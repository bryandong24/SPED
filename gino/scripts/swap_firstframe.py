"""Swap after exactly K latent frames using 1-frame blocks (finest granularity).

Answers: "what if you swap right after the first sun frame?" The main model uses
3-frame chunks; here we force num_frame_per_block=1 so we can swap after a single
latent frame. Generates 21 latent frames; sun for the first K, night after.
"""
import argparse, os
import numpy as np, torch, imageio
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from pipeline import CausalInferencePipeline

FRAME_SEQ = 1560
SUN = "a dog running in a sunny green meadow full of wildflowers, bright daylight, cinematic, highly detailed"
NIGHT = "a dog running through a snowy field on a cold winter night, deep blue moonlight, falling snow, cinematic, highly detailed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap_after", type=int, default=1, help="swap after this many latent frames")
    ap.add_argument("--total", type=int, default=21)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="../out/swap_ff")
    args = ap.parse_args()
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/self_forcing_dmd.yaml"))
    pipe = CausalInferencePipeline(cfg, device=device)
    pipe.generator.load_state_dict(torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")["generator_ema"])
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
    pipe.num_frame_per_block = 1
    pipe.generator.model.num_frame_per_block = 1

    sun = pipe.text_encoder(text_prompts=[SUN])["prompt_embeds"]
    night = pipe.text_encoder(text_prompts=[NIGHT])["prompt_embeds"]

    g = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn([1, args.total, 16, 60, 104], generator=g, dtype=torch.bfloat16).to(device)
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, noise.dtype, device)
    pipe._initialize_crossattn_cache(1, noise.dtype, device)
    out = torch.zeros_like(noise)

    cond = sun
    for f in range(args.total):
        if f == args.swap_after:
            cond = night
            for c in pipe.crossattn_cache:
                c["is_init"] = False
        noisy = noise[:, f:f + 1]
        for i, ts in enumerate(pipe.denoising_step_list):
            timestep = torch.ones([1, 1], device=device, dtype=torch.int64) * ts
            _, den = pipe.generator(noisy_image_or_video=noisy, conditional_dict={"prompt_embeds": cond},
                                    timestep=timestep, kv_cache=pipe.kv_cache1,
                                    crossattn_cache=pipe.crossattn_cache, current_start=f * FRAME_SEQ)
            if i < len(pipe.denoising_step_list) - 1:
                nt = pipe.denoising_step_list[i + 1]
                noisy = pipe.scheduler.add_noise(den.flatten(0, 1), torch.randn_like(den.flatten(0, 1)),
                                                 nt * torch.ones([1], device=device, dtype=torch.long)).unflatten(0, den.shape[:2])
        out[:, f:f + 1] = den
        ctx_t = torch.ones_like(timestep) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=den, conditional_dict={"prompt_embeds": cond},
                       timestep=ctx_t, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                       current_start=f * FRAME_SEQ)

    video = (pipe.vae.decode_to_pixel(out, use_cache=False) * 0.5 + 0.5).clamp(0, 1)
    frames = (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()
    name = f"swap_after{args.swap_after}f"
    imageio.mimwrite(os.path.join(args.out_dir, f"{name}.mp4"), frames, fps=16, codec="libx264")
    idxs = np.linspace(0, frames.shape[0] - 1, 9).round().astype(int)
    swap_px = args.swap_after * 4
    tiles = [frames[i].copy() for i in idxs]
    for t, i in zip(tiles, idxs):
        if i >= swap_px: t[:5, :] = [255, 0, 0]
    Image.fromarray(np.concatenate(tiles, axis=1)).save(os.path.join(args.out_dir, f"{name}_sheet.png"))
    w = frames.reshape(frames.shape[0], -1, 3).mean(axis=1); warmth = w[:, 0] - w[:, 2]
    print(f"[{name}] warmth first/mid/last {warmth[0]:.0f}/{warmth[len(warmth)//2]:.0f}/{warmth[-1]:.0f}")


if __name__ == "__main__":
    main()
