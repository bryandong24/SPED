"""Prompt-vector swap experiments v2 — break the 'resist' failure mode.

v1 finding: hardcut/crossfade/soft-evict all RESIST; the scene stays a sunny
meadow (self-attn KV old-world momentum dominates). Only the dog breed morphed.

v2 levers (run one --variant per GPU to parallelize):
  - cfg<scale>      : classifier-free guidance to push the new prompt harder.
                      Separate cond/uncond cross-attn caches; shared visual KV.
  - wipe            : at swap, ZERO the self-attn KV data + reset indices (truly
                      clear old-world history, unlike v1's index-only evict).
  - resetpos        : with wipe, also restart RoPE frame counter at 0 -> the
                      post-swap chunks are a fresh clip under the new prompt.
  - early           : swap at chunk 1 instead of 3 (less history to fight).

Metric: warmth = mean(R) - mean(B) per frame. Sunny meadow = warm (+), winter
night = cool (-). Snow-at-night is BRIGHT so luminance is useless here.
"""
import argparse, os, time
import numpy as np
import torch
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
import imageio
from pipeline import CausalInferencePipeline

FRAME_SEQ = 1560
SUN = "a dog running in a sunny green meadow full of wildflowers, bright daylight, cinematic, highly detailed"
NIGHT = "a dog running through a snowy field on a cold winter night, deep blue moonlight, falling snow, cinematic, highly detailed"

VARIANTS = {
    "instant":        dict(swap_at=0, cfg=1.0, wipe=False, resetpos=False),
    "hardcut":        dict(swap_at=3, cfg=1.0, wipe=False, resetpos=False),
    "cfg6":           dict(swap_at=3, cfg=6.0, wipe=False, resetpos=False),
    "cfg10":          dict(swap_at=3, cfg=10.0, wipe=False, resetpos=False),
    "wipe":           dict(swap_at=3, cfg=1.0, wipe=True,  resetpos=False),
    "wipe_resetpos":  dict(swap_at=3, cfg=1.0, wipe=True,  resetpos=True),
    "cfg6_wipe":      dict(swap_at=3, cfg=6.0, wipe=True,  resetpos=False),
    "early_hardcut":  dict(swap_at=1, cfg=1.0, wipe=False, resetpos=False),
    "early_cfg6":     dict(swap_at=1, cfg=6.0, wipe=False, resetpos=False),
}


def build(config_path, ckpt, device):
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(config_path))
    pipe = CausalInferencePipeline(cfg, device=device)
    pipe.generator.load_state_dict(torch.load(ckpt, map_location="cpu")["generator_ema"])
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
    return pipe


def new_xattn_cache(pipe, device):
    cache = []
    for _ in range(pipe.num_transformer_blocks):
        cache.append({"k": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
                      "v": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
                      "is_init": False})
    return cache


def reset_init(cache):
    for c in cache:
        c["is_init"] = False


def wipe_kv(pipe):
    for kv in pipe.kv_cache1:
        kv["k"].zero_(); kv["v"].zero_()
        kv["global_end_index"].zero_(); kv["local_end_index"].zero_()


@torch.no_grad()
def rollout(pipe, sun_emb, night_emb, neg_emb, *, swap_at, cfg, wipe, resetpos,
            num_blocks, seed, device):
    cur = pipe.num_frame_per_block
    g = torch.Generator("cpu").manual_seed(seed)
    noise = torch.randn([1, num_blocks * cur, 16, 60, 104], generator=g, dtype=torch.bfloat16).to(device)
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, noise.dtype, device)
    cond_cache = new_xattn_cache(pipe, device)
    uncond_cache = new_xattn_cache(pipe, device)
    out = torch.zeros([1, num_blocks * cur, 16, 60, 104], device=device, dtype=noise.dtype)

    cond_emb = sun_emb
    pos_off = 0          # RoPE frame offset (for resetpos)
    start = 0; swap_frame = None
    for blk in range(num_blocks):
        if blk == swap_at and night_emb is not None:
            cond_emb = night_emb
            reset_init(cond_cache)
            if wipe:
                wipe_kv(pipe)
            if resetpos:
                pos_off = start          # post-swap chunks count RoPE from 0
            swap_frame = start

        cs = (start - pos_off) * FRAME_SEQ
        noisy = noise[:, start:start + cur]
        for i, ts in enumerate(pipe.denoising_step_list):
            timestep = torch.ones([1, cur], device=device, dtype=torch.int64) * ts
            _, x0c = pipe.generator(noisy_image_or_video=noisy, conditional_dict={"prompt_embeds": cond_emb},
                                    timestep=timestep, kv_cache=pipe.kv_cache1,
                                    crossattn_cache=cond_cache, current_start=cs)
            if cfg > 1.0:
                _, x0u = pipe.generator(noisy_image_or_video=noisy, conditional_dict={"prompt_embeds": neg_emb},
                                        timestep=timestep, kv_cache=pipe.kv_cache1,
                                        crossattn_cache=uncond_cache, current_start=cs)
                x0 = x0u + cfg * (x0c - x0u)
            else:
                x0 = x0c
            denoised = x0
            if i < len(pipe.denoising_step_list) - 1:
                nt = pipe.denoising_step_list[i + 1]
                noisy = pipe.scheduler.add_noise(
                    denoised.flatten(0, 1), torch.randn_like(denoised.flatten(0, 1)),
                    nt * torch.ones([cur], device=device, dtype=torch.long)).unflatten(0, denoised.shape[:2])
        out[:, start:start + cur] = denoised
        # clean-context pass (cond only) -> writes clean K/V for future chunks
        ctx_t = torch.ones_like(timestep) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=denoised, conditional_dict={"prompt_embeds": cond_emb},
                       timestep=ctx_t, kv_cache=pipe.kv_cache1, crossattn_cache=cond_cache, current_start=cs)
        start += cur

    video = pipe.vae.decode_to_pixel(out, use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    frames = (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()
    return frames, (None if swap_frame is None else swap_frame * 4)


def sheet(frames, path, swap_px, cols=9):
    idxs = np.linspace(0, frames.shape[0] - 1, cols).round().astype(int)
    tiles = []
    for i in idxs:
        f = frames[i].copy()
        if swap_px is not None and i >= swap_px:
            f[:5, :] = [255, 0, 0]
        tiles.append(f)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS) + ["ref_sun", "ref_night"])
    ap.add_argument("--out_dir", default="../out/swap2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    args = ap.parse_args()
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = build(args.config_path, args.checkpoint_path, device)
    sun = pipe.text_encoder(text_prompts=[SUN])["prompt_embeds"]
    night = pipe.text_encoder(text_prompts=[NIGHT])["prompt_embeds"]
    neg = pipe.text_encoder(text_prompts=[""])["prompt_embeds"]

    if args.variant == "ref_sun":
        frames, swp = rollout(pipe, sun, None, neg, swap_at=99, cfg=1.0, wipe=False, resetpos=False, num_blocks=7, seed=args.seed, device=device)
    elif args.variant == "ref_night":
        frames, swp = rollout(pipe, night, None, neg, swap_at=99, cfg=1.0, wipe=False, resetpos=False, num_blocks=7, seed=args.seed, device=device)
    else:
        cfgd = VARIANTS[args.variant]
        t0 = time.time()
        frames, swp = rollout(pipe, sun, night, neg, num_blocks=7, seed=args.seed, device=device, **cfgd)
        print(f"[{args.variant}] gen {time.time()-t0:.1f}s")

    imageio.mimwrite(os.path.join(args.out_dir, f"{args.variant}.mp4"), frames, fps=16, codec="libx264")
    sheet(frames, os.path.join(args.out_dir, f"{args.variant}_sheet.png"), swp)
    warmth = frames.reshape(frames.shape[0], -1, 3).mean(axis=1)
    warmth = warmth[:, 0] - warmth[:, 2]  # R - B
    pre = warmth[:swp].mean() if swp else warmth.mean()
    post = warmth[swp:].mean() if swp else warmth.mean()
    print(f"[{args.variant}] swap_px={swp} warmth pre={pre:.1f} post={post:.1f} "
          f"delta={post-pre:.1f} | first/mid/last warmth {warmth[0]:.0f}/{warmth[len(warmth)//2]:.0f}/{warmth[-1]:.0f}")


if __name__ == "__main__":
    main()
