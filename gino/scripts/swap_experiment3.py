"""Swap v3 — manually scale TEXT (cross-attn) vs PAST-FRAME (self-attn) conditioning.

Hypothesis (user): the swap resists because the text cross-attn signal is too weak
relative to the self-attn past-frame signal. Fix: directly scale each pathway in
the DiT block, applied POST-SWAP only:
  - text_scale  (alpha>1): multiply cross-attn output  -> louder text.
  - frame_scale (beta<1):  multiply self-attn output    -> quieter past-frame momentum.

Far more surgical than CFG (which amplifies the whole prediction and blew up).
We monkeypatch each block's cross_attn/self_attn .forward to multiply their output
by a shared, mutable scale (set to 1.0 pre-swap, to alpha/beta post-swap). The KV
cache write is unaffected (K/V are computed before the output projection), so
history integrity is preserved. Swap at chunk 3 (the hard 'resist' case).
"""
import argparse, os, types
import numpy as np, torch, imageio
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from pipeline import CausalInferencePipeline

FRAME_SEQ = 1560
SUN = "a dog running in a sunny green meadow full of wildflowers, bright daylight, cinematic, highly detailed"
NIGHT = "a dog running through a snowy field on a cold winter night, deep blue moonlight, falling snow, cinematic, highly detailed"

# name -> (text_scale alpha, frame_scale beta)
VARIANTS = {
    "text1p5":        (1.5, 1.0),
    "text2":          (2.0, 1.0),
    "text3":          (3.0, 1.0),
    "text4":          (4.0, 1.0),
    "frame0p6":       (1.0, 0.6),
    "frame0p4":       (1.0, 0.4),
    "text2_frame0p6": (2.0, 0.6),
    "text3_frame0p5": (3.0, 0.5),
}


class Scales:
    text = 1.0
    frame = 1.0


def patch_blocks(model, S):
    """Wrap each block's cross_attn/self_attn forward to apply S.text / S.frame."""
    for blk in model.blocks:
        ca, sa = blk.cross_attn, blk.self_attn
        ca._orig = ca.forward; sa._orig = sa.forward
        def ca_fwd(self, *a, **k): return S.text * self._orig(*a, **k)
        def sa_fwd(self, *a, **k): return S.frame * self._orig(*a, **k)
        ca.forward = types.MethodType(ca_fwd, ca)
        sa.forward = types.MethodType(sa_fwd, sa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--swap_at", type=int, default=3)
    ap.add_argument("--out_dir", default="../out/swap3")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)
    alpha, beta = VARIANTS[args.variant]

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/self_forcing_dmd.yaml"))
    pipe = CausalInferencePipeline(cfg, device=device)
    pipe.generator.load_state_dict(torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")["generator_ema"])
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)

    S = Scales()
    patch_blocks(pipe.generator.model, S)

    sun = pipe.text_encoder(text_prompts=[SUN])["prompt_embeds"]
    night = pipe.text_encoder(text_prompts=[NIGHT])["prompt_embeds"]

    cur = pipe.num_frame_per_block; num_blocks = 7
    g = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn([1, num_blocks * cur, 16, 60, 104], generator=g, dtype=torch.bfloat16).to(device)
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, noise.dtype, device)
    pipe._initialize_crossattn_cache(1, noise.dtype, device)
    out = torch.zeros_like(noise)

    cond = sun; start = 0; swap_px = None
    for blk in range(num_blocks):
        if blk == args.swap_at:
            cond = night
            for c in pipe.crossattn_cache: c["is_init"] = False
            S.text, S.frame = alpha, beta   # turn the knobs ON post-swap
            swap_px = start * 4
        noisy = noise[:, start:start + cur]
        for i, ts in enumerate(pipe.denoising_step_list):
            timestep = torch.ones([1, cur], device=device, dtype=torch.int64) * ts
            _, den = pipe.generator(noisy_image_or_video=noisy, conditional_dict={"prompt_embeds": cond},
                                    timestep=timestep, kv_cache=pipe.kv_cache1,
                                    crossattn_cache=pipe.crossattn_cache, current_start=start * FRAME_SEQ)
            if i < len(pipe.denoising_step_list) - 1:
                nt = pipe.denoising_step_list[i + 1]
                noisy = pipe.scheduler.add_noise(den.flatten(0, 1), torch.randn_like(den.flatten(0, 1)),
                                                 nt * torch.ones([cur], device=device, dtype=torch.long)).unflatten(0, den.shape[:2])
        out[:, start:start + cur] = den
        ctx_t = torch.ones_like(timestep) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=den, conditional_dict={"prompt_embeds": cond},
                       timestep=ctx_t, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache, current_start=start * FRAME_SEQ)
        start += cur

    video = (pipe.vae.decode_to_pixel(out, use_cache=False) * 0.5 + 0.5).clamp(0, 1)
    frames = (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()
    imageio.mimwrite(os.path.join(args.out_dir, f"{args.variant}.mp4"), frames, fps=16, codec="libx264")
    idxs = np.linspace(0, frames.shape[0] - 1, 9).round().astype(int)
    tiles = [frames[i].copy() for i in idxs]
    for t, i in zip(tiles, idxs):
        if swap_px is not None and i >= swap_px: t[:5, :] = [255, 0, 0]
    Image.fromarray(np.concatenate(tiles, axis=1)).save(os.path.join(args.out_dir, f"{args.variant}_sheet.png"))
    w = frames.reshape(frames.shape[0], -1, 3).mean(axis=1); warmth = w[:, 0] - w[:, 2]
    pre = warmth[:swap_px].mean(); post = warmth[swap_px:].mean()
    print(f"[{args.variant}] a={alpha} b={beta} warmth pre={pre:.0f} post={post:.0f} "
          f"first/mid/last {warmth[0]:.0f}/{warmth[len(warmth)//2]:.0f}/{warmth[-1]:.0f}")


if __name__ == "__main__":
    main()
