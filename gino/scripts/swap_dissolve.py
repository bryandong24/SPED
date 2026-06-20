"""Dual-branch latent DISSOLVE for a smooth swap moment.

At the swap, fork into two rollouts that share the pre-swap past:
  - Branch A: OLD prompt + the unchanged cache  -> smooth continuation of the
    current scene/motion.
  - Branch B: NEW prompt + recached cache       -> the target scene.
For N transition frames, both branches denoise the SAME noise; the output latent
= lerp(A, B, beta), beta 0->1. The blended latent's clean-context K/V are written
back into BOTH caches each frame, so the two branches stay synced to the actually-
shown video (no drift). After N frames, drop A and continue with B (new prompt).

This is a generative cross-dissolve localized at the swap -> the transition is
gradual instead of a hard cut. Costs 2x compute only during the N-frame window.
"""
import argparse, os, copy
import numpy as np, torch, imageio
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from pipeline import CausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper

FRAME_SEQ = 1560
SUN = "a dog running in a sunny green meadow full of wildflowers, bright daylight, cinematic, highly detailed"
NIGHT = "a dog running through a snowy field on a cold winter night, deep blue moonlight, falling snow, cinematic, highly detailed"


def clone_cache(cache):
    return [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in blk.items()} for blk in cache]


def denoise_chunk(pipe, noise_chunk, cond, kvc, xc, cs):
    """Run the few-step denoise for one chunk on the given caches; return clean latent."""
    cur = noise_chunk.shape[1]; noisy = noise_chunk
    for i, ts in enumerate(pipe.denoising_step_list):
        timestep = torch.ones([1, cur], device=noisy.device, dtype=torch.int64) * ts
        _, den = pipe.generator(noisy_image_or_video=noisy, conditional_dict={"prompt_embeds": cond},
                                timestep=timestep, kv_cache=kvc, crossattn_cache=xc, current_start=cs)
        if i < len(pipe.denoising_step_list) - 1:
            nt = pipe.denoising_step_list[i + 1]
            noisy = pipe.scheduler.add_noise(den.flatten(0, 1), torch.randn_like(den.flatten(0, 1)),
                                             nt * torch.ones([cur], device=noisy.device, dtype=torch.long)).unflatten(0, den.shape[:2])
    return den


def clean_ctx(pipe, den, cond, kvc, xc, cs):
    ctx_t = torch.ones([1, den.shape[1]], device=den.device, dtype=torch.int64) * pipe.args.context_noise
    pipe.generator(noisy_image_or_video=den, conditional_dict={"prompt_embeds": cond},
                   timestep=ctx_t, kv_cache=kvc, crossattn_cache=xc, current_start=cs)


def recache(pipe, prefix, cond_emb, kvc, xc, device, w):
    K = prefix.shape[1]; s0 = max(0, K - w)
    for c in xc: c["is_init"] = False
    nb = pipe.num_frame_per_block; i = s0
    while i < K:
        cur = min(nb, K - i)
        clean_ctx(pipe, prefix[:, i:i + cur], cond_emb, kvc, xc, (i - s0) * FRAME_SEQ)
        i += cur
    return s0


def new_caches(pipe, device):
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, torch.bfloat16, device)
    pipe._initialize_crossattn_cache(1, torch.bfloat16, device)
    return pipe.kv_cache1, pipe.crossattn_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dissolve", type=int, default=6, help="transition frames to cross-dissolve over")
    ap.add_argument("--recache", type=int, default=9)
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--swap_frame", type=int, default=30)
    ap.add_argument("--out_dir", default="../out/swap_dissolve")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p1", default=SUN); ap.add_argument("--p2", default=NIGHT)
    args = ap.parse_args()
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/self_forcing_dmd.yaml"))
    gen = WanDiffusionWrapper(is_causal=True, local_attn_size=21, sink_size=1)
    gen.load_state_dict(torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")["generator_ema"])
    pipe = CausalInferencePipeline(cfg, device=device, generator=gen)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
    nb = 3; pipe.num_frame_per_block = nb; pipe.generator.model.num_frame_per_block = nb

    sun = pipe.text_encoder(text_prompts=[args.p1])["prompt_embeds"]
    night = pipe.text_encoder(text_prompts=[args.p2])["prompt_embeds"]

    g = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn([1, args.total, 16, 60, 104], generator=g, dtype=torch.bfloat16).to(device)
    kvA, xA = new_caches(pipe, device)
    out = torch.zeros_like(noise)

    # Phase 1: pre-swap, single branch (old prompt)
    start = 0
    while start + nb <= args.swap_frame:
        den = denoise_chunk(pipe, noise[:, start:start + nb], sun, kvA, xA, start * FRAME_SEQ)
        out[:, start:start + nb] = den
        clean_ctx(pipe, den, sun, kvA, xA, start * FRAME_SEQ)
        start += nb
    swap_at = start

    # Fork: branch A = clone (continue old); branch B = recache new on a fresh cache
    kvA_b, xA_b = clone_cache(kvA), clone_cache(xA)
    kvB, xB = new_caches(pipe, device)
    posB = recache(pipe, out[:, :start], night, kvB, xB, device, args.recache)
    kvB_b, xB_b = clone_cache(pipe.kv_cache1), clone_cache(pipe.crossattn_cache)

    # Phase 2: dissolve over N frames (chunks)
    n_chunks = max(1, (args.dissolve + nb - 1) // nb)
    for c in range(n_chunks):
        cur = min(nb, args.total - start)
        if cur <= 0: break
        beta = min(1.0, (c + 1) / float(n_chunks + 1))
        nz = noise[:, start:start + cur]
        denA = denoise_chunk(pipe, nz, sun, kvA_b, xA_b, start * FRAME_SEQ)
        denB = denoise_chunk(pipe, nz, night, kvB_b, xB_b, (start - posB) * FRAME_SEQ)
        blend = (1 - beta) * denA + beta * denB
        out[:, start:start + cur] = blend
        # each branch keeps its OWN pure latent in its OWN cache (so B stays
        # committed to night); only the OUTPUT is dissolved. At beta->1 the blend
        # converges to denB, matching B's continuation -> smooth handoff.
        clean_ctx(pipe, denA, sun, kvA_b, xA_b, start * FRAME_SEQ)
        clean_ctx(pipe, denB, night, kvB_b, xB_b, (start - posB) * FRAME_SEQ)
        start += cur

    # Phase 3: continue with branch B (new prompt) only
    kvB2, xB2 = kvB_b, xB_b
    while start < args.total:
        cur = min(nb, args.total - start)
        den = denoise_chunk(pipe, noise[:, start:start + cur], night, kvB2, xB2, (start - posB) * FRAME_SEQ)
        out[:, start:start + cur] = den
        clean_ctx(pipe, den, night, kvB2, xB2, (start - posB) * FRAME_SEQ)
        start += cur

    video = (pipe.vae.decode_to_pixel(out, use_cache=True) * 0.5 + 0.5).clamp(0, 1)
    frames = (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()
    imageio.mimwrite(os.path.join(args.out_dir, f"{args.name}.mp4"), frames, fps=16, codec="libx264")
    idxs = np.linspace(0, frames.shape[0] - 1, 16).round().astype(int)
    sp = swap_at * 4
    tiles = [frames[i].copy() for i in idxs]
    for t, i in zip(tiles, idxs):
        if i >= sp: t[:5, :] = [255, 0, 0]
    Image.fromarray(np.concatenate(tiles, axis=1)).save(os.path.join(args.out_dir, f"{args.name}_sheet.png"))
    w = frames.reshape(frames.shape[0], -1, 3).mean(1); warmth = w[:, 0] - w[:, 2]
    cut = np.abs(frames[sp].astype(np.int16) - frames[sp - 1].astype(np.int16)).mean() if sp > 0 else 0
    # max single-frame jump in the transition window (the 'abruptness')
    win = frames[max(0, sp - 4):sp + args.dissolve * 4 + 8].astype(np.int16)
    maxjump = np.abs(np.diff(win, axis=0)).mean(axis=(1, 2, 3)).max() if len(win) > 1 else 0
    qs = [warmth[int(p * (len(warmth) - 1))] for p in (0, .25, .5, .75, 1.0)]
    print(f"[{args.name}] dissolve={args.dissolve} warmth=" + "/".join(f"{x:.0f}" for x in qs) +
          f" | cut@swap={cut:.1f} | max-jump-in-window={maxjump:.1f}")


if __name__ == "__main__":
    main()
