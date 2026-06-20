"""Waypoint prompt-stepping for a smooth swap.

Instead of interpolating the cache/output (still cuts), step the PROMPT through
intermediate vectors with ACTUAL rollout + recache at each waypoint:

  betas = e.g. [0.5, 1.0]  (the avg of old/new, then the new prompt)
  for each beta:
     cond = mix(P_old, P_new, beta)      # intermediate prompt vector
     recache the generated prefix UNDER cond   # cache consistent w/ waypoint
     roll out `hold` chunks under cond    # generate real intermediate-semantic frames

So the video passes through genuinely-generated intermediate states (e.g. a dusk
scene) rather than a blend -> gradual, in-distribution transition. 15s default.
"""
import argparse, os
import numpy as np, torch, imageio
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from pipeline import CausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper

FRAME_SEQ = 1560
SUN = "a dog running in a sunny green meadow full of wildflowers, bright daylight, cinematic, highly detailed"
NIGHT = "a dog running through a snowy field on a cold winter night, deep blue moonlight, falling snow, cinematic, highly detailed"


def set_window(model, W):
    for blk in model.blocks:
        blk.self_attn.local_attn_size = W
        blk.self_attn.max_attention_size = W * FRAME_SEQ


def mix(a, b, t, slerp):
    if not slerp:
        return (1 - t) * a + t * b
    an = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    bn = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    dot = (an * bn).sum(-1, keepdim=True).clamp(-1, 1)
    om = torch.acos(dot); so = torch.sin(om).clamp_min(1e-6)
    w = (torch.sin((1 - t) * om) / so) * a + (torch.sin(t * om) / so) * b
    return torch.where(so < 1e-4, (1 - t) * a + t * b, w)


def recache(pipe, prefix, cond, device, w):
    K = prefix.shape[1]; s0 = max(0, K - w)
    pipe.kv_cache1 = None; pipe._initialize_kv_cache(1, prefix.dtype, device)
    for c in pipe.crossattn_cache: c["is_init"] = False
    nb = pipe.num_frame_per_block; i = s0
    while i < K:
        cur = min(nb, K - i)
        ctx_t = torch.ones([1, cur], device=device, dtype=torch.int64) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=prefix[:, i:i + cur], conditional_dict={"prompt_embeds": cond},
                       timestep=ctx_t, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                       current_start=(i - s0) * FRAME_SEQ)
        i += cur
    return s0


def gen_chunk(pipe, noise_chunk, cond, pos_off, start):
    cur = noise_chunk.shape[1]; noisy = noise_chunk
    for i, ts in enumerate(pipe.denoising_step_list):
        timestep = torch.ones([1, cur], device=noisy.device, dtype=torch.int64) * ts
        _, den = pipe.generator(noisy_image_or_video=noisy, conditional_dict={"prompt_embeds": cond},
                                timestep=timestep, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                                current_start=(start - pos_off) * FRAME_SEQ)
        if i < len(pipe.denoising_step_list) - 1:
            nt = pipe.denoising_step_list[i + 1]
            noisy = pipe.scheduler.add_noise(den.flatten(0, 1), torch.randn_like(den.flatten(0, 1)),
                                             nt * torch.ones([cur], device=noisy.device, dtype=torch.long)).unflatten(0, den.shape[:2])
    ctx_t = torch.ones([1, cur], device=noisy.device, dtype=torch.int64) * pipe.args.context_noise
    pipe.generator(noisy_image_or_video=den, conditional_dict={"prompt_embeds": cond},
                   timestep=ctx_t, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                   current_start=(start - pos_off) * FRAME_SEQ)
    return den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--betas", default="0.5,1.0", help="comma list of mix coeffs old->new to step through (end at 1.0)")
    ap.add_argument("--hold", type=int, default=2, help="chunks (x3 frames) generated per waypoint")
    ap.add_argument("--slerp", action="store_true")
    ap.add_argument("--recache", type=int, default=9)
    ap.add_argument("--window", type=int, default=9, help="self-attn window during/after transition")
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--swap_frame", type=int, default=30)
    ap.add_argument("--out_dir", default="../out/swap_wp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p1", default=SUN); ap.add_argument("--p2", default=NIGHT)
    ap.add_argument("--prompt_chain", default="", help="'|'-separated REAL intermediate prompts to step through (semantic waypoints); overrides --betas")
    args = ap.parse_args()
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)
    betas = [float(x) for x in args.betas.split(",")]
    chain = [s for s in args.prompt_chain.split("|") if s.strip()] if args.prompt_chain else None

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
    pipe.kv_cache1 = None; pipe._initialize_kv_cache(1, noise.dtype, device)
    pipe._initialize_crossattn_cache(1, noise.dtype, device)
    out = torch.zeros_like(noise)

    # Phase 1: pre-swap (old prompt)
    start = 0; pos_off = 0
    while start + nb <= args.swap_frame:
        out[:, start:start + nb] = gen_chunk(pipe, noise[:, start:start + nb], sun, pos_off, start)
        start += nb

    # Phase 2: waypoints — either embedding-mix betas, or REAL intermediate prompts
    waypoint_conds = ([pipe.text_encoder(text_prompts=[p])["prompt_embeds"] for p in chain]
                      if chain else [mix(sun, night, b, args.slerp) for b in betas])
    if chain:
        night = waypoint_conds[-1]  # last chain prompt is the final target
    for cond in waypoint_conds:
        pos_off = recache(pipe, out[:, :start], cond, device, args.recache)
        set_window(pipe.generator.model, args.window)
        for _ in range(args.hold):
            if start >= args.total: break
            cur = min(nb, args.total - start)
            out[:, start:start + cur] = gen_chunk(pipe, noise[:, start:start + cur], cond, pos_off, start)
            start += cur

    # Phase 3: continue under new prompt
    while start < args.total:
        cur = min(nb, args.total - start)
        out[:, start:start + cur] = gen_chunk(pipe, noise[:, start:start + cur], night, pos_off, start)
        start += cur

    video = (pipe.vae.decode_to_pixel(out, use_cache=True) * 0.5 + 0.5).clamp(0, 1)
    frames = (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()
    imageio.mimwrite(os.path.join(args.out_dir, f"{args.name}.mp4"), frames, fps=16, codec="libx264")
    idxs = np.linspace(0, frames.shape[0] - 1, 16).round().astype(int)
    sp = args.swap_frame * 4
    tiles = [frames[i].copy() for i in idxs]
    for t, i in zip(tiles, idxs):
        if i >= sp: t[:5, :] = [255, 0, 0]
    Image.fromarray(np.concatenate(tiles, axis=1)).save(os.path.join(args.out_dir, f"{args.name}_sheet.png"))
    w = frames.reshape(frames.shape[0], -1, 3).mean(1); warmth = w[:, 0] - w[:, 2]
    djump = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
    sp_lo, sp_hi = sp, min(len(djump), sp + len(betas) * args.hold * 4 + 4)
    qs = [warmth[int(p * (len(warmth) - 1))] for p in (0, .25, .5, .75, 1.0)]
    print(f"[{args.name}] betas={args.betas} hold={args.hold} warmth=" + "/".join(f"{x:.0f}" for x in qs) +
          f" | maxjump-in-transition={djump[sp_lo:sp_hi].max():.1f}")


if __name__ == "__main__":
    main()
