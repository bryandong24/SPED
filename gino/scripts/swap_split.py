"""Split-encode recache: recache FEWER frames under the new prompt, and keep the
rest of the window as the ORIGINAL (un-re-encoded) old-prompt frames.

Root cause of the double-cut (see experiments/12): plain recache rebuilds the
whole window from old-structure frames re-encoded under the new prompt, which then
get evicted as a block -> a second cut at swap+window.

Fix idea (user): rebuild the window as a MIX:
  [ W-M frames encoded under OLD prompt | M most-recent frames encoded under NEW ]
so the window attends to both un-recached (old) and recached (new) frames. As the
native-new frames accumulate, the old frames evict ONE AT A TIME -> gradual roll,
no abrupt bridge-drop.

--recache W      total frames in the rebuilt window
--recache_new M  how many of the last frames to encode under the NEW prompt
                 (the earlier W-M stay OLD). M=W -> plain recache; M=0 -> no swap.
"""
import argparse, os
import numpy as np, torch, imageio
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from pipeline import CausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper

FRAME_SEQ = 1560
SUN = "A fluffy golden retriever joyfully sprinting through a vast sunlit meadow of vivid orange and yellow wildflowers, warm golden afternoon sunlight, lush green grass, distant rolling hills under a clear blue sky, cinematic, photorealistic, highly detailed, 4k"
NIGHT = "A fluffy golden retriever running across a deep snow-covered field on a frigid winter night, cold silver moonlight from a large full moon, gently falling snow, frosted pine trees, deep blue and indigo tones, cinematic, photorealistic, highly detailed, 4k"


def set_window(model, W):
    for blk in model.blocks:
        blk.self_attn.local_attn_size = W
        blk.self_attn.max_attention_size = W * FRAME_SEQ


def recache_split(pipe, prefix, sun_e, night_e, device, w, m):
    """Rebuild window: first (w-m) frames under OLD prompt, last m under NEW."""
    K = prefix.shape[1]; s0 = max(0, K - w)
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, prefix.dtype, device)
    nb = pipe.num_frame_per_block
    boundary = w - m   # relative index at/after which we use NEW prompt
    prev = None; i = s0
    while i < K:
        cur = min(nb, K - i)
        rel = i - s0
        cond = night_e if rel >= boundary else sun_e
        if cond is not prev:                       # prompt changed -> recompute text K/V
            for c in pipe.crossattn_cache: c["is_init"] = False
            prev = cond
        ctx_t = torch.ones([1, cur], device=device, dtype=torch.int64) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=prefix[:, i:i + cur], conditional_dict={"prompt_embeds": cond},
                       timestep=ctx_t, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                       current_start=(i - s0) * FRAME_SEQ)
        i += cur
    for c in pipe.crossattn_cache: c["is_init"] = False  # continued gen uses NEW prompt
    return s0


def gen_chunk(pipe, nz, cond, pos_off, start):
    cur = nz.shape[1]; noisy = nz
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
    ap.add_argument("--recache", type=int, default=9)
    ap.add_argument("--recache_new", type=int, default=3)
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--grow_to", type=int, default=0)
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--swap_frame", type=int, default=30)
    ap.add_argument("--out_dir", default="../out/swap_split")
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
    pipe.kv_cache1 = None; pipe._initialize_kv_cache(1, noise.dtype, device)
    pipe._initialize_crossattn_cache(1, noise.dtype, device)
    out = torch.zeros_like(noise)

    start = 0; pos_off = 0
    while start + nb <= args.swap_frame:
        out[:, start:start + nb] = gen_chunk(pipe, noise[:, start:start + nb], sun, 0, start)
        start += nb
    swap_at = start
    pos_off = recache_split(pipe, out[:, :start], sun, night, device, args.recache, args.recache_new)
    set_window(pipe.generator.model, args.window)

    while start < args.total:
        cur = min(nb, args.total - start)
        if args.grow_to > 0:
            set_window(pipe.generator.model, min(args.grow_to, max(args.window, start - swap_at + args.window)))
        out[:, start:start + cur] = gen_chunk(pipe, noise[:, start:start + cur], night, pos_off, start)
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
    d = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
    pk = sorted(np.argsort(d[sp - 4:sp + 40])[::-1][:3] + (sp - 4))
    w = frames.reshape(frames.shape[0], -1, 3).mean(1); warmth = w[:, 0] - w[:, 2]
    qs = [warmth[int(p * (len(warmth) - 1))] for p in (0, .25, .5, .75, 1.0)]
    print(f"[{args.name}] rc={args.recache} new={args.recache_new} win={args.window} "
          f"warmth=" + "/".join(f"{x:.0f}" for x in qs) +
          f" | cuts(px:Δ)=" + " ".join(f"{int(i)}:{d[i]:.0f}" for i in pk))


if __name__ == "__main__":
    main()
