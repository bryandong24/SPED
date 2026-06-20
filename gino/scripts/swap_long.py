"""Long-horizon swap: 20s video, swap prompt at 2.5s, watch the morph develop.

Uses the rolling KV window (local_attn_size=21) so it can exceed 21 frames.
Configurable block size (1-frame for finest 'swap after 1f' granularity, or 3),
and optional post-swap text-scale boost. Streaming cached VAE decode (use_cache)
keeps the 80-frame decode memory-light and seam-free.
"""
import argparse, os, types
import numpy as np, torch, imageio
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from pipeline import CausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper

FRAME_SEQ = 1560
SUN = "a dog running in a sunny green meadow full of wildflowers, bright daylight, cinematic, highly detailed"
NIGHT = "a dog running through a snowy field on a cold winter night, deep blue moonlight, falling snow, cinematic, highly detailed"


class S:
    text = 1.0; frame = 1.0


def set_window(model, W):
    """Set the self-attn read window (in latent frames) on every block live.
    Shrinking it after a swap flushes old-prompt frames from attention faster."""
    for blk in model.blocks:
        blk.self_attn.local_attn_size = W
        blk.self_attn.max_attention_size = W * FRAME_SEQ


def recache(pipe, prefix, cond_emb, device, w):
    """LongLive KV-Recache (training-free): rebuild the self-attn KV cache by
    re-encoding the already-generated clean frames `prefix` [1,K,16,60,104]
    UNDER THE NEW PROMPT. Preserves visual/motion state (same frames) while
    refreshing cached semantics, so future frames continue smoothly but comply
    with the new prompt. Replays the most recent `w` frames at clean-context
    timestep. crossattn_cache must already be reset to the new prompt.

    Frames are replayed at RELATIVE positions 0..W-1 (not absolute), so the
    cache never overflows for late swaps and RoPE stays in-distribution. Returns
    `pos_offset` = real index of the first replayed frame; subsequent generation
    must use current_start = (real_frame - pos_offset) * FRAME_SEQ."""
    K = prefix.shape[1]
    s0 = max(0, K - w)
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, prefix.dtype, device)
    nb = pipe.num_frame_per_block
    i = s0
    while i < K:
        cur = min(nb, K - i)
        chunk = prefix[:, i:i + cur]
        ctx_t = torch.ones([1, cur], device=device, dtype=torch.int64) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=chunk, conditional_dict={"prompt_embeds": cond_emb},
                       timestep=ctx_t, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                       current_start=(i - s0) * FRAME_SEQ)   # RELATIVE position
        i += cur
    return s0


def patch(model):
    for blk in model.blocks:
        ca, sa = blk.cross_attn, blk.self_attn
        ca._o = ca.forward; sa._o = sa.forward
        ca.forward = types.MethodType(lambda self, *a, **k: S.text * self._o(*a, **k), ca)
        sa.forward = types.MethodType(lambda self, *a, **k: S.frame * self._o(*a, **k), sa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=80, help="latent frames (80 ~= 20s)")
    ap.add_argument("--swap_frame", type=int, default=10, help="latent frame of swap (10 = 2.5s)")
    ap.add_argument("--nfpb", type=int, default=1, help="frames per block (1=finest)")
    ap.add_argument("--text_scale", type=float, default=1.0)
    ap.add_argument("--frame_scale", type=float, default=1.0)
    ap.add_argument("--local_attn_size", type=int, default=21)
    ap.add_argument("--sink_size", type=int, default=1)
    ap.add_argument("--post_window", type=int, default=0,
                    help="shrink read window to this many frames right after swap (0=no change)")
    ap.add_argument("--restore_after", type=int, default=10**9,
                    help="frames after swap to restore the window to local_attn_size")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="../out/swap_long")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p1", default=SUN, help="initial prompt")
    ap.add_argument("--p2", default=NIGHT, help="swap-to prompt")
    ap.add_argument("--ramp", type=int, default=0,
                    help="ramp window from local_attn_size down to post_window over this many frames")
    ap.add_argument("--recache", type=int, default=0,
                    help="LongLive KV-recache: replay this many recent frames under the new prompt at swap (0=off)")
    ap.add_argument("--grow_to", type=int, default=0,
                    help="grow the window from post_window up to this over post-swap frames (anchored at swap so it never re-includes pre-swap content)")
    args = ap.parse_args()
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/self_forcing_dmd.yaml"))
    gen = WanDiffusionWrapper(is_causal=True, local_attn_size=args.local_attn_size, sink_size=args.sink_size)
    gen.load_state_dict(torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")["generator_ema"])
    pipe = CausalInferencePipeline(cfg, device=device, generator=gen)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
    pipe.num_frame_per_block = args.nfpb
    pipe.generator.model.num_frame_per_block = args.nfpb
    patch(pipe.generator.model)

    sun = pipe.text_encoder(text_prompts=[args.p1])["prompt_embeds"]
    night = pipe.text_encoder(text_prompts=[args.p2])["prompt_embeds"]

    g = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn([1, args.total, 16, 60, 104], generator=g, dtype=torch.bfloat16).to(device)
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, noise.dtype, device)
    pipe._initialize_crossattn_cache(1, noise.dtype, device)
    out = torch.zeros_like(noise)

    nb = args.nfpb; cond = sun; start = 0; swapped_at = None; restored = False; pos_offset = 0
    while start < args.total:
        cur = min(nb, args.total - start)
        if cond is sun and start + cur > args.swap_frame:
            cond = night
            for c in pipe.crossattn_cache: c["is_init"] = False
            S.text, S.frame = args.text_scale, args.frame_scale
            swapped_at = start
            if args.recache > 0 and start > 0:
                # rebuild KV from the generated prefix under the new prompt
                pos_offset = recache(pipe, out[:, :start], night, device, args.recache)
            if args.post_window > 0 and args.ramp == 0:
                set_window(pipe.generator.model, args.post_window)
        if swapped_at is not None and args.ramp > 0 and args.post_window > 0:
            prog = (start - swapped_at) / float(args.ramp)
            W = round(args.local_attn_size + (args.post_window - args.local_attn_size) * min(1.0, prog))
            set_window(pipe.generator.model, max(args.post_window, W))
        if swapped_at is not None and args.grow_to > 0:
            # window grows with frames-since-swap (stability on new content) but
            # stays anchored so it never reaches back into pre-swap frames (no inertia)
            base = args.post_window if args.post_window > 0 else nb
            W = min(args.grow_to, max(base, start - swapped_at))
            set_window(pipe.generator.model, W)
        if swapped_at is not None and not restored and start - swapped_at >= args.restore_after:
            set_window(pipe.generator.model, args.local_attn_size)
            restored = True
        noisy = noise[:, start:start + cur]
        for i, ts in enumerate(pipe.denoising_step_list):
            timestep = torch.ones([1, cur], device=device, dtype=torch.int64) * ts
            _, den = pipe.generator(noisy_image_or_video=noisy, conditional_dict={"prompt_embeds": cond},
                                    timestep=timestep, kv_cache=pipe.kv_cache1,
                                    crossattn_cache=pipe.crossattn_cache, current_start=(start - pos_offset) * FRAME_SEQ)
            if i < len(pipe.denoising_step_list) - 1:
                nt = pipe.denoising_step_list[i + 1]
                noisy = pipe.scheduler.add_noise(den.flatten(0, 1), torch.randn_like(den.flatten(0, 1)),
                                                 nt * torch.ones([cur], device=device, dtype=torch.long)).unflatten(0, den.shape[:2])
        out[:, start:start + cur] = den
        ctx_t = torch.ones_like(timestep) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=den, conditional_dict={"prompt_embeds": cond},
                       timestep=ctx_t, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache, current_start=(start - pos_offset) * FRAME_SEQ)
        start += cur

    video = (pipe.vae.decode_to_pixel(out, use_cache=True) * 0.5 + 0.5).clamp(0, 1)
    frames = (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()
    imageio.mimwrite(os.path.join(args.out_dir, f"{args.name}.mp4"), frames, fps=16, codec="libx264")
    # dense contact sheet (more tiles for a long clip)
    idxs = np.linspace(0, frames.shape[0] - 1, 16).round().astype(int)
    swap_px = args.swap_frame * 4
    tiles = []
    for i in idxs:
        f = frames[i].copy()
        if i >= swap_px: f[:5, :] = [255, 0, 0]
        tiles.append(f)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(os.path.join(args.out_dir, f"{args.name}_sheet.png"))
    w = frames.reshape(frames.shape[0], -1, 3).mean(axis=1); warmth = w[:, 0] - w[:, 2]
    qs = [warmth[int(p*(len(warmth)-1))] for p in (0, .2, .4, .6, .8, 1.0)]
    print(f"[{args.name}] {frames.shape[0]} px frames | swap_px={swap_px} | "
          f"warmth @0/20/40/60/80/100% = " + "/".join(f"{x:.0f}" for x in qs))


if __name__ == "__main__":
    main()
