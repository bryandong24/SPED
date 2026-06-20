"""Causal Forcing — LongLive KV-Recache vs plain hard-cut prompt swap.

Port of gino/scripts/swap_long.py to the Causal Forcing repo (thu-ml fork of
Self-Forcing; same Wan2.1 base, same WanDiffusionWrapper(is_causal, local_attn_size,
sink_size), byte-identical wan/modules/model.py).

Two modes, selected by --recache:
  * --recache 0  : plain HARD CUT — re-encode new prompt, flip crossattn is_init,
                   leave self-attn KV cache untouched. Transition is gated by how
                   fast the rolling window flushes old frames (slow, ~6-8s).
  * --recache N  : LongLive KV-RECACHE — rebuild the self-attn KV cache by replaying
                   the last N clean frames UNDER THE NEW PROMPT at the clean-context
                   timestep. Same pixels (motion preserved) but refreshed cached
                   semantics -> fast, smooth compliance with the new prompt.
  --grow_to / --post_window : grow-window schedule (stability after the swap).

Run from Causal-Forcing/ with PYTHONPATH=. and the Self-Forcing venv.
"""
import argparse, os
import numpy as np, torch, imageio
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from pipeline import CausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper

FRAME_SEQ = 1560
CKPT = "checkpoints/chunkwise/causal_forcing.pt"
CFG = "configs/causal_forcing_dmd_chunkwise.yaml"


def set_window(model, W):
    """Set the self-attn read window (latent frames) on every block, live."""
    for blk in model.blocks:
        blk.self_attn.local_attn_size = W
        blk.self_attn.max_attention_size = W * FRAME_SEQ


def recache(pipe, prefix, cond_emb, device, w):
    """LongLive KV-Recache: rebuild self-attn KV by re-encoding the last `w` clean
    frames of `prefix` under the new prompt at clean-context timestep, at RELATIVE
    positions 0..w-1. crossattn_cache must already be reset to the new prompt.
    Returns pos_offset = real index of the first replayed frame (subsequent gen uses
    current_start = (real_frame - pos_offset) * FRAME_SEQ)."""
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
                       current_start=(i - s0) * FRAME_SEQ)
        i += cur
    return s0


def build_pipeline(device, local_attn_size, sink_size):
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(CFG))
    gen = WanDiffusionWrapper(is_causal=True, local_attn_size=local_attn_size, sink_size=sink_size)
    gen.load_state_dict(torch.load(CKPT, map_location="cpu")["generator"])
    pipe = CausalInferencePipeline(cfg, device=device, generator=gen)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=40, help="latent frames (40 ~= 10s)")
    ap.add_argument("--swap_frame", type=int, default=12, help="latent frame of swap (12 = 3s)")
    ap.add_argument("--local_attn_size", type=int, default=21)
    ap.add_argument("--sink_size", type=int, default=3)
    ap.add_argument("--recache", type=int, default=0, help="replay N recent frames under new prompt (0=hard cut)")
    ap.add_argument("--post_window", type=int, default=0, help="shrink read window right after swap")
    ap.add_argument("--grow_to", type=int, default=0, help="grow window from post_window up to this, anchored at swap")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="../out/cf_recache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p1", required=True)
    ap.add_argument("--p2", required=True)
    args = ap.parse_args()
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = build_pipeline(device, args.local_attn_size, args.sink_size)
    nb = pipe.num_frame_per_block
    p1 = pipe.text_encoder(text_prompts=[args.p1])["prompt_embeds"]
    p2 = pipe.text_encoder(text_prompts=[args.p2])["prompt_embeds"]

    g = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn([1, args.total, 16, 60, 104], generator=g, dtype=torch.bfloat16).to(device)
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, noise.dtype, device)
    pipe._initialize_crossattn_cache(1, noise.dtype, device)
    out = torch.zeros_like(noise)

    cond = p1; start = 0; swapped_at = None; pos_offset = 0
    while start < args.total:
        cur = min(nb, args.total - start)
        if cond is p1 and start + cur > args.swap_frame:
            cond = p2
            for c in pipe.crossattn_cache: c["is_init"] = False
            swapped_at = start
            if args.recache > 0 and start > 0:
                pos_offset = recache(pipe, out[:, :start], p2, device, args.recache)
            if args.post_window > 0 and args.grow_to == 0:
                set_window(pipe.generator.model, args.post_window)
        if swapped_at is not None and args.grow_to > 0:
            base = args.post_window if args.post_window > 0 else nb
            W = min(args.grow_to, max(base, start - swapped_at))
            set_window(pipe.generator.model, W)
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
                       timestep=ctx_t, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                       current_start=(start - pos_offset) * FRAME_SEQ)
        start += cur

    video = (pipe.vae.decode_to_pixel(out, use_cache=True) * 0.5 + 0.5).clamp(0, 1)
    frames = (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()
    imageio.mimwrite(os.path.join(args.out_dir, f"{args.name}.mp4"), frames, fps=16, codec="libx264")
    np.save(os.path.join(args.out_dir, f"{args.name}_frames.npy"), frames[::8])  # subsample for grids
    w = frames.reshape(frames.shape[0], -1, 3).mean(axis=1); warmth = w[:, 0] - w[:, 2]
    swap_px = args.swap_frame * 4
    qs = [warmth[int(p*(len(warmth)-1))] for p in (0, .25, .5, .75, 1.0)]
    print(f"[{args.name}] {frames.shape[0]}f swap_px={swap_px} warmth@0/25/50/75/100% = "
          + "/".join(f"{x:.0f}" for x in qs))


if __name__ == "__main__":
    main()
