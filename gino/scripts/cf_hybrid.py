"""Causal Forcing — hybrid hard-cut x KV-recache sweeps.

Loads the model ONCE and runs a whole group of variants for one example, so we can
shard (example x group) across GPUs.

Idea 1 (group=delay): apply the HARD CUT at the swap, let it morph for D frames,
THEN apply the recache over the last N frames. NOT equivalent to a plain hard cut:
hard cut only swaps the cross-attn prompt and never rebuilds the self-attn KV cache;
delayed recache rebuilds that KV from the already-morphing frames. The window
schedule (post_window->grow_to) is anchored at the RECACHE frame, so during the D
hard-cut frames the full window is kept (natural gradual onset), then recache locks
it in. D=0 would be the immediate recache.

Idea 2 (group=sweep): immediate recache, sweep N = how many recent frames are
re-encoded under the new prompt (the recache lookback).

Run from Causal-Forcing/ with PYTHONPATH=. and the Self-Forcing venv.
"""
import argparse, os, time
import numpy as np, torch, imageio
from omegaconf import OmegaConf
from einops import rearrange
from pipeline import CausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper

FRAME_SEQ = 1560
CKPT = "checkpoints/chunkwise/causal_forcing.pt"
CFG = "configs/causal_forcing_dmd_chunkwise.yaml"

PROMPTS = {
    "dog": ("A fluffy golden retriever sprinting through a sunlit meadow of orange wildflowers, warm golden daylight, lush green grass, cinematic, photorealistic, 4k",
            "A fluffy golden retriever running across a deep snowy field on a frigid winter night, full moon, falling snow, frosted pine trees, deep blue moonlight, cinematic, photorealistic, 4k"),
    "car": ("A red sports car driving along a sunny coastal highway by the blue ocean, bright clear daylight, palm trees, cinematic aerial view, photorealistic, 4k",
            "A red sports car driving down a rain-soaked neon city street at night, vibrant pink and blue neon reflections on wet asphalt, towering skyscrapers, cinematic, photorealistic, 4k"),
    "jungle": ("A drone shot flying low over a lush green tropical jungle canopy, misty waterfalls, bright daylight, vivid green foliage, cinematic, photorealistic, 4k",
               "A drone shot flying low over vast red sand desert dunes at golden hour, rippling sand, long shadows, arid wasteland, cinematic, photorealistic, 4k"),
}


def set_window(model, W):
    for blk in model.blocks:
        blk.self_attn.local_attn_size = W
        blk.self_attn.max_attention_size = W * FRAME_SEQ


def do_recache(pipe, prefix, cond_emb, device, w):
    """Rebuild self-attn KV from last `w` clean frames under new prompt; relative pos."""
    K = prefix.shape[1]
    s0 = max(0, K - w)
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, prefix.dtype, device)
    nb = pipe.num_frame_per_block
    i = s0
    while i < K:
        cur = min(nb, K - i)
        ctx_t = torch.ones([1, cur], device=device, dtype=torch.int64) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=prefix[:, i:i + cur], conditional_dict={"prompt_embeds": cond_emb},
                       timestep=ctx_t, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                       current_start=(i - s0) * FRAME_SEQ)
        i += cur
    return s0


@torch.no_grad()
def generate(pipe, noise, p1, p2, device, *, swap_frame, recache_n, recache_delay,
             window, sink, post_window, grow_to):
    """One full clip. recache_delay=0 -> immediate recache; >0 -> hard-cut then recache."""
    set_window(pipe.generator.model, window)
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, noise.dtype, device)
    pipe._initialize_crossattn_cache(1, noise.dtype, device)
    out = torch.zeros_like(noise)
    total = noise.shape[1]; nb = pipe.num_frame_per_block
    cond = p1; start = 0; swapped_at = None; anchor = None; pos_offset = 0; recached = False

    while start < total:
        cur = min(nb, total - start)
        if cond is p1 and start + cur > swap_frame:
            cond = p2
            for c in pipe.crossattn_cache: c["is_init"] = False
            swapped_at = start
            if recache_delay == 0 and start > 0:
                pos_offset = do_recache(pipe, out[:, :start], p2, device, recache_n)
                anchor = start; recached = True
        # delayed recache: fire once after `recache_delay` frames of hard cut
        if swapped_at is not None and not recached and recache_delay > 0 and start - swapped_at >= recache_delay and start > 0:
            pos_offset = do_recache(pipe, out[:, :start], p2, device, recache_n)
            anchor = start; recached = True
        # grow-window schedule, anchored at the recache frame
        if anchor is not None and grow_to > 0:
            W = min(grow_to, max(post_window, start - anchor))
            set_window(pipe.generator.model, W)
        noisy = noise[:, start:start + cur]
        for i, ts in enumerate(pipe.denoising_step_list):
            timestep = torch.ones([1, cur], device=device, dtype=torch.int64) * ts
            _, den = pipe.generator(noisy_image_or_video=noisy, conditional_dict={"prompt_embeds": cond},
                                    timestep=timestep, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                                    current_start=(start - pos_offset) * FRAME_SEQ)
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
    try:
        pipe.vae.model.clear_cache()
    except Exception:
        pass
    # use_cache=False so each clip decodes independently (no cross-job VAE state bleed)
    video = (pipe.vae.decode_to_pixel(out, use_cache=False) * 0.5 + 0.5).clamp(0, 1)
    return (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", required=True, choices=list(PROMPTS))
    ap.add_argument("--group", required=True, choices=["delay", "sweep"])
    ap.add_argument("--out_dir", default="../out/cf_hybrid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(CFG))
    gen = WanDiffusionWrapper(is_causal=True, local_attn_size=21, sink_size=3)
    gen.load_state_dict(torch.load(CKPT, map_location="cpu")["generator"])
    pipe = CausalInferencePipeline(cfg, device=device, generator=gen).to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)

    p1txt, p2txt = PROMPTS[args.example]
    p1 = pipe.text_encoder(text_prompts=[p1txt])["prompt_embeds"]
    p2 = pipe.text_encoder(text_prompts=[p2txt])["prompt_embeds"]
    g = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn([1, 40, 16, 60, 104], generator=g, dtype=torch.bfloat16).to(device)

    if args.group == "delay":   # idea 1: hard-cut for D frames, then recache N=9
        jobs = [("delay%d" % d, dict(recache_n=9, recache_delay=d)) for d in (2, 4, 6)]
    else:                        # idea 2: immediate recache, sweep lookback N
        jobs = [("rc%d" % n, dict(recache_n=n, recache_delay=0)) for n in (3, 6, 9, 15, 21)]

    for name, kw in jobs:
        t0 = time.time()
        frames = generate(pipe, noise, p1, p2, device, swap_frame=12, window=21, sink=3,
                          post_window=3, grow_to=15, **kw)
        tag = f"{args.example}_{name}"
        imageio.mimwrite(os.path.join(args.out_dir, f"{tag}.mp4"), frames, fps=16, codec="libx264")
        w = frames.reshape(frames.shape[0], -1, 3).mean(axis=1); warm = w[:, 0] - w[:, 2]
        qs = [warm[int(p * (len(warm) - 1))] for p in (0, .25, .5, .75, 1.0)]
        print(f"[{tag}] {time.time()-t0:.1f}s warmth@0/25/50/75/100 = " + "/".join(f"{x:.0f}" for x in qs), flush=True)
    print(f"GROUP DONE {args.example} {args.group}", flush=True)


if __name__ == "__main__":
    main()
