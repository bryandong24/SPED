"""Causal Forcing — mid-rollout TEXT-SWAP proof-of-concept.

Port of gino/scripts/text_swap_poc.py to the Causal Forcing repo (thu-ml).
CF is a Self-Forcing fork: wan/modules/model.py is IDENTICAL, so the text-swap
hook is the same — re-encode the new prompt and flip every block's
crossattn_cache["is_init"]=False so the text K/V is recomputed, while the
self-attn KV cache of past frames (kv_cache1) is left untouched.

Core question this answers: does swapping the prompt midway actually change the
video, or does the KV-cache momentum make CF resist (like Self-Forcing did)?

Run from the Causal-Forcing dir:
  cd Causal-Forcing && PYTHONPATH=. ../Self-Forcing/.venv/bin/python \
      ../scripts/cf_swap_poc.py \
      --config_path configs/causal_forcing_dmd_chunkwise.yaml \
      --checkpoint_path checkpoints/chunkwise/causal_forcing.pt \
      --prompt  "<base>" --prompt2 "<new>" --swap_at K \
      --num_output_frames 21 --out ../out/cf_phase1/swap.mp4
"""
import argparse
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from einops import rearrange
from torchvision.io import write_video
from PIL import Image

from pipeline import CausalInferencePipeline


def build_pipeline(config_path, checkpoint_path, use_ema, device):
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load(config_path))
    pipeline = CausalInferencePipeline(config, device=device)
    if checkpoint_path:
        sd = torch.load(checkpoint_path, map_location="cpu")
        gen_sd = sd["generator_ema" if use_ema else "generator"]
        try:
            pipeline.generator.load_state_dict(gen_sd)
        except RuntimeError:
            fixed = {(k.replace("model._fsdp_wrapped_module.", "model.", 1)
                      if k.startswith("model._fsdp_wrapped_module.") else k): v
                     for k, v in gen_sd.items()}
            pipeline.generator.load_state_dict(fixed, strict=False)
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)
    return pipeline, config


def reset_crossattn(pipeline):
    for c in pipeline.crossattn_cache:
        c["is_init"] = False


def set_rolling_window(pipeline, window, sink):
    """Enable a rolling local-attention KV window so we can roll past 21 frames.

    The chunkwise model defaults to global attention (local_attn_size=-1, caps at
    21 latent frames = 32760 tokens). Setting a window of `window` frames on the
    model + every self-attn block makes the KV cache roll (evict oldest, keep
    `sink` sink frames) — the same mechanism the long-video model uses (window=21).
    block_mask is per-chunk so it's unaffected by the window.
    """
    m = pipeline.generator.model
    m.local_attn_size = window
    m.block_mask = None
    for blk in m.blocks:
        a = blk.self_attn
        a.local_attn_size = window
        a.sink_size = sink
        a.max_attention_size = 32760 if window == -1 else window * 1560
    pipeline.local_attn_size = window  # pipeline sizes the cache from this


@torch.no_grad()
def rollout_with_swap(pipeline, *, base_prompt, new_prompt, swap_at, crossfade,
                      num_output_frames, seed, device, log, window=-1, sink=0):
    p = pipeline
    if window != -1:
        set_rolling_window(p, window, sink)
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn([1, num_output_frames, 16, 60, 104], generator=g,
                        dtype=torch.bfloat16).to(device)
    base = p.text_encoder(text_prompts=[base_prompt])
    new = p.text_encoder(text_prompts=[new_prompt]) if new_prompt else None

    bsz, num_frames = noise.shape[0], noise.shape[1]
    assert num_frames % p.num_frame_per_block == 0
    num_blocks = num_frames // p.num_frame_per_block

    p.kv_cache1 = None
    p._initialize_kv_cache(bsz, noise.dtype, device)
    p._initialize_crossattn_cache(bsz, noise.dtype, device)

    output = torch.zeros([bsz, num_frames, 16, 60, 104], device=device, dtype=noise.dtype)
    cond = base
    cur_start = 0
    per_chunk_ms = []
    swap_done = False

    for blk in range(num_blocks):
        if new is not None and blk >= swap_at:
            if crossfade > 0 and blk < swap_at + crossfade:
                a = (blk - swap_at + 1) / float(crossfade)
                cond = {"prompt_embeds": (1 - a) * base["prompt_embeds"] + a * new["prompt_embeds"]}
                reset_crossattn(p)
            elif not swap_done:
                cond = new
                reset_crossattn(p)
                log["swap_chunk"] = blk
                swap_done = True

        torch.cuda.synchronize(); t0 = time.time()
        cur = p.num_frame_per_block
        noisy = noise[:, cur_start:cur_start + cur]
        for i, ts in enumerate(p.denoising_step_list):
            timestep = torch.ones([bsz, cur], device=device, dtype=torch.int64) * ts
            _, den = p.generator(noisy_image_or_video=noisy, conditional_dict=cond,
                                 timestep=timestep, kv_cache=p.kv_cache1,
                                 crossattn_cache=p.crossattn_cache,
                                 current_start=cur_start * p.frame_seq_length)
            if i < len(p.denoising_step_list) - 1:
                nt = p.denoising_step_list[i + 1]
                noisy = p.scheduler.add_noise(
                    den.flatten(0, 1), torch.randn_like(den.flatten(0, 1)),
                    nt * torch.ones([bsz * cur], device=device, dtype=torch.long)
                ).unflatten(0, den.shape[:2])
        output[:, cur_start:cur_start + cur] = den
        ctx_t = torch.ones_like(timestep) * p.args.context_noise
        p.generator(noisy_image_or_video=den, conditional_dict=cond, timestep=ctx_t,
                    kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                    current_start=cur_start * p.frame_seq_length)
        torch.cuda.synchronize(); per_chunk_ms.append((time.time() - t0) * 1e3)
        cur_start += cur

    video = p.vae.decode_to_pixel(output, use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    return video, per_chunk_ms


def save_contact_sheet(frames_uint8, path, swap_px=None, n=18):
    idx = np.linspace(0, frames_uint8.shape[0] - 1, n).round().astype(int)
    tiles = []
    for i in idx:
        f = np.array(Image.fromarray(frames_uint8[i]).resize((166, 96)))
        if swap_px is not None and 0 <= i - swap_px < 6:  # red stripe at swap frame
            f[:4, :] = [255, 0, 0]
        tiles.append(f)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prompt2", default=None)
    ap.add_argument("--swap_at", type=int, default=3)
    ap.add_argument("--crossfade", type=int, default=0)
    ap.add_argument("--num_output_frames", type=int, default=21)
    ap.add_argument("--window", type=int, default=-1, help="rolling local-attn window (-1=global, caps at 21f)")
    ap.add_argument("--sink", type=int, default=0, help="attention-sink frames kept when rolling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    pipeline, _ = build_pipeline(args.config_path, args.checkpoint_path, args.use_ema, device)

    log = {}
    t0 = time.time()
    video, per_chunk_ms = rollout_with_swap(
        pipeline, base_prompt=args.prompt, new_prompt=args.prompt2,
        swap_at=args.swap_at, crossfade=args.crossfade,
        num_output_frames=args.num_output_frames, seed=args.seed, device=device, log=log,
        window=args.window, sink=args.sink)
    total_s = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames = (255.0 * rearrange(video, "b t c h w -> b t h w c").cpu())[0].to(torch.uint8).numpy()
    write_video(args.out, frames, fps=16)
    swap_px = log["swap_chunk"] * pipeline.num_frame_per_block * 4 if "swap_chunk" in log else None
    save_contact_sheet(frames, args.out.replace(".mp4", "_sheet.png"), swap_px=swap_px)

    print(f"[done] {args.out}  ({frames.shape[0]} px frames)")
    print(f"  mean chunk ms={sum(per_chunk_ms)/len(per_chunk_ms):.1f}  wall={total_s:.2f}s  "
          f"FPS(incl VAE)={frames.shape[0]/total_s:.2f}")
    if "swap_chunk" in log:
        print(f"  swapped at chunk {log['swap_chunk']} (latent {log['swap_chunk']*pipeline.num_frame_per_block}, "
              f"px frame {swap_px})")


if __name__ == "__main__":
    main()
