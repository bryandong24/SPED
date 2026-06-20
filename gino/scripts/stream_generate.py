"""Simplest headless STREAMING generation for Self-Forcing.

Unlike inference.py (generate all 7 chunks, then decode once at the end), this
mirrors demo.py's real-time loop WITHOUT the Flask/socketio/browser stack:
generate a chunk -> immediately decode it with the block-causal streaming VAE
(demo_utils/vae_block3.VAEDecoderWrapper + a threaded vae_cache) -> emit its
pixel frames -> next chunk. Prints per-chunk generate+decode latency and the
true streaming FPS, and assembles the streamed frames into a final mp4.

This is the substrate for between-chunk text interruption: the swap hook goes
right at the top of the per-chunk loop (re-encode prompt + reset crossattn_cache
is_init). See ARCH_MAP.md / text_swap_poc.py.

Run:
  cd Self-Forcing && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \
    ../scripts/stream_generate.py --prompt "a dog running in a meadow" \
    --out ../out/stream/run.mp4
"""
import argparse
import os
import time

import numpy as np
import torch
import imageio
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from demo_utils.vae_block3 import VAEDecoderWrapper
from demo_utils.constant import ZERO_VAE_CACHE


def build_streaming_pipeline(config_path, checkpoint_path, device,
                             local_attn_size=21, sink_size=1):
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load(config_path))

    # Block-causal streaming VAE decoder (decoder weights only).
    vae_decoder = VAEDecoderWrapper()
    vae_sd = torch.load("wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", map_location="cpu")
    vae_decoder.load_state_dict({k: v for k, v in vae_sd.items()
                                 if "decoder." in k or "conv2" in k})
    vae_decoder.eval().to(dtype=torch.float16).requires_grad_(False).to(device)

    # local_attn_size>0 turns on the rolling sliding-window KV cache (with
    # eviction), enabling rollouts longer than the 21-frame global cache.
    # A small sink pins the first latent frame as a long-horizon anchor.
    transformer = WanDiffusionWrapper(is_causal=True, local_attn_size=local_attn_size,
                                      sink_size=sink_size)
    sd = torch.load(checkpoint_path, map_location="cpu")
    transformer.load_state_dict(sd["generator_ema"])
    # bf16 transformer (matches the bf16 text encoder, as in inference.py). The
    # streaming VAE stays fp16; we cast the latent to .half() at decode time.
    transformer.eval().to(dtype=torch.bfloat16).requires_grad_(False).to(device)

    text_encoder = WanTextEncoder()
    text_encoder.eval().to(dtype=torch.bfloat16).requires_grad_(False).to(device)

    pipeline = CausalInferencePipeline(config, device=device, generator=transformer,
                                       text_encoder=text_encoder, vae=vae_decoder)
    return pipeline, vae_decoder


@torch.no_grad()
def stream(pipeline, vae_decoder, *, prompt, seed, num_blocks, device, on_chunk):
    """Run the chunk-wise AR rollout, decoding+emitting each chunk as produced.

    on_chunk(idx, pixels[F,H,W,C] uint8, gen_ms, dec_ms) is called per chunk.
    Returns total wall time (s) and total emitted pixel-frame count.
    """
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])

    rnd = torch.Generator(device).manual_seed(seed)
    noise = torch.randn([1, num_blocks * pipeline.num_frame_per_block, 16, 60, 104],
                        device=device, dtype=torch.bfloat16, generator=rnd)
    cache_dtype = noise.dtype
    pipeline._initialize_kv_cache(1, cache_dtype, device)
    pipeline._initialize_crossattn_cache(1, cache_dtype, device)

    vae_cache = [c.to(device=device, dtype=torch.float16) for c in ZERO_VAE_CACHE]

    current_start = 0
    n_emitted = 0
    t_start = time.time()

    for idx in range(num_blocks):
        cur = pipeline.num_frame_per_block

        # ---- (text swap would go HERE: re-encode + reset crossattn is_init) ----

        torch.cuda.synchronize(); g0 = time.time()
        noisy = noise[:, current_start:current_start + cur]
        for i, t_cur in enumerate(pipeline.denoising_step_list):
            timestep = torch.ones([1, cur], device=device, dtype=torch.int64) * t_cur
            _, denoised = pipeline.generator(
                noisy_image_or_video=noisy, conditional_dict=conditional_dict,
                timestep=timestep, kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start * pipeline.frame_seq_length)
            if i < len(pipeline.denoising_step_list) - 1:
                nt = pipeline.denoising_step_list[i + 1]
                noisy = pipeline.scheduler.add_noise(
                    denoised.flatten(0, 1), torch.randn_like(denoised.flatten(0, 1)),
                    nt * torch.ones([cur], device=device, dtype=torch.long)
                ).unflatten(0, denoised.shape[:2])

        # clean-context pass: write this chunk's K/V into the self-attn cache
        if idx != num_blocks - 1:
            pipeline.generator(
                noisy_image_or_video=denoised, conditional_dict=conditional_dict,
                timestep=torch.zeros_like(timestep), kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start * pipeline.frame_seq_length)
        torch.cuda.synchronize(); gen_ms = (time.time() - g0) * 1e3

        # ---- streaming decode of THIS chunk only ----
        d0 = time.time()
        pixels, vae_cache = vae_decoder(denoised.half(), *vae_cache)
        if idx == 0:
            pixels = pixels[:, 3:]  # drop warm-up frames of first block (as in demo.py)
        torch.cuda.synchronize(); dec_ms = (time.time() - d0) * 1e3

        frames = (torch.clamp(pixels.float(), -1, 1) * 0.5 + 0.5)[0]  # [F,C,H,W]
        frames = (frames * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        on_chunk(idx, frames, gen_ms, dec_ms)
        n_emitted += frames.shape[0]
        current_start += cur

    return time.time() - t_start, n_emitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    ap.add_argument("--num_blocks", type=int, default=7)
    ap.add_argument("--local_attn_size", type=int, default=21,
                    help="rolling KV window in latent frames; -1 = global (unsafe beyond the fixed cache horizon)")
    ap.add_argument("--sink_size", type=int, default=1,
                    help="pinned attention-sink frames kept across eviction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--out", default="../out/stream/run.mp4")
    ap.add_argument("--dump_chunks", action="store_true",
                    help="also write each chunk as its own mp4 as it streams")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    print("loading models...")
    t0 = time.time()
    pipeline, vae_decoder = build_streaming_pipeline(
        args.config_path, args.checkpoint_path, device,
        local_attn_size=args.local_attn_size, sink_size=args.sink_size)
    print(f"models loaded in {time.time()-t0:.1f}s")

    all_frames = []

    def on_chunk(idx, frames, gen_ms, dec_ms):
        all_frames.append(frames)
        fps = frames.shape[0] / ((gen_ms + dec_ms) / 1e3)
        print(f"  chunk {idx}: {frames.shape[0]:2d} frames | gen {gen_ms:6.1f}ms "
              f"+ decode {dec_ms:5.1f}ms -> {fps:5.1f} FPS streaming")
        if args.dump_chunks:
            imageio.mimwrite(os.path.join(out_dir, f"chunk_{idx:02d}.mp4"),
                             frames, fps=args.fps, codec="libx264")

    print(f"streaming '{args.prompt}' ...")
    wall, n = stream(pipeline, vae_decoder, prompt=args.prompt, seed=args.seed,
                     num_blocks=args.num_blocks, device=device, on_chunk=on_chunk)

    video = np.concatenate(all_frames, axis=0)
    imageio.mimwrite(args.out, video, fps=args.fps, codec="libx264")
    print(f"\n[done] {args.out}")
    print(f"  {n} frames across {args.num_blocks} chunks in {wall:.2f}s "
          f"-> {n/wall:.1f} FPS end-to-end streaming "
          f"(playback {n/args.fps:.1f}s at {args.fps}fps)")


if __name__ == "__main__":
    main()
