"""Task 2/3 — between-chunk text-swap proof-of-concept for Self-Forcing.

Faithful reimplementation of CausalInferencePipeline.inference()'s chunk loop
(pipeline/causal_inference.py:176-244) with a mid-rollout TEXT SWAP hook:

  * --prompt / --prompt2 / --swap_at K : generate with `prompt`, then at chunk
    index K re-encode `prompt2` and flip every block's crossattn_cache["is_init"]
    = False so the new text K/V is recomputed (wan/modules/model.py:174-186).
    The self-attn KV cache of past frames (kv_cache1) is deliberately untouched
    -> this is the out-of-distribution regime the brief wants characterized.

Task-3 levers (cheap, training-free):
  * --crossfade N : linearly interpolate old->new prompt_embeds over N chunks
                    instead of a hard cut.
  * --evict_after_swap : after the swap, drop old-prompt frames from the
                    self-attn KV cache so the new prompt re-anchors faster.

NOTE on CFG: the distilled DMD few-step model runs ONE forward per denoise step
with no classifier-free guidance (causal_inference.py:188-221). The "raise CFG
on the new prompt" trick from the brief therefore has no native hook here; it
would require re-adding an unconditional pass. Documented, not implemented.

Run (after env ready):
  cd Self-Forcing && PYTHONPATH=. ../gino/.../python scripts/text_swap_poc.py \
      --config_path configs/self_forcing_dmd.yaml \
      --checkpoint_path checkpoints/self_forcing_dmd.pt --use_ema \
      --prompt  "A sunny green meadow with wildflowers, bright daylight, gentle breeze" \
      --prompt2 "A snowy meadow at night under a full moon, cold blue moonlight, falling snow" \
      --swap_at 3 --out ../gino/out/swap_hardcut.mp4
"""
import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf
from einops import rearrange
from torchvision.io import write_video

from pipeline import CausalInferencePipeline


def build_pipeline(config_path, checkpoint_path, use_ema, device):
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), config)
    pipeline = CausalInferencePipeline(config, device=device)
    if checkpoint_path:
        sd = torch.load(checkpoint_path, map_location="cpu")
        pipeline.generator.load_state_dict(sd["generator_ema" if use_ema else "generator"])
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)
    return pipeline, config


def encode(pipeline, prompt):
    """Return the conditional_dict {'prompt_embeds': [B, L, C]} for one prompt."""
    return pipeline.text_encoder(text_prompts=[prompt])


def reset_crossattn(pipeline):
    """Force every block to re-encode text K/V on the next forward."""
    for c in pipeline.crossattn_cache:
        c["is_init"] = False


@torch.no_grad()
def rollout_with_swap(pipeline, *, base_prompt, new_prompt, swap_at, crossfade,
                      evict_after_swap, num_output_frames, seed, device,
                      interrupt_log):
    """Chunk-wise AR rollout that swaps text at chunk `swap_at`.

    Mirrors CausalInferencePipeline.inference() but re-binds conditional_dict
    between chunks. Returns (video[B,T,C,H,W] in [0,1], per_chunk_latency_ms).
    """
    p = pipeline
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn([1, num_output_frames, 16, 60, 104], generator=g,
                        dtype=torch.bfloat16).to(device)

    base = encode(p, base_prompt)
    new = encode(p, new_prompt) if new_prompt is not None else None

    batch_size, num_frames = noise.shape[0], noise.shape[1]
    assert num_frames % p.num_frame_per_block == 0
    num_blocks = num_frames // p.num_frame_per_block

    p.kv_cache1 = None  # fresh caches
    p._initialize_kv_cache(batch_size, noise.dtype, device)
    p._initialize_crossattn_cache(batch_size, noise.dtype, device)

    output = torch.zeros([batch_size, num_frames, 16, 60, 104], device=device,
                         dtype=noise.dtype)

    conditional_dict = base
    current_start_frame = 0
    per_chunk_ms = []
    swap_done = False

    for blk in range(num_blocks):
        # ---- TEXT SWAP between chunks ----
        if new is not None and blk >= swap_at:
            if crossfade > 0 and blk < swap_at + crossfade:
                a = (blk - swap_at + 1) / float(crossfade)  # 0<..<=1
                mixed = (1 - a) * base["prompt_embeds"] + a * new["prompt_embeds"]
                conditional_dict = {"prompt_embeds": mixed}
                reset_crossattn(p)  # re-encode mixed text each fade step
            elif not swap_done:
                conditional_dict = new
                reset_crossattn(p)
                if evict_after_swap:
                    _evict_kv(p)
                interrupt_log["swap_chunk"] = blk
                interrupt_log["swap_t"] = time.time()
                swap_done = True

        torch.cuda.synchronize(); t0 = time.time()
        cur = p.num_frame_per_block
        noisy_input = noise[:, current_start_frame:current_start_frame + cur]

        for index, current_timestep in enumerate(p.denoising_step_list):
            timestep = torch.ones([batch_size, cur], device=device,
                                  dtype=torch.int64) * current_timestep
            _, denoised_pred = p.generator(
                noisy_image_or_video=noisy_input, conditional_dict=conditional_dict,
                timestep=timestep, kv_cache=p.kv_cache1,
                crossattn_cache=p.crossattn_cache,
                current_start=current_start_frame * p.frame_seq_length)
            if index < len(p.denoising_step_list) - 1:
                nt = p.denoising_step_list[index + 1]
                noisy_input = p.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    nt * torch.ones([batch_size * cur], device=device, dtype=torch.long),
                ).unflatten(0, denoised_pred.shape[:2])

        output[:, current_start_frame:current_start_frame + cur] = denoised_pred

        # clean-context pass writes past-frame K/V (causal_inference.py:226-235)
        ctx_t = torch.ones_like(timestep) * p.args.context_noise
        p.generator(noisy_image_or_video=denoised_pred, conditional_dict=conditional_dict,
                    timestep=ctx_t, kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                    current_start=current_start_frame * p.frame_seq_length)

        torch.cuda.synchronize(); per_chunk_ms.append((time.time() - t0) * 1e3)
        current_start_frame += cur

    video = p.vae.decode_to_pixel(output, use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    return video, per_chunk_ms


def _evict_kv(pipeline):
    """Task-3 lever: drop accumulated past-frame K/V so the new prompt re-anchors.

    Resets each block's self-attn cache pointers to zero (keeps allocation).
    Aggressive (full reset) — a softer version would keep only `sink_size`
    frames. See ARCH_MAP.md §3.
    """
    for kv in pipeline.kv_cache1:
        kv["global_end_index"].zero_()
        kv["local_end_index"].zero_()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prompt2", default=None, help="swap-to prompt; omit for baseline")
    ap.add_argument("--swap_at", type=int, default=3, help="chunk index of the swap")
    ap.add_argument("--crossfade", type=int, default=0, help="fade over N chunks (0=hard cut)")
    ap.add_argument("--evict_after_swap", action="store_true")
    ap.add_argument("--num_output_frames", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    pipeline, _ = build_pipeline(args.config_path, args.checkpoint_path, args.use_ema, device)

    interrupt_log = {}
    t_start = time.time()
    video, per_chunk_ms = rollout_with_swap(
        pipeline, base_prompt=args.prompt, new_prompt=args.prompt2,
        swap_at=args.swap_at, crossfade=args.crossfade,
        evict_after_swap=args.evict_after_swap,
        num_output_frames=args.num_output_frames, seed=args.seed, device=device,
        interrupt_log=interrupt_log)
    total_s = time.time() - t_start

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames = (255.0 * rearrange(video, "b t c h w -> b t h w c").cpu())[0]
    write_video(args.out, frames, fps=16)

    n_frames_px = frames.shape[0]
    print(f"[done] {args.out}")
    print(f"  chunks={len(per_chunk_ms)}  px_frames={n_frames_px}")
    print(f"  per-chunk denoise+ctx ms: {[round(x,1) for x in per_chunk_ms]}")
    print(f"  mean chunk ms={sum(per_chunk_ms)/len(per_chunk_ms):.1f}  total wall={total_s:.2f}s")
    print(f"  end-to-end FPS (incl VAE) = {n_frames_px/total_s:.2f}")
    if "swap_chunk" in interrupt_log:
        print(f"  swap at chunk {interrupt_log['swap_chunk']} "
              f"(latent frame {interrupt_log['swap_chunk']*pipeline.num_frame_per_block})")


if __name__ == "__main__":
    main()
