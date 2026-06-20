"""Prompt-vector swap experiments for Self-Forcing (Task 2/3).

Loads the model ONCE and runs a list of variants, each producing:
  - <name>.mp4              (full 5s rollout, clean full-VAE decode)
  - <name>_sheet.png        (frame contact sheet for eyeballing the transition)
  - appends a per-frame luminance row to luminance.csv

Swap mechanism (verified): re-bind conditional_dict to a PRECOMPUTED embedding +
flip every block's crossattn_cache["is_init"]=False. Self-attn KV cache (past
frames) and latents are untouched. Levers:
  - hardcut: instantaneous embedding swap at chunk K.
  - crossfade N: lerp old->new prompt_embeds over N chunks.
  - evict {none|full|soft}: after swap, drop old-prompt frames from the self-attn
    KV cache so the new prompt re-anchors faster (full=reset, soft=keep sink+last).

Metric: sun->winter-night is bright->dark, so mean per-frame luminance tracks the
interrupt->response. Compared against all-sun / all-night references (same seed).
"""
import argparse, csv, os, time
import numpy as np
import torch
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from pipeline import CausalInferencePipeline

FRAME_SEQ = 1560


def build_pipeline(config_path, ckpt, device):
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load(config_path))
    pipe = CausalInferencePipeline(cfg, device=device)
    sd = torch.load(ckpt, map_location="cpu")
    pipe.generator.load_state_dict(sd["generator_ema"])
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
    return pipe


def reset_xattn(pipe):
    for c in pipe.crossattn_cache:
        c["is_init"] = False


def evict_kv(pipe, mode):
    """Drop old-prompt frames from the self-attn cache. full=reset to empty;
    soft=keep frame 0 (sink) + the most recent frame, drop the middle."""
    if mode == "full":
        for kv in pipe.kv_cache1:
            kv["global_end_index"].zero_(); kv["local_end_index"].zero_()
    elif mode == "soft":
        for kv in pipe.kv_cache1:
            E = kv["local_end_index"].item()
            if E <= 2 * FRAME_SEQ:
                continue
            # keep tokens [0:FRAME_SEQ] (frame0) and last frame [E-FRAME_SEQ:E]
            last = kv["k"][:, E - FRAME_SEQ:E].clone()
            kv["k"][:, FRAME_SEQ:2 * FRAME_SEQ] = last
            last_v = kv["v"][:, E - FRAME_SEQ:E].clone()
            kv["v"][:, FRAME_SEQ:2 * FRAME_SEQ] = last_v
            kv["local_end_index"].fill_(2 * FRAME_SEQ)
            # global stays (RoPE absolute positions preserved for the kept window)


@torch.no_grad()
def rollout(pipe, base_emb, new_emb, *, swap_at, crossfade, evict, num_blocks,
            seed, device):
    cur = pipe.num_frame_per_block
    g = torch.Generator("cpu").manual_seed(seed)
    noise = torch.randn([1, num_blocks * cur, 16, 60, 104], generator=g,
                        dtype=torch.bfloat16).to(device)
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, noise.dtype, device)
    pipe._initialize_crossattn_cache(1, noise.dtype, device)
    out = torch.zeros([1, num_blocks * cur, 16, 60, 104], device=device, dtype=noise.dtype)

    cond = {"prompt_embeds": base_emb}
    start = 0; per_chunk_ms = []; swap_frame = None
    for blk in range(num_blocks):
        # ---- swap logic between chunks ----
        if new_emb is not None and blk >= swap_at:
            if crossfade > 0 and blk < swap_at + crossfade:
                a = (blk - swap_at + 1) / float(crossfade + 1)
                cond = {"prompt_embeds": (1 - a) * base_emb + a * new_emb}
                reset_xattn(pipe)
            elif blk == swap_at or (crossfade > 0 and blk == swap_at + crossfade):
                cond = {"prompt_embeds": new_emb}
                reset_xattn(pipe)
                if blk == swap_at and evict != "none":
                    evict_kv(pipe, evict)
                if swap_frame is None:
                    swap_frame = start  # latent frame index of swap

        torch.cuda.synchronize(); t0 = time.time()
        noisy = noise[:, start:start + cur]
        for i, ts in enumerate(pipe.denoising_step_list):
            timestep = torch.ones([1, cur], device=device, dtype=torch.int64) * ts
            _, denoised = pipe.generator(
                noisy_image_or_video=noisy, conditional_dict=cond, timestep=timestep,
                kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                current_start=start * FRAME_SEQ)
            if i < len(pipe.denoising_step_list) - 1:
                nt = pipe.denoising_step_list[i + 1]
                noisy = pipe.scheduler.add_noise(
                    denoised.flatten(0, 1), torch.randn_like(denoised.flatten(0, 1)),
                    nt * torch.ones([cur], device=device, dtype=torch.long)
                ).unflatten(0, denoised.shape[:2])
        out[:, start:start + cur] = denoised
        # clean-context pass (writes this chunk's K/V into self-attn cache)
        ctx_t = torch.ones_like(timestep) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=denoised, conditional_dict=cond,
                       timestep=ctx_t, kv_cache=pipe.kv_cache1,
                       crossattn_cache=pipe.crossattn_cache, current_start=start * FRAME_SEQ)
        torch.cuda.synchronize(); per_chunk_ms.append((time.time() - t0) * 1e3)
        start += cur

    video = pipe.vae.decode_to_pixel(out, use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    frames = (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()
    swap_pixel = None if swap_frame is None else (swap_frame * 4)  # latent->pixel (approx)
    return frames, per_chunk_ms, swap_pixel


def contact_sheet(frames, path, cols=9, swap_pixel=None):
    n = frames.shape[0]
    idxs = np.linspace(0, n - 1, cols).round().astype(int)
    tiles = []
    for i in idxs:
        f = frames[i].copy()
        if swap_pixel is not None and i >= swap_pixel:
            f[:4, :] = [255, 0, 0]  # red top stripe = post-swap frames
        tiles.append(f)
    sheet = np.concatenate(tiles, axis=1)
    Image.fromarray(sheet).save(path)
    return idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    ap.add_argument("--out_dir", default="../out/swap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--swap_at", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    SUN = "a dog running in a sunny green meadow full of wildflowers, bright daylight, cinematic, highly detailed"
    NIGHT = "a dog running through a snowy field on a cold winter night, deep blue moonlight, falling snow, cinematic, highly detailed"

    t0 = time.time()
    pipe = build_pipeline(args.config_path, args.checkpoint_path, device)
    print(f"model loaded in {time.time()-t0:.1f}s")
    sun_emb = pipe.text_encoder(text_prompts=[SUN])["prompt_embeds"]
    night_emb = pipe.text_encoder(text_prompts=[NIGHT])["prompt_embeds"]
    print(f"precomputed both embeddings: sun {tuple(sun_emb.shape)}, night {tuple(night_emb.shape)}")

    # (name, base, new, crossfade, evict)
    variants = [
        ("ref_sun",      sun_emb,   None,      0, "none"),
        ("ref_night",    night_emb, None,      0, "none"),
        ("hardcut",      sun_emb,   night_emb, 0, "none"),
        ("crossfade2",   sun_emb,   night_emb, 2, "none"),
        ("evict_soft",   sun_emb,   night_emb, 0, "soft"),
    ]

    csv_path = os.path.join(args.out_dir, "luminance.csv")
    rows = []
    for name, base, new, cf, ev in variants:
        st = time.time()
        frames, ms, swap_px = rollout(pipe, base, new, swap_at=args.swap_at,
                                      crossfade=cf, evict=ev, num_blocks=7,
                                      seed=args.seed, device=device)
        mp4 = os.path.join(args.out_dir, f"{name}.mp4")
        import imageio; imageio.mimwrite(mp4, frames, fps=16, codec="libx264")
        contact_sheet(frames, os.path.join(args.out_dir, f"{name}_sheet.png"), swap_pixel=swap_px)
        lum = frames.reshape(frames.shape[0], -1, 3).mean(axis=(1, 2))  # per-frame mean luminance
        rows.append((name, lum, swap_px))
        print(f"[{name}] {frames.shape[0]} frames in {time.time()-st:.1f}s | "
              f"swap_px={swap_px} | lum first/mid/last = "
              f"{lum[0]:.0f}/{lum[len(lum)//2]:.0f}/{lum[-1]:.0f} | "
              f"mean chunk {np.mean(ms):.0f}ms")

    # luminance CSV + interrupt->response analysis
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        maxlen = max(len(l) for _, l, _ in rows)
        w.writerow(["frame"] + [n for n, _, _ in rows])
        for i in range(maxlen):
            w.writerow([i] + [f"{l[i]:.1f}" if i < len(l) else "" for _, l, _ in rows])

    print("\n=== interrupt->response (luminance) ===")
    sun_lum = dict(rows[0:1])  # not used directly
    for name, lum, swap_px in rows:
        if swap_px is None:
            continue
        pre = lum[max(0, swap_px - 8):swap_px].mean() if swap_px > 0 else lum[:8].mean()
        # first post-swap frame dropping >15% below pre-swap brightness
        resp = None
        for i in range(swap_px, len(lum)):
            if lum[i] < 0.85 * pre:
                resp = i; break
        lat = None if resp is None else (resp - swap_px)
        print(f"  {name:12s} swap@px{swap_px:3d} pre-lum {pre:.0f} -> end {lum[-1]:.0f} "
              f"| response @+{lat} frames ({'%.2fs'%(lat/16) if lat is not None else 'n/a'})")
    print(f"\nwrote {csv_path} and per-variant mp4/sheet to {args.out_dir}")


if __name__ == "__main__":
    main()
