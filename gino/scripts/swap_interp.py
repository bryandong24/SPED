"""Interpolation-based smoothing for prompt swaps (kill abrupt cuts + jitter).

Builds on swap_control's cache constructions but adds TIME interpolation so the
handoff is gradual instead of a step:

  --pfade N            prompt crossfade: cross-attn context = mix(P_old,P_new,beta)
                       with beta ramping 0->1 over N post-swap frames.
  --vfade N            VALUE crossfade + K-PRESERVE: keep cached K_old (attention
                       routing / motion) fixed; blend cached V from V_old->V_new
                       over N frames (smooth content morph, motion held).
  --slerp              use spherical interpolation (norm-preserving) for the mix.
  --base {new,null,vswap,ortho}  cache construction applied at the swap.
  --grow_to / --post_window / --recache  as before.

Goal: smoother transitions and a more temporally-consistent subject.
"""
import argparse, os, math
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
    omega = torch.acos(dot); so = torch.sin(omega).clamp_min(1e-6)
    w = (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b
    return torch.where(so < 1e-4, (1 - t) * a + t * b, w)


def replay(pipe, prefix, cond_emb, device, w):
    K = prefix.shape[1]; s0 = max(0, K - w)
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, prefix.dtype, device)
    for c in pipe.crossattn_cache: c["is_init"] = False
    nb = pipe.num_frame_per_block; i = s0
    while i < K:
        cur = min(nb, K - i)
        ctx_t = torch.ones([1, cur], device=device, dtype=torch.int64) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=prefix[:, i:i + cur], conditional_dict={"prompt_embeds": cond_emb},
                       timestep=ctx_t, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                       current_start=(i - s0) * FRAME_SEQ)
        i += cur
    ks = [b["k"].clone() for b in pipe.kv_cache1]; vs = [b["v"].clone() for b in pipe.kv_cache1]
    ge = pipe.kv_cache1[0]["global_end_index"].clone(); le = pipe.kv_cache1[0]["local_end_index"].clone()
    return ks, vs, ge, le, s0


def neutralize(x, x_null):
    d = (x - x_null); u = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return x - (x * u).sum(-1, keepdim=True) * u


def build(pipe, prefix, sun_e, night_e, null_e, device, w, base):
    snap_new = replay(pipe, prefix, night_e, device, w)
    snap_old = replay(pipe, prefix, sun_e, device, w)
    snap_null = replay(pipe, prefix, null_e, device, w) if base in ("null", "ortho") else None
    nb = len(pipe.kv_cache1); ge, le, s0 = snap_new[2], snap_new[3], snap_new[4]
    fk, fv = [], []
    for b in range(nb):
        if base == "new":   k, v = snap_new[0][b], snap_new[1][b]
        elif base == "null": k, v = snap_null[0][b], snap_null[1][b]
        elif base == "vswap": k, v = snap_old[0][b], snap_new[1][b]
        elif base == "ortho":
            k = neutralize(snap_old[0][b], snap_null[0][b]); v = neutralize(snap_old[1][b], snap_null[1][b])
        fk.append(k); fv.append(v)
    pipe.kv_cache1 = None; pipe._initialize_kv_cache(1, prefix.dtype, device)
    for b in range(nb):
        pipe.kv_cache1[b]["k"].copy_(fk[b]); pipe.kv_cache1[b]["v"].copy_(fv[b])
        pipe.kv_cache1[b]["global_end_index"].copy_(ge); pipe.kv_cache1[b]["local_end_index"].copy_(le)
    for c in pipe.crossattn_cache: c["is_init"] = False
    # also return old/new V snapshots + prefix token count for V-fade
    pref_tok = int(le.item())
    return s0, snap_old[1], snap_new[1], pref_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="vswap", choices=["new", "null", "vswap", "ortho", "hardcut"])
    ap.add_argument("--pfade", type=int, default=0)
    ap.add_argument("--vfade", type=int, default=0)
    ap.add_argument("--slerp", action="store_true")
    ap.add_argument("--recache", type=int, default=9)
    ap.add_argument("--post_window", type=int, default=3)
    ap.add_argument("--grow_to", type=int, default=15)
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--swap_frame", type=int, default=30)
    ap.add_argument("--local_attn_size", type=int, default=21)
    ap.add_argument("--out_dir", default="../out/swap_interp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p1", default=SUN); ap.add_argument("--p2", default=NIGHT)
    args = ap.parse_args()
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/self_forcing_dmd.yaml"))
    gen = WanDiffusionWrapper(is_causal=True, local_attn_size=args.local_attn_size, sink_size=1)
    gen.load_state_dict(torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")["generator_ema"])
    pipe = CausalInferencePipeline(cfg, device=device, generator=gen)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
    nb = 3; pipe.num_frame_per_block = nb; pipe.generator.model.num_frame_per_block = nb

    sun = pipe.text_encoder(text_prompts=[args.p1])["prompt_embeds"]
    night = pipe.text_encoder(text_prompts=[args.p2])["prompt_embeds"]
    null = pipe.text_encoder(text_prompts=[""])["prompt_embeds"]

    g = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn([1, args.total, 16, 60, 104], generator=g, dtype=torch.bfloat16).to(device)
    pipe.kv_cache1 = None; pipe._initialize_kv_cache(1, noise.dtype, device)
    pipe._initialize_crossattn_cache(1, noise.dtype, device)
    out = torch.zeros_like(noise)

    cond = sun; start = 0; swapped_at = None; pos_offset = 0
    v_old = v_new = None; pref_tok = 0
    while start < args.total:
        cur = min(nb, args.total - start)
        if cond is sun and start + cur > args.swap_frame:
            cond = night; swapped_at = start
            if args.base == "hardcut":
                # PURE hard cut: keep the real (sun) cache, just swap the text.
                # No recache -> no snap; the prompt-ramp (pfade) reduces the
                # sun-cache vs night-prompt conflict that causes artifacts.
                for c in pipe.crossattn_cache: c["is_init"] = False
                pos_offset = 0
            else:
                pos_offset, v_old, v_new, pref_tok = build(pipe, out[:, :start], sun, night, null, device, args.recache, args.base)
            if args.post_window > 0:
                set_window(pipe.generator.model, args.post_window)
        # interpolation schedules (post-swap)
        if swapped_at is not None:
            fsraw = start - swapped_at
            if args.grow_to > 0:
                set_window(pipe.generator.model, min(args.grow_to, max(args.post_window, fsraw)))
            if args.pfade > 0:
                beta = min(1.0, (fsraw + cur) / float(args.pfade * nb))
                cond_emb = mix(sun, night, beta, args.slerp)
                for c in pipe.crossattn_cache: c["is_init"] = False
            else:
                cond_emb = night
            if args.vfade > 0 and v_old is not None and fsraw < args.vfade * nb + nb:
                beta = min(1.0, (fsraw + cur) / float(args.vfade * nb))
                for b in range(len(pipe.kv_cache1)):
                    blended = mix(v_old[b][:, :pref_tok], v_new[b][:, :pref_tok], beta, args.slerp)
                    pipe.kv_cache1[b]["v"][:, :pref_tok] = blended
        else:
            cond_emb = sun
        noisy = noise[:, start:start + cur]
        for i, ts in enumerate(pipe.denoising_step_list):
            timestep = torch.ones([1, cur], device=device, dtype=torch.int64) * ts
            _, den = pipe.generator(noisy_image_or_video=noisy, conditional_dict={"prompt_embeds": cond_emb},
                                    timestep=timestep, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                                    current_start=(start - pos_offset) * FRAME_SEQ)
            if i < len(pipe.denoising_step_list) - 1:
                nt = pipe.denoising_step_list[i + 1]
                noisy = pipe.scheduler.add_noise(den.flatten(0, 1), torch.randn_like(den.flatten(0, 1)),
                                                 nt * torch.ones([cur], device=device, dtype=torch.long)).unflatten(0, den.shape[:2])
        out[:, start:start + cur] = den
        ctx_t = torch.ones_like(timestep) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=den, conditional_dict={"prompt_embeds": cond_emb},
                       timestep=ctx_t, kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                       current_start=(start - pos_offset) * FRAME_SEQ)
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
    post = frames[sp:].astype(np.int16)
    jit = np.abs(np.diff(post, axis=0)).mean() if len(post) > 1 else 0
    # cut sharpness: pixel delta exactly at the swap boundary (lower=smoother)
    cut = np.abs(frames[sp].astype(np.int16) - frames[sp - 1].astype(np.int16)).mean() if sp > 0 else 0
    qs = [warmth[int(p * (len(warmth) - 1))] for p in (0, .25, .5, .75, 1.0)]
    print(f"[{args.name}] warmth=" + "/".join(f"{x:.0f}" for x in qs) +
          f" | jitter={jit:.2f} | cut@swap={cut:.1f}")


if __name__ == "__main__":
    main()
