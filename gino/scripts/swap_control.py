"""Control-theoretic / mechinterp cache constructions for prompt swapping.

At the swap we rebuild the self-attn KV cache from the generated prefix, but
instead of plain recache-under-new-prompt we try principled constructions that
separate STRUCTURE from OLD-PROMPT influence:

  --mode new     : plain recache under new prompt (baseline; full replacement)
  --mode null    : recache under the EMPTY prompt -> prompt-neutral structure;
                   new frames inject new semantics via their own cross-attn
  --mode vswap   : K-preserve / V-swap -> keep K_old (attention routing = motion/
                   structure), swap V->V_new (content/appearance). Mechinterp:
                   K = where you attend, V = what you read.
  --mode kswap   : ablation -> K_new + V_old
  --mode ortho   : minimum-norm old-prompt removal -> per (token,head) remove the
                   component of K/V along the old-prompt direction (K_old - K_null),
                   keeping the orthogonal complement (structure). New semantics
                   then come from the new frames' cross-attn.
  --mode ortho_new : ortho-neutralize, then add a scaled new-prompt delta.

K = sun frames re-encoded under old/new/null prompts give K_old/K_new/K_null.
Optional --grow_to combines with the growing-window stability schedule.
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


def set_window(model, W):
    for blk in model.blocks:
        blk.self_attn.local_attn_size = W
        blk.self_attn.max_attention_size = W * FRAME_SEQ


def replay(pipe, prefix, cond_emb, device, w):
    """Replay last w frames of prefix under cond_emb; return per-block (k,v) snapshots
    + (global_end, local_end, s0). Leaves kv_cache1 populated under cond_emb."""
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
    ks = [b["k"].clone() for b in pipe.kv_cache1]
    vs = [b["v"].clone() for b in pipe.kv_cache1]
    ge = pipe.kv_cache1[0]["global_end_index"].clone()
    le = pipe.kv_cache1[0]["local_end_index"].clone()
    return ks, vs, ge, le, s0


def neutralize(x, x_null):
    """Remove, per (token,head) vector, the component of x along (x - x_null)."""
    d = (x - x_null)
    n = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    u = d / n
    proj = (x * u).sum(dim=-1, keepdim=True) * u
    return x - proj


def build_cache(pipe, prefix, sun_e, night_e, null_e, device, w, mode, new_scale=0.0):
    need_old = mode in ("vswap", "kswap", "ortho", "ortho_new")
    need_null = mode in ("ortho", "ortho_new", "null")
    snaps = {}
    if mode in ("new", "vswap", "kswap", "ortho_new"):
        snaps["new"] = replay(pipe, prefix, night_e, device, w)
    if mode == "null":
        snaps["null"] = replay(pipe, prefix, null_e, device, w)
    if need_old:
        snaps["old"] = replay(pipe, prefix, sun_e, device, w)
    if need_null and "null" not in snaps:
        snaps["null"] = replay(pipe, prefix, null_e, device, w)

    nb_blocks = len(pipe.kv_cache1)
    fk, fv = [None] * nb_blocks, [None] * nb_blocks
    ref = snaps.get("new") or snaps.get("null") or snaps.get("old")
    ge, le, s0 = ref[2], ref[3], ref[4]
    for b in range(nb_blocks):
        if mode == "new":
            fk[b], fv[b] = snaps["new"][0][b], snaps["new"][1][b]
        elif mode == "null":
            fk[b], fv[b] = snaps["null"][0][b], snaps["null"][1][b]
        elif mode == "vswap":   # keep K_old (routing/motion), swap V_new (content)
            fk[b], fv[b] = snaps["old"][0][b], snaps["new"][1][b]
        elif mode == "kswap":   # ablation
            fk[b], fv[b] = snaps["new"][0][b], snaps["old"][1][b]
        elif mode in ("ortho", "ortho_new"):
            ko, vo = snaps["old"][0][b], snaps["old"][1][b]
            kn, vn = snaps["null"][0][b], snaps["null"][1][b]
            k = neutralize(ko, kn); v = neutralize(vo, vn)
            if mode == "ortho_new":
                k = k + new_scale * (snaps["new"][0][b] - kn)   # add scaled new-prompt delta
                v = v + new_scale * (snaps["new"][1][b] - vn)
            fk[b], fv[b] = k, v

    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, prefix.dtype, device)
    for b in range(nb_blocks):
        pipe.kv_cache1[b]["k"].copy_(fk[b]); pipe.kv_cache1[b]["v"].copy_(fv[b])
        pipe.kv_cache1[b]["global_end_index"].copy_(ge)
        pipe.kv_cache1[b]["local_end_index"].copy_(le)
    for c in pipe.crossattn_cache: c["is_init"] = False  # continued gen refills with new prompt
    return s0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["new", "null", "vswap", "kswap", "ortho", "ortho_new"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--recache", type=int, default=9)
    ap.add_argument("--post_window", type=int, default=5)
    ap.add_argument("--grow_to", type=int, default=0)
    ap.add_argument("--new_scale", type=float, default=0.5)
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--swap_frame", type=int, default=30)
    ap.add_argument("--local_attn_size", type=int, default=21)
    ap.add_argument("--sink_size", type=int, default=1)
    ap.add_argument("--out_dir", default="../out/swap_ctrl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p1", default=SUN); ap.add_argument("--p2", default=NIGHT)
    args = ap.parse_args()
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/self_forcing_dmd.yaml"))
    gen = WanDiffusionWrapper(is_causal=True, local_attn_size=args.local_attn_size, sink_size=args.sink_size)
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
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, noise.dtype, device)
    pipe._initialize_crossattn_cache(1, noise.dtype, device)
    out = torch.zeros_like(noise)

    cond = sun; start = 0; swapped_at = None; pos_offset = 0
    while start < args.total:
        cur = min(nb, args.total - start)
        if cond is sun and start + cur > args.swap_frame:
            cond = night; swapped_at = start
            pos_offset = build_cache(pipe, out[:, :start], sun, night, null, device, args.recache, args.mode, args.new_scale)
            if args.post_window > 0 and args.grow_to == 0:
                set_window(pipe.generator.model, args.post_window)
        if swapped_at is not None and args.grow_to > 0:
            W = min(args.grow_to, max(args.post_window if args.post_window > 0 else nb, start - swapped_at))
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

    video = (pipe.vae.decode_to_pixel(out, use_cache=True) * 0.5 + 0.5).clamp(0, 1)
    frames = (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()
    imageio.mimwrite(os.path.join(args.out_dir, f"{args.name}.mp4"), frames, fps=16, codec="libx264")
    idxs = np.linspace(0, frames.shape[0] - 1, 16).round().astype(int)
    swap_px = args.swap_frame * 4
    tiles = [frames[i].copy() for i in idxs]
    for t, i in zip(tiles, idxs):
        if i >= swap_px: t[:5, :] = [255, 0, 0]
    Image.fromarray(np.concatenate(tiles, axis=1)).save(os.path.join(args.out_dir, f"{args.name}_sheet.png"))
    w = frames.reshape(frames.shape[0], -1, 3).mean(axis=1); warmth = w[:, 0] - w[:, 2]
    # crude temporal-stability proxy: mean abs frame-to-frame pixel delta, post-swap
    post = frames[swap_px:].astype(np.int16)
    jitter = np.abs(np.diff(post, axis=0)).mean() if len(post) > 1 else 0.0
    qs = [warmth[int(p * (len(warmth) - 1))] for p in (0, .2, .4, .6, .8, 1.0)]
    print(f"[{args.name}] mode={args.mode} warmth=" + "/".join(f"{x:.0f}" for x in qs) +
          f" | post-swap jitter={jitter:.2f}")


if __name__ == "__main__":
    main()
