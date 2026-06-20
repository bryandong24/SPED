"""Baseline HARD-CUT prompt swap on Rolling Forcing.

RF supports prompt switching natively ("discard the cross-attention cache of
previous text prompts and apply the new prompts"). This replicates RF's
rolling-window inference (pipeline/rolling_forcing_inference.py) but switches the
conditional_dict from prompt A -> B at a chosen block, resetting the cross-attn
is_init so new text K/V are recomputed. The visual KV cache (attention sink +
temporal window) is untouched -> RF's long-horizon anchoring should make this
hard cut smoother than on Self-Forcing.
"""
import argparse, os
from collections import OrderedDict
import numpy as np, torch, imageio
from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from pipeline import CausalInferencePipeline


def build(config_path, ckpt, device):
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(config_path))
    pipe = CausalInferencePipeline(config, device=device)
    sd = torch.load(ckpt, map_location="cpu")["generator_ema"]
    sd = OrderedDict((k.replace("_fsdp_wrapped_module.", ""), v) for k, v in sd.items())
    pipe.generator.load_state_dict(sd)
    pipe = pipe.to(device=device, dtype=torch.bfloat16)
    return pipe


@torch.no_grad()
def rolling_swap(pipe, noise, cond_a, cond_b, swap_block, kv_reset=False, recache=0, ramp=0, full_reset=False, warm_seed=False):
    """RF rolling-window inference with a prompt swap at `swap_block`.
    Interventions to break RF's resistance (attention-sink anchors old scene):
      kv_reset : wipe the clean KV cache (drop sink + history) at the swap.
      recache  : re-encode the last `recache` committed frames UNDER the new prompt
                 into a fresh cache (drop old semantics, keep recent structure).
      ramp     : ramp prompt A->B over `ramp` blocks instead of a hard cut.
    """
    b, num_frames, C, H, W = noise.shape
    nfp = pipe.num_frame_per_block
    num_blocks = num_frames // nfp
    FS = pipe.frame_seq_length
    dev = noise.device

    def mix(t):
        return {"prompt_embeds": (1 - t) * cond_a["prompt_embeds"] + t * cond_b["prompt_embeds"]}

    pipe._initialize_kv_cache(b, noise.dtype, dev)
    pipe._initialize_crossattn_cache(b, noise.dtype, dev)
    output = torch.zeros_like(noise)
    noisy_cache = torch.zeros_like(noise)

    nstep = len(pipe.denoising_step_list)
    win_len = nstep
    window_num = num_blocks + win_len - 1
    starts = [max(0, wi - win_len + 1) for wi in range(window_num)]
    ends = [min(num_blocks - 1, wi) for wi in range(window_num)]

    shared_t = torch.ones([b, win_len * nfp], device=dev, dtype=torch.float32)
    for i, ct in enumerate(reversed(pipe.denoising_step_list)):
        shared_t[:, i * nfp:(i + 1) * nfp] *= ct

    cond = cond_a
    swapped = False
    for wi in range(window_num):
        sb, eb = starts[wi], ends[wi]
        cs = sb * nfp; ce = (eb + 1) * nfp; cn = ce - cs

        # ---- SWAP: when the newest entering block reaches swap_block ----
        if (not swapped) and eb >= swap_block:
            cond = cond_b
            for c in pipe.crossattn_cache:
                c["is_init"] = False
            if recache > 0 or kv_reset or full_reset:
                pipe._initialize_kv_cache(b, noise.dtype, dev)  # wipe clean cache (sink+history)
                for c in pipe.crossattn_cache:
                    c["is_init"] = False
            if recache > 0:
                # SEED the fresh clean cache with the last `recache` committed frames
                # RE-ENCODED under the new prompt (gives night context -> less washed gap)
                s0 = max(0, cs - recache)
                fi = s0
                while fi < cs:
                    chunk = output[:, fi:fi + nfp]
                    ctx = torch.ones([b, nfp], device=dev, dtype=torch.float32) * pipe.args.context_noise
                    pipe.generator(noisy_image_or_video=chunk, conditional_dict=cond_b, timestep=ctx,
                                   kv_cache=pipe.kv_cache_clean, crossattn_cache=pipe.crossattn_cache,
                                   current_start=fi * FS, updating_cache=True)
                    fi += nfp
            if full_reset:
                noisy_cache[:, cs:] = 0  # flush the in-flight (old-noised) rolling staircase
                if warm_seed:
                    # WARM-START the staircase: generate one night frame, then re-noise it
                    # to each staircase level so the rolling window has coherent night
                    # content instead of zeros (removes the grey cold-start gap).
                    nx = noise[:, cs:cs + nfp].clone()
                    for i, tt in enumerate(pipe.denoising_step_list):
                        ts1 = torch.ones([b, nfp], device=dev, dtype=torch.float32) * tt
                        _, x0w = pipe.generator(noisy_image_or_video=nx, conditional_dict=cond_b,
                                                timestep=ts1, kv_cache=pipe.kv_cache_clean,
                                                crossattn_cache=pipe.crossattn_cache, current_start=cs * FS)
                        if i < len(pipe.denoising_step_list) - 1:
                            nt = pipe.denoising_step_list[i + 1].to(dev)
                            nx = pipe.scheduler.add_noise(
                                x0w.flatten(0, 1), torch.randn_like(x0w.flatten(0, 1)),
                                nt * torch.ones([b * nfp], device=dev, dtype=torch.long)).unflatten(0, x0w.shape[:2])
                    rev = list(reversed(pipe.denoising_step_list.tolist()))  # clean->noisy levels
                    for p in range(min(win_len - 1, (num_frames - cs) // nfp)):
                        lvl = int(rev[p])
                        seed = pipe.scheduler.add_noise(
                            x0w.flatten(0, 1), torch.randn_like(x0w.flatten(0, 1)),
                            lvl * torch.ones([b * nfp], device=dev, dtype=torch.long)).unflatten(0, x0w.shape[:2])
                        noisy_cache[:, cs + p * nfp:cs + (p + 1) * nfp] = seed
            swapped = True
        # prompt ramp (overrides cond after swap for `ramp` blocks)
        if ramp > 0 and swapped:
            t = min(1.0, (eb - swap_block + 1) / float(ramp))
            cond = mix(t)
            for c in pipe.crossattn_cache:
                c["is_init"] = False

        if cn == win_len * nfp or cs == 0:
            noisy_input = torch.cat([noisy_cache[:, cs:ce - nfp], noise[:, ce - nfp:ce]], dim=1)
        else:
            noisy_input = noisy_cache[:, cs:ce]

        if cn == win_len * nfp:
            ct = shared_t
        elif cs == 0:
            ct = shared_t[:, -cn:]
        else:
            ct = shared_t[:, :cn]

        _, denoised = pipe.generator(noisy_image_or_video=noisy_input, conditional_dict=cond,
                                     timestep=ct, kv_cache=pipe.kv_cache_clean,
                                     crossattn_cache=pipe.crossattn_cache, current_start=cs * FS)
        output[:, cs:ce] = denoised

        for bi in range(sb, eb + 1):
            bt = ct[:, (bi - sb) * nfp:(bi - sb + 1) * nfp].mean().item()
            idx = torch.nonzero(torch.abs(pipe.denoising_step_list - bt) < 1e-4, as_tuple=True)[0]
            if idx == len(pipe.denoising_step_list) - 1:
                continue
            nt = pipe.denoising_step_list[idx + 1].to(dev)
            noisy_cache[:, bi * nfp:(bi + 1) * nfp] = pipe.scheduler.add_noise(
                denoised.flatten(0, 1), torch.randn_like(denoised.flatten(0, 1)),
                nt * torch.ones([b * cn], device=dev, dtype=torch.long)
            ).unflatten(0, denoised.shape[:2])[:, (bi - sb) * nfp:(bi - sb + 1) * nfp]

        # commit front block to clean cache
        ctx_t = torch.ones_like(ct) * pipe.args.context_noise
        d0 = denoised[:, :nfp]; ctx0 = ctx_t[:, :nfp]
        pipe.generator(noisy_image_or_video=d0, conditional_dict=cond, timestep=ctx0,
                       kv_cache=pipe.kv_cache_clean, crossattn_cache=pipe.crossattn_cache,
                       current_start=cs * FS, updating_cache=True)

    video = (pipe.vae.decode_to_pixel(output, use_cache=False) * 0.5 + 0.5).clamp(0, 1)
    return (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--p1", required=True); ap.add_argument("--p2", required=True)
    ap.add_argument("--swap_frame", type=int, default=12)
    ap.add_argument("--kv_reset", action="store_true")
    ap.add_argument("--recache", type=int, default=0)
    ap.add_argument("--ramp", type=int, default=0)
    ap.add_argument("--full_reset", action="store_true")
    ap.add_argument("--warm_seed", action="store_true")
    ap.add_argument("--num_frames", type=int, default=63, help="latent frames (must be mult of 3)")
    ap.add_argument("--config_path", default="configs/rolling_forcing_dmd.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/rolling_forcing_dmd.pt")
    ap.add_argument("--out_dir", default="../out/rf_swap")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = build(args.config_path, args.checkpoint_path, device)
    cond_a = pipe.text_encoder(text_prompts=[args.p1])
    cond_b = pipe.text_encoder(text_prompts=[args.p2])

    g = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn([1, args.num_frames, 16, 60, 104], generator=g, dtype=torch.bfloat16).to(device)
    swap_block = args.swap_frame // pipe.num_frame_per_block
    frames = rolling_swap(pipe, noise, cond_a, cond_b, swap_block,
                          kv_reset=args.kv_reset, recache=args.recache, ramp=args.ramp, full_reset=args.full_reset, warm_seed=args.warm_seed)

    imageio.mimwrite(os.path.join(args.out_dir, f"{args.name}.mp4"), frames, fps=16, codec="libx264")
    idxs = np.linspace(0, frames.shape[0] - 1, 16).round().astype(int)
    sp = args.swap_frame * 4
    tiles = [frames[i].copy() for i in idxs]
    for t, i in zip(tiles, idxs):
        if i >= sp: t[:5, :] = [255, 0, 0]
    Image.fromarray(np.concatenate(tiles, axis=1)).save(os.path.join(args.out_dir, f"{args.name}_sheet.png"))
    w = frames.reshape(frames.shape[0], -1, 3).mean(1); warmth = w[:, 0] - w[:, 2]
    qs = [warmth[int(p * (len(warmth) - 1))] for p in (0, .25, .5, .75, 1.0)]
    print(f"[{args.name}] {frames.shape[0]} px frames | swap_block={swap_block} | warmth=" + "/".join(f"{x:.0f}" for x in qs))


if __name__ == "__main__":
    main()
