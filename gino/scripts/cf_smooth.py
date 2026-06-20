"""Causal Forcing — principled smooth-transition harness (Tier-1 of the approved plan).

Implements, as ablatable toggles, the cut-moment + artifact methods from
~/.claude/plans/modular-toasting-planet.md, all training-free, extending the
cf_recache / cf_hybrid machinery (CF = Self-Forcing fork; same Wan2.1 base):

  A1  min-jerk / raised-cosine conditioning ramp on the embedding geodesic (SLERP)
      -> replaces the C0 hard step in the conditioning with a C2-smooth A->B ramp.
  A2  critically-damped grow-window co-schedule (single bandwidth / horizon knob).
  A3  Kalman-gain KV handoff: keep KV_old (window re-encoded under A) AND KV_new
      (under B), blend (1-g)*old + g*new with a smooth gain g(t) instead of the hard
      recache replace -> smooth self-attn STATE handoff, and no block-evict double-cut.
  B1  CORAL / OT-Gaussian alignment of the recached K/V second-order statistics to the
      pre-swap (in-distribution) statistics -> kills the OOD recache artifacts.

Method registry (--method) gives the baselines for free:
  hardcut       : flip crossattn is_init, keep KV cache (no recache)         [baseline]
  recache       : plain LongLive recache under B                            [baseline]
  coral         : recache under B + B1 alignment
  kalman        : A3 KV handoff (+ A2 grow window)
  ramp          : A1 conditioning ramp only (hard KV recache under B)
  smooth_all    : A1 + A2 + A3 + B1 (the Tier-1 stack)

CPU self-check of the pure-math helpers (no GPU, no model):
  python cf_smooth.py --self_check
GPU run (when a GPU is free; NOT run during scaffolding):
  cd Causal-Forcing && PYTHONPATH=. ../Self-Forcing/.venv/bin/python ../scripts/cf_smooth.py \
      --example dog --method smooth_all --out_dir ../out/cf_smooth
"""
import argparse, math, os, time
import numpy as np
import torch

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
    # --- really rich, training-matched prompts (same subject both ends, different scene) ---
    "dog_rich": (
        "A fluffy golden retriever with a glossy honey-gold coat and a red leather collar bounds through a vast sunlit meadow of orange and yellow wildflowers, captured in crisp HDR 4K with razor-sharp fur detail. Warm golden afternoon light rakes across the lush green grass and pollen drifts glowing in the air. The low camera tracks fast alongside the dog at a full sprint, tongue out and ears flapping, petals scattering in its wake, rolling green hills and a bright blue sky beyond, cinematic and photorealistic.",
        "A fluffy golden retriever with a glossy honey-gold coat and a red leather collar runs across a deep snow-covered field on a frigid winter night, captured in crisp HDR 4K with razor-sharp fur and frost detail. A huge full moon and deep blue moonlight wash over the sparkling powder as thick snowflakes fall and the dog's breath fogs the air. The low camera tracks fast alongside the dog at a full sprint, paws kicking up snow, frosted pine trees and silver-lit drifts beyond under a starry sky, cinematic and photorealistic."),
    "car_rich": (
        "A glossy candy-red sports car with polished chrome wheels speeds along a sunny coastal highway hugging a turquoise ocean, filmed in ultra-realistic 4K with crisp mirror reflections on the paint. Bright midday sunlight glints off the hood and heat shimmer rises from the asphalt as palm trees and white guardrails blur past. A low cinematic chase camera follows close behind and slightly to the side, gulls wheeling over the cliffs, sparkling sea and pale blue sky beyond, photorealistic and razor sharp.",
        "A glossy candy-red sports car with polished chrome wheels glides down a rain-soaked neon city street at night, filmed in ultra-realistic 4K with vivid reflections on the wet paint. Vibrant pink and electric-blue neon signs smear across the puddled asphalt as rain streaks through the headlights beneath towering glass skyscrapers. A low cinematic chase camera follows close behind and slightly to the side, mist rising from the road, glowing signage and a dark rainy sky beyond, photorealistic and razor sharp."),
    "jungle_rich": (
        "A smooth low aerial drone shot skims fast over a lush emerald tropical jungle canopy in vivid 4K, soft mist curling between the treetops. Bright equatorial daylight pours over broad green leaves while silver waterfalls cascade down mossy cliffs into a winding river and flocks of birds lift off. The camera glides forward low and steady just above the canopy, sunlight flickering through the foliage, layered green ridgelines fading into a hazy horizon, cinematic and photorealistic.",
        "A smooth low aerial drone shot skims fast over vast rippling red desert sand dunes at golden hour in vivid 4K, fine sand streaming off the wind-sculpted crests. Warm low sunlight rakes long shadows across the burnt-orange ridges with scattered dry rock and rippled sand below under a heat-hazed sky. The camera glides forward low and steady just above the dunes, grains glittering in the light, endless sculpted dunes fading into a hazy horizon, cinematic and photorealistic."),
}

# --- method registry: which toggles each named method turns on ---
METHODS = {
    "hardcut":    dict(recache=False, ramp=False, kalman=False, coral=False),
    "recache":    dict(recache=True,  ramp=False, kalman=False, coral=False),
    "coral":      dict(recache=True,  ramp=False, kalman=False, coral=True),
    "kalman":     dict(recache=True,  ramp=False, kalman=True,  coral=False),
    "ramp":       dict(recache=True,  ramp=True,  kalman=False, coral=False),
    "smooth_all": dict(recache=True,  ramp=True,  kalman=True,  coral=True),
    # pure FORWARD conditioning ramp, NO retroactive recache: each new frame is
    # generated AND cached under its own old->new interpolated prompt, so the cache
    # stays self-consistent (no phantom rewritten history). Trades latency (the ramp
    # horizon = transition_chunks) for continuity.
    "fwd_ramp":   dict(recache=False, ramp=True,  kalman=False, coral=False),
}


# =========================================================================
# Pure-math helpers (CPU-checkable, model-free)
# =========================================================================
def minjerk(tau):
    """5th-order min-jerk: s(0)=0,s(1)=1, s'=s''=0 at both ends (C2)."""
    t = float(min(1.0, max(0.0, tau)))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def raised_cosine(tau):
    t = float(min(1.0, max(0.0, tau)))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def linear(tau):
    return float(min(1.0, max(0.0, tau)))


def critically_damped_schedule(n, settle_frac=0.9):
    """Discrete unit-step response of a critically-damped 2nd-order system,
    sampled at n points, normalised to reach ~1 by the end. ζ=1 -> no overshoot.
    Returns a length-n list of gains in [0,1]. Used to co-schedule A1/A3/A2."""
    if n <= 1:
        return [1.0]
    # choose omega so the response has settled (1-(1+wt)e^{-wt} ~ settle_frac) at t=1
    wn = 6.0  # ~critical settle within the horizon; horizon mapped to [0,1]
    out = []
    for i in range(n):
        t = (i + 1) / n
        out.append(1.0 - (1.0 + wn * t) * math.exp(-wn * t))
    # renormalise so the last value is exactly 1
    last = out[-1] if out[-1] > 1e-6 else 1.0
    return [min(1.0, v / last) for v in out]


def gain_at(kind, c, T):
    """Gain in [0,1] for transition chunk c (0-indexed) of T chunks."""
    if kind == "critically_damped":
        return critically_damped_schedule(T)[min(c, T - 1)]
    tau = (c + 1) / float(T)
    return {"minjerk": minjerk, "raised_cosine": raised_cosine, "linear": linear}[kind](tau)


def slerp(a, b, s, eps=1e-6):
    """Per-token spherical interpolation between embedding tensors a,b: [.., L, C].
    Falls back to lerp where the tokens are near-parallel/anti-parallel or tiny."""
    a32, b32 = a.float(), b.float()
    na = a32.norm(dim=-1, keepdim=True).clamp_min(eps)
    nb = b32.norm(dim=-1, keepdim=True).clamp_min(eps)
    ua, ub = a32 / na, b32 / nb
    dot = (ua * ub).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    # direction via slerp, magnitude via lerp (keeps embedding scale sane)
    w_a = torch.sin((1 - s) * omega) / so
    w_b = torch.sin(s * omega) / so
    dir_interp = w_a * ua + w_b * ub
    mag = (1 - s) * na + s * nb
    out = dir_interp * mag
    lerp = (1 - s) * a32 + s * b32
    use_lerp = (so.abs() < 1e-3)
    out = torch.where(use_lerp, lerp, out)
    return out.to(a.dtype)


def lerp_emb(a, b, s):
    return ((1 - s) * a.float() + s * b.float()).to(a.dtype)


def coral_align(new, old, eps=1e-3):
    """OT-Gaussian / CORAL whiten-recolor: map `new` so its per-head mean+cov match
    `old`'s. new,old: [1, ntok, H, D]. Returns aligned copy of `new`.
    x' = (x - mu_new) @ Sigma_new^{-1/2} @ Sigma_old^{1/2} + mu_old   (per head)."""
    xn = new[0].float(); xo = old[0].float()            # [ntok, H, D]
    ntok, H, D = xn.shape
    mun = xn.mean(0); muo = xo.mean(0)                  # [H, D]
    cn = xn - mun; co = xo - muo
    Sn = torch.einsum('thd,the->hde', cn, cn) / ntok    # [H, D, D]
    So = torch.einsum('thd,the->hde', co, co) / ntok
    I = torch.eye(D, device=xn.device, dtype=xn.dtype).expand(H, D, D)
    Sn = Sn + eps * I; So = So + eps * I
    # symmetric eig (batched over heads)
    ln, Un = torch.linalg.eigh(Sn)
    lo, Uo = torch.linalg.eigh(So)
    Wn = Un @ torch.diag_embed(1.0 / torch.sqrt(ln.clamp_min(eps))) @ Un.transpose(-1, -2)   # Sn^{-1/2}
    Co = Uo @ torch.diag_embed(torch.sqrt(lo.clamp_min(eps))) @ Uo.transpose(-1, -2)         # So^{1/2}
    A = Wn @ Co                                          # [H, D, D]
    xa = torch.einsum('thd,hde->the', cn, A) + muo       # [ntok, H, D]
    return xa.unsqueeze(0).to(new.dtype)


# =========================================================================
# Model / cache helpers (GPU; not exercised by --self_check)
# =========================================================================
def set_window(model, W):
    for blk in model.blocks:
        blk.self_attn.local_attn_size = W
        blk.self_attn.max_attention_size = W * FRAME_SEQ


def build_pipeline(device, window, sink):
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    from utils.wan_wrapper import WanDiffusionWrapper
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(CFG))
    gen = WanDiffusionWrapper(is_causal=True, local_attn_size=window, sink_size=sink)
    gen.load_state_dict(torch.load(CKPT, map_location="cpu")["generator"])
    pipe = CausalInferencePipeline(cfg, device=device, generator=gen).to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
    return pipe


def recache_fill(pipe, prefix, cond_emb, W, device):
    """Re-init the self-attn KV cache and refill it by re-encoding the last W clean
    frames of `prefix` under `cond_emb` at the clean-context timestep (relative pos).
    Returns pos_offset (= real index of the first replayed frame)."""
    K = prefix.shape[1]
    s0 = max(0, K - W)
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


def snapshot_window(pipe, ntok):
    """Clone the first ntok tokens of every block's self-attn K/V."""
    return [(pipe.kv_cache1[b]["k"][:, :ntok].clone(),
             pipe.kv_cache1[b]["v"][:, :ntok].clone()) for b in range(len(pipe.kv_cache1))]


def write_window(pipe, kv_list, ntok):
    for b, (k, v) in enumerate(kv_list):
        pipe.kv_cache1[b]["k"][:, :ntok] = k
        pipe.kv_cache1[b]["v"][:, :ntok] = v


def blend_window(pipe, old, new, g, ntok):
    """Write (1-g)*old + g*new into the first ntok token slots of every block."""
    for b in range(len(pipe.kv_cache1)):
        ok, ov = old[b]; nk, nv = new[b]
        pipe.kv_cache1[b]["k"][:, :ntok] = (1 - g) * ok + g * nk
        pipe.kv_cache1[b]["v"][:, :ntok] = (1 - g) * ov + g * nv


# =========================================================================
# Generation with the smooth-transition controls
# =========================================================================
@torch.no_grad()
def generate(pipe, noise, p1, p2, device, *, swap_frame, transition_chunks,
             recache_W, window, sink, post_window, grow_to,
             flags, gain_kind="minjerk", use_slerp=True):
    nb = pipe.num_frame_per_block
    total = noise.shape[1]
    # A3 requires the blended window to survive (no eviction) for the whole transition.
    if flags["kalman"]:
        need = recache_W + nb * transition_chunks
        assert need <= window, (f"kalman needs recache_W + nfpb*transition_chunks <= window "
                                 f"({recache_W}+{nb}*{transition_chunks}={need} > {window}); "
                                 f"lower transition_chunks/recache_W or raise window.")
    set_window(pipe.generator.model, window)
    pipe.local_attn_size = window
    pipe.kv_cache1 = None
    pipe._initialize_kv_cache(1, noise.dtype, device)
    pipe._initialize_crossattn_cache(1, noise.dtype, device)
    out = torch.zeros_like(noise)

    cond = p1; start = 0; swapped_at = None; pos_offset = 0
    KV_old = KV_new = None; ntok = recache_W * FRAME_SEQ

    while start < total:
        cur = min(nb, total - start)

        # ---- swap setup (once) ----
        if cond is p1 and start + cur > swap_frame and start > 0:
            swapped_at = start
            if flags["recache"]:
                # KV_old: window re-encoded under A (in-distribution reference)
                if flags["kalman"] or flags["coral"]:
                    recache_fill(pipe, out[:, :start], p1, recache_W, device)
                    KV_old = snapshot_window(pipe, ntok)
                # KV_new: window re-encoded under B
                pos_offset = recache_fill(pipe, out[:, :start], p2, recache_W, device)
                if flags["coral"]:                       # B1: align new->old statistics
                    for b in range(len(pipe.kv_cache1)):
                        pipe.kv_cache1[b]["k"][:, :ntok] = coral_align(pipe.kv_cache1[b]["k"][:, :ntok], KV_old[b][0])
                        pipe.kv_cache1[b]["v"][:, :ntok] = coral_align(pipe.kv_cache1[b]["v"][:, :ntok], KV_old[b][1])
                if flags["kalman"]:                      # A3: keep both for per-chunk blend
                    KV_new = snapshot_window(pipe, ntok)
                    blend_window(pipe, KV_old, KV_new, gain_at(gain_kind, 0, transition_chunks), ntok)
            else:
                # hard-cut: keep the KV cache, only the prompt changes (below)
                pass
            if not flags["ramp"]:                        # immediate prompt swap
                cond = p2
                for c in pipe.crossattn_cache: c["is_init"] = False

        # ---- per-chunk transition controls ----
        if swapped_at is not None:
            since = start - swapped_at
            c_idx = since // nb
            in_transition = c_idx < transition_chunks
            g = gain_at(gain_kind, min(c_idx, transition_chunks - 1), transition_chunks)
            # A1: conditioning ramp along the embedding geodesic
            if flags["ramp"]:
                if in_transition:
                    interp = (slerp if use_slerp else lerp_emb)(p1, p2, g)
                    cond = interp
                    for cc in pipe.crossattn_cache: cc["is_init"] = False
                elif cond is not p2:
                    cond = p2
                    for cc in pipe.crossattn_cache: cc["is_init"] = False
            # A3: re-blend the (non-evicted) window slots at the current gain
            if flags["kalman"] and in_transition and KV_old is not None:
                blend_window(pipe, KV_old, KV_new, g, ntok)
            # A2: critically-damped grow-window, anchored at the swap
            if grow_to > 0:
                base = post_window if post_window > 0 else nb
                W = min(grow_to, max(base, since))
                set_window(pipe.generator.model, W)

        # ---- denoise this chunk (4-step) ----
        noisy = noise[:, start:start + cur]
        cs = (start - pos_offset) * FRAME_SEQ
        cond_dict = cond if isinstance(cond, dict) else {"prompt_embeds": cond}
        for i, ts in enumerate(pipe.denoising_step_list):
            timestep = torch.ones([1, cur], device=device, dtype=torch.int64) * ts
            _, den = pipe.generator(noisy_image_or_video=noisy, conditional_dict=cond_dict,
                                    timestep=timestep, kv_cache=pipe.kv_cache1,
                                    crossattn_cache=pipe.crossattn_cache, current_start=cs)
            if i < len(pipe.denoising_step_list) - 1:
                nt = pipe.denoising_step_list[i + 1]
                noisy = pipe.scheduler.add_noise(den.flatten(0, 1), torch.randn_like(den.flatten(0, 1)),
                                                 nt * torch.ones([cur], device=device, dtype=torch.long)).unflatten(0, den.shape[:2])
        out[:, start:start + cur] = den
        ctx_t = torch.ones_like(timestep) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=den, conditional_dict=cond_dict, timestep=ctx_t,
                       kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache, current_start=cs)
        start += cur

    try:
        pipe.vae.model.clear_cache()
    except Exception:
        pass
    from einops import rearrange
    video = (pipe.vae.decode_to_pixel(out, use_cache=False) * 0.5 + 0.5).clamp(0, 1)
    return (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()


# =========================================================================
# CPU self-check (no GPU, no model) — validates the pure-math helpers
# =========================================================================
def self_check():
    print("== schedules (gain per chunk, T=6) ==")
    for k in ("minjerk", "raised_cosine", "linear", "critically_damped"):
        gs = [round(gain_at(k, c, 6), 3) for c in range(6)]
        ok = abs(gs[0]) < 0.5 and abs(gs[-1] - 1.0) < 1e-6 and all(gs[i] <= gs[i+1] + 1e-6 for i in range(5))
        # min-jerk should start very flat (C2): first step small
        flat = (k != "minjerk") or gs[0] < 0.06
        print(f"  {k:18s} {gs}  monotonic&endpoints={ok} flatstart={flat}")
        assert ok and flat, f"schedule {k} failed"

    print("== slerp vs lerp (random embeddings) ==")
    torch.manual_seed(0)
    a = torch.randn(1, 4, 4096); b = torch.randn(1, 4, 4096)
    e0 = slerp(a, b, 0.0); e1 = slerp(a, b, 1.0); em = slerp(a, b, 0.5)
    assert torch.allclose(e0, a, atol=1e-3), "slerp(s=0) != a"
    assert torch.allclose(e1, b, atol=1e-3), "slerp(s=1) != b"
    # midpoint norm between the endpoints' norms (magnitude lerp)
    nm = em.norm().item(); na = a.norm().item(); nb = b.norm().item()
    print(f"  norms a={na:.1f} mid={nm:.1f} b={nb:.1f}  endpoints exact=OK")
    assert min(na, nb) - 1 <= nm <= max(na, nb) + 1

    print("== CORAL alignment (new -> old stats) ==")
    H, D, ntok = 12, 128, 2000
    old = torch.randn(1, ntok, H, D) * 2.5 + 1.0           # target distribution
    new = torch.randn(1, ntok, H, D) * 0.7 - 0.5           # source distribution
    aligned = coral_align(new, old)
    # per-head mean/std should now match old's much better than new's did
    def stats(x):
        return x[0].mean(0).abs().mean().item(), x[0].std(0).mean().item()
    mo, so = stats(old); mn, sn = stats(new); ma, sa = stats(aligned)
    print(f"  old(mean|abs|,std)=({mo:.2f},{so:.2f})  new=({mn:.2f},{sn:.2f})  aligned=({ma:.2f},{sa:.2f})")
    assert abs(sa - so) < 0.15, "CORAL std not matched"
    assert abs(ma - mo) < 0.15, "CORAL mean not matched"

    print("== blend gain bounds ==")
    assert gain_at("minjerk", 0, 4) >= 0 and gain_at("minjerk", 3, 4) == 1.0
    print("\nALL SELF-CHECKS PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_check", action="store_true", help="CPU-only math validation (no GPU/model)")
    ap.add_argument("--example", choices=list(PROMPTS), default="dog")
    ap.add_argument("--method", choices=list(METHODS) + ["all"], default="smooth_all")
    ap.add_argument("--total", type=int, default=40)
    ap.add_argument("--swap_frame", type=int, default=12)
    ap.add_argument("--transition_chunks", type=int, default=4)
    ap.add_argument("--recache_W", type=int, default=9)
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--sink", type=int, default=3)
    ap.add_argument("--post_window", type=int, default=3)
    ap.add_argument("--grow_to", type=int, default=15)
    ap.add_argument("--gain_kind", default="minjerk",
                    choices=["minjerk", "raised_cosine", "linear", "critically_damped"])
    ap.add_argument("--lerp", action="store_true", help="use linear emb interp instead of SLERP")
    ap.add_argument("--out_dir", default="../out/cf_smooth")
    ap.add_argument("--tag", default="", help="suffix appended to output filename (avoid clobber across horizons)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.self_check:
        self_check(); return

    import imageio
    device = torch.device("cuda"); torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)
    pipe = build_pipeline(device, args.window, args.sink)
    p1t, p2t = PROMPTS[args.example]
    p1 = pipe.text_encoder(text_prompts=[p1t])["prompt_embeds"]
    p2 = pipe.text_encoder(text_prompts=[p2t])["prompt_embeds"]
    g = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn([1, args.total, 16, 60, 104], generator=g, dtype=torch.bfloat16).to(device)

    methods = list(METHODS) if args.method == "all" else [args.method]
    for m in methods:
        t0 = time.time()
        frames = generate(pipe, noise, p1, p2, device, swap_frame=args.swap_frame,
                          transition_chunks=args.transition_chunks, recache_W=args.recache_W,
                          window=args.window, sink=args.sink, post_window=args.post_window,
                          grow_to=args.grow_to, flags=METHODS[m],
                          gain_kind=args.gain_kind, use_slerp=not args.lerp)
        tag = f"{args.example}_{m}" + (f"_{args.tag}" if args.tag else "")
        imageio.mimwrite(os.path.join(args.out_dir, f"{tag}.mp4"), frames, fps=16, codec="libx264", macro_block_size=1)
        w = frames.reshape(frames.shape[0], -1, 3).mean(axis=1); warm = w[:, 0] - w[:, 2]
        # smoothness proxy: mean |2nd difference| of per-frame luma (lower = smoother)
        luma = frames.reshape(frames.shape[0], -1).mean(axis=1)
        jerk = float(np.abs(np.diff(luma, n=2)).mean()) if len(luma) > 2 else 0.0
        qs = [round(float(warm[int(p * (len(warm) - 1))]), 1) for p in (0, .25, .5, .75, 1.0)]
        print(f"[{tag}] {frames.shape[0]}f {time.time()-t0:.1f}s warmth@0/25/50/75/100={qs} jerk(luma)={jerk:.2f}", flush=True)
        del frames
        import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"DONE {args.example} ({args.method})", flush=True)


if __name__ == "__main__":
    main()
