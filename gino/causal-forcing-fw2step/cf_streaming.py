"""Driveable streaming generator for CAUSAL FORCING++ — FRAME-WISE 2-STEP variant.

Lightweight sibling of gino/audio_stream/cf_streaming.py (which runs the chunk-wise
4-step model). This one points at the Causal Forcing++ frame-wise 2-step checkpoint and
config in the SHARED gino/Causal-Forcing repo — no repo clone, no duplicated model code.

Differences vs the chunk-wise version (and vs the poorly-done framewise-1step copy):
  1. CKPT/CFG point at the frame-wise 2-step model (num_frame_per_block=1,
     denoising_step_list=[1000,500], denoising_step_list_first_chunk=[1000,750,500,250]).
  2. Robust checkpoint loader: framewise-2step.pt is an FSDP-wrapped EMA dict
     (keys 'model._fsdp_wrapped_module.*'); we strip those prefixes — same fix the
     repo's inference.py uses (L69-82).
  3. step() applies the FIRST-CHUNK 4-step schedule on chunk 0 (the ASD first-frame
     trick that makes CF++ 1/2-step actually good). The batch path does this in
     pipeline/causal_inference.py (L224-228); the streaming path must too.

Prompt switching: HARD CUT (re-encode + reset cross-attn) or a smooth forward SLERP
ramp (ramp_to). A thread-safe PromptBus lets an ASR/UI thread steer between chunks.
Used by web_live_cf.py (live audio/text steering) and demo.py (click-to-generate UI).
"""
import os, sys, time, math, threading
import torch

# This file lives at gino/causal-forcing-fw2step/, so the shared Causal-Forcing repo
# is one level up. All heavy code (model, pipeline, wan, VAE) + configs + weights live there.
CF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Causal-Forcing"))
FRAME_SEQ = 1560
CKPT = "checkpoints/causal-forcing++/framewise-2step.pt"
CFG = "configs/causal_forcing_dmd_framewise_2step.yaml"


class PromptBus:
    """Thread-safe 'current prompt' with a version counter (debounce by version)."""
    def __init__(self, initial=""):
        self._lock = threading.Lock()
        self._prompt = initial
        self._version = 0

    def set(self, prompt):
        with self._lock:
            if prompt is not None and prompt != self._prompt:
                self._prompt = prompt
                self._version += 1

    def get(self):
        with self._lock:
            return self._prompt, self._version


def _minjerk(t):
    t = min(1.0, max(0.0, t))
    return t * t * t * (10 + t * (-15 + 6 * t))


def _slerp(a, b, s, eps=1e-6):
    """Per-token spherical interpolation between embedding tensors [.., L, C]."""
    a32, b32 = a.float(), b.float()
    na = a32.norm(dim=-1, keepdim=True).clamp_min(eps)
    nb = b32.norm(dim=-1, keepdim=True).clamp_min(eps)
    ua, ub = a32 / na, b32 / nb
    dot = (ua * ub).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(dot); so = torch.sin(omega)
    out = (torch.sin((1 - s) * omega) / so) * ua + (torch.sin(s * omega) / so) * ub
    out = out * ((1 - s) * na + s * nb)
    lerp = (1 - s) * a32 + s * b32
    return torch.where(so.abs() < 1e-3, lerp, out).to(a.dtype)


def load_cf_pipeline(window=21, sink=3, device=None):
    """Load the CF++ frame-wise 2-step pipeline with a rolling window. Returns the pipeline."""
    if CF_DIR not in sys.path:
        sys.path.insert(0, CF_DIR)
    os.chdir(CF_DIR)  # repo configs/weights are referenced by relative path
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    from utils.wan_wrapper import WanDiffusionWrapper

    device = device or torch.device("cuda")
    torch.set_grad_enabled(False)
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(CFG))
    gen = WanDiffusionWrapper(is_causal=True, local_attn_size=window, sink_size=sink)

    # framewise-2step.pt is an FSDP-wrapped EMA dict; pull the right sub-dict and strip
    # the wrapper prefixes so keys line up with WanDiffusionWrapper's 'model.*' (same
    # approach as the repo's inference.py --use_ema fallback).
    sd = torch.load(CKPT, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("generator_ema", "generator", "model", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    clean = {}
    for name, w in sd.items():
        name = name.replace("._fsdp_wrapped_module", "").replace("_fsdp_wrapped_module.", "")
        if name.startswith("module."):
            name = name[len("module."):]
        clean[name] = w
    missing, unexpected = gen.load_state_dict(clean, strict=False)
    print(f"[cf_streaming] framewise-2step loaded: {len(clean)} params | "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("  unexpected[:5]:", list(unexpected)[:5])
    if missing:
        print("  missing[:5]:", list(missing)[:5])

    pipe = CausalInferencePipeline(cfg, device=device, generator=gen).to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
    return pipe


class StreamingCF:
    def __init__(self, pipeline, seed=1, window=21, sink=3):
        self.p = pipeline
        self.device = next(pipeline.generator.parameters()).device
        self.nfpb = pipeline.num_frame_per_block          # 1 for frame-wise
        self.fsl = pipeline.frame_seq_length
        self.window = window
        self.sink = sink
        self.seed = seed
        self.H, self.W, self.C = 60, 104, 16

    def _set_window(self, W):
        for blk in self.p.generator.model.blocks:
            blk.self_attn.local_attn_size = W
            blk.self_attn.sink_size = self.sink
            blk.self_attn.max_attention_size = W * FRAME_SEQ

    @torch.no_grad()
    def start(self, prompt, total_frames):
        assert total_frames % self.nfpb == 0
        self.total = total_frames
        g = torch.Generator("cpu").manual_seed(self.seed)
        self.noise = torch.randn([1, total_frames, self.C, self.H, self.W],
                                 generator=g, dtype=torch.bfloat16).to(self.device)
        self._set_window(self.window)
        self.p.local_attn_size = self.window
        self.p.kv_cache1 = None
        self.p._initialize_kv_cache(1, dtype=self.noise.dtype, device=self.device)
        self.p._initialize_crossattn_cache(batch_size=1, dtype=self.noise.dtype, device=self.device)
        self.cond = self.p.text_encoder(text_prompts=[prompt])
        self.cur_prompt = prompt
        self.cur_frame = 0
        self.p.vae.model.clear_cache()
        self._ramp = None

    @torch.no_grad()
    def hardcut(self, new_prompt):
        """Swap prompt mid-stream via HARD CUT: re-encode + reset cross-attn only."""
        self.cond = self.p.text_encoder(text_prompts=[new_prompt])
        for c in self.p.crossattn_cache:
            c["is_init"] = False
        self.cur_prompt = new_prompt
        self._ramp = None

    @torch.no_grad()
    def ramp_to(self, new_prompt, k=6):
        """FORWARD conditioning ramp (no recache): over the next k chunks, smoothly
        SLERP the prompt embedding old->new so each new frame is generated AND cached
        under its own interpolated prompt -> continuous, self-consistent transition."""
        if not new_prompt or new_prompt == self.cur_prompt:
            return
        self._ramp = {"old": self.cond["prompt_embeds"],
                      "new": self.p.text_encoder(text_prompts=[new_prompt])["prompt_embeds"],
                      "i": 0, "k": max(1, int(k))}
        self.cur_prompt = new_prompt

    @torch.no_grad()
    def step(self):
        """Generate one chunk; return its clean latents [1, nfpb, C, H, W]."""
        # advance an in-progress forward ramp (interpolated conditioning this chunk)
        if self._ramp is not None:
            r = self._ramp; r["i"] += 1
            g = _minjerk(r["i"] / r["k"])
            self.cond = {"prompt_embeds": _slerp(r["old"], r["new"], g)}
            for c in self.p.crossattn_cache:
                c["is_init"] = False
            if r["i"] >= r["k"]:
                self.cond = {"prompt_embeds": r["new"]}
                self._ramp = None

        # FIRST-CHUNK schedule: chunk 0 runs the 4-step ASD schedule, every later chunk
        # runs the regular 2-step list. Mirrors pipeline/causal_inference.py L224-228.
        first = (self.cur_frame == 0)
        sched = (self.p.denoising_step_list_first_chunk
                 if first and getattr(self.p, "denoising_step_list_first_chunk", None) is not None
                 else self.p.denoising_step_list)

        cur = min(self.nfpb, self.total - self.cur_frame)
        noisy = self.noise[:, self.cur_frame:self.cur_frame + cur]
        cs = self.cur_frame * self.fsl
        for i, ts in enumerate(sched):
            timestep = torch.ones([1, cur], device=self.device, dtype=torch.int64) * ts
            _, den = self.p.generator(noisy_image_or_video=noisy, conditional_dict=self.cond,
                                      timestep=timestep, kv_cache=self.p.kv_cache1,
                                      crossattn_cache=self.p.crossattn_cache, current_start=cs)
            if i < len(sched) - 1:
                nt = sched[i + 1]
                noisy = self.p.scheduler.add_noise(
                    den.flatten(0, 1), torch.randn_like(den.flatten(0, 1)),
                    nt * torch.ones([cur], device=self.device, dtype=torch.long)).unflatten(0, den.shape[:2])
        # clean-context pass (write this chunk's K/V at context_noise timestep)
        ctx_t = torch.ones_like(timestep) * self.p.args.context_noise
        self.p.generator(noisy_image_or_video=den, conditional_dict=self.cond, timestep=ctx_t,
                         kv_cache=self.p.kv_cache1, crossattn_cache=self.p.crossattn_cache, current_start=cs)
        self.cur_frame += cur
        return den

    @torch.no_grad()
    def decode_chunk(self, den):
        """Streaming per-chunk decode -> uint8 frames [nf, H, W, 3]."""
        pix = self.p.vae.decode_to_pixel(den, use_cache=True)
        return ((pix * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)[0].permute(0, 2, 3, 1).cpu().numpy()


# ----------------------- headless smoke test -----------------------
if __name__ == "__main__":
    import argparse, numpy as np, imageio
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)   # ~4 latent frames / sec
    ap.add_argument("--out", default="/data/SPED/gino/causal-forcing-fw2step/out/smoke.mp4")
    args = ap.parse_args()
    print("loading CF++ framewise-2step pipeline...")
    pipe = load_cf_pipeline()
    gen = StreamingCF(pipe, seed=0)
    P1 = "A fluffy golden retriever sprinting through a sunlit meadow of orange wildflowers, warm golden daylight, cinematic, photorealistic, 4k"
    P2 = "A fluffy golden retriever running across a deep snowy field on a frigid winter night, full moon, falling snow, frosted pine trees, deep blue moonlight, cinematic, 4k"
    total = max(gen.nfpb, int(round(args.seconds * 4)))
    total -= total % gen.nfpb
    gen.start(P1, total_frames=total)
    pipe.vae.model.clear_cache()
    frames = []
    n_chunks = total // gen.nfpb
    t0 = time.time()
    for c in range(n_chunks):
        sched = (pipe.denoising_step_list_first_chunk if c == 0 and pipe.denoising_step_list_first_chunk is not None
                 else pipe.denoising_step_list)
        if c < 2 or c == n_chunks // 3:
            print(f"  chunk {c}: {len(sched)} denoising steps")
        if c == n_chunks // 3:
            t = time.time(); gen.hardcut(P2); print(f"  HARDCUT at chunk {c} ({(time.time()-t)*1e3:.0f}ms)")
        den = gen.step()
        frames.append(gen.decode_chunk(den))
    frames = np.concatenate(frames, axis=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=16, codec="libx264", macro_block_size=1)
    dt = time.time() - t0
    print(f"[smoke] {frames.shape[0]} frames in {dt:.1f}s = {frames.shape[0]/dt:.1f} FPS -> {args.out}")
