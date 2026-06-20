"""Driveable streaming generator for CAUSAL FORCING — live prompt swaps via HARD CUT.

Port of streaming_longlive.py to Causal Forcing (thu-ml). Two differences from the
LongLive version:
  1. Loads the CF chunkwise model (no LoRA; plain `generator` key) with a rolling
     local-attention window (21) + sink (3) so it can stream indefinitely.
  2. Prompt switching uses a plain HARD CUT (re-encode the new prompt + flip every
     crossattn_cache["is_init"]=False, leave the self-attn KV cache alone). No recache.
     The transition is then gated by the rolling window flushing old frames (gradual).

The PromptBus is reused from streaming_longlive so an ASR/UI thread can steer between
chunks. Used by web_live_cf.py and cf_run_audio.py.
"""
import os, sys, time
import torch

CF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Causal-Forcing"))
FRAME_SEQ = 1560
CKPT = "checkpoints/chunkwise/causal_forcing.pt"
CFG = "configs/causal_forcing_dmd_chunkwise.yaml"

# reuse the thread-safe PromptBus
sys.path.insert(0, os.path.dirname(__file__))
from streaming_longlive import PromptBus  # noqa: E402
import math


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
    """Load the CF chunkwise pipeline with a rolling window. Returns the pipeline."""
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
    gen.load_state_dict(torch.load(CKPT, map_location="cpu")["generator"])
    pipe = CausalInferencePipeline(cfg, device=device, generator=gen).to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device); pipe.generator.to(device); pipe.vae.to(device)
    return pipe


class StreamingCF:
    def __init__(self, pipeline, seed=1, window=21, sink=3):
        self.p = pipeline
        self.device = next(pipeline.generator.parameters()).device
        self.nfpb = pipeline.num_frame_per_block
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
        cur = min(self.nfpb, self.total - self.cur_frame)
        noisy = self.noise[:, self.cur_frame:self.cur_frame + cur]
        cs = self.cur_frame * self.fsl
        for i, ts in enumerate(self.p.denoising_step_list):
            timestep = torch.ones([1, cur], device=self.device, dtype=torch.int64) * ts
            _, den = self.p.generator(noisy_image_or_video=noisy, conditional_dict=self.cond,
                                      timestep=timestep, kv_cache=self.p.kv_cache1,
                                      crossattn_cache=self.p.crossattn_cache, current_start=cs)
            if i < len(self.p.denoising_step_list) - 1:
                nt = self.p.denoising_step_list[i + 1]
                noisy = self.p.scheduler.add_noise(
                    den.flatten(0, 1), torch.randn_like(den.flatten(0, 1)),
                    nt * torch.ones([cur], device=self.device, dtype=torch.long)).unflatten(0, den.shape[:2])
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
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--out", default="/data/SPED/gino/out/cf_live/smoke.mp4")
    args = ap.parse_args()
    print("loading CF pipeline...")
    pipe = load_cf_pipeline()
    gen = StreamingCF(pipe, seed=0)
    P1 = "A fluffy golden retriever sprinting through a sunlit meadow of orange wildflowers, warm golden daylight, cinematic, photorealistic, 4k"
    P2 = "A fluffy golden retriever running across a deep snowy field on a frigid winter night, full moon, falling snow, frosted pine trees, deep blue moonlight, cinematic, 4k"
    gen.start(P1, total_frames=args.total)
    pipe.vae.model.clear_cache()
    frames = []
    n_chunks = args.total // gen.nfpb
    t0 = time.time()
    for c in range(n_chunks):
        if c == n_chunks // 3:
            t = time.time(); gen.hardcut(P2); print(f"  HARDCUT at chunk {c} ({(time.time()-t)*1e3:.0f}ms)")
        den = gen.step()
        frames.append(gen.decode_chunk(den))
    frames = np.concatenate(frames, axis=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=16, codec="libx264", macro_block_size=1)
    dt = time.time() - t0
    print(f"[smoke] {frames.shape[0]} frames in {dt:.1f}s = {frames.shape[0]/dt:.1f} FPS -> {args.out}")
