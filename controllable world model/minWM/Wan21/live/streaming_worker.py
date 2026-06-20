"""Persistent-cache streaming worker (Option B) for minWM Wan Action2V — the fast path.

Mirrors gino/audio_stream/streaming_longlive.py, adapted for minWM's camera/PRoPE path:
  - KV / PRoPE / cross-attn caches are initialized ONCE in start() and NEVER reset across
    chunks -> no per-chunk recache; chunk N+1 attends to the growing cache.
  - Each chunk feeds the new block's viewmats in one GLOBAL trajectory (cur_frame grows);
    the cache's local-window eviction (local_attn_size=20) keeps it bounded.
  - Decode only the new chunk via the VAE temporal cache (use_cache=True) -> ~4x less
    decode than re-decoding the whole window.
  - generator+text_encoder and the VAE may live on SEPARATE GPUs, so the server can overlap
    generation of chunk N+1 with decode of chunk N (gen_step on gpu0 || decode_step on gpu1).
  - Optional torch.compile on the generator.

Versus MinWMWorker (Option A): no `initial_latent` recache, no full-window re-decode.
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_WAN21 = os.path.dirname(_HERE)
for _p in (_HERE, _WAN21):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline
from camera import CameraController

_DEFAULT_CONFIG = os.path.join(_WAN21, "configs", "default_config.yaml")


class MinWMStreamingWorker:
    def __init__(self, config_path, checkpoint_path,
                 gen_device="cuda:0", vae_device=None, dtype=torch.bfloat16, compile=False):
        torch.set_grad_enabled(False)
        cfg = OmegaConf.merge(OmegaConf.load(_DEFAULT_CONFIG), OmegaConf.load(config_path))
        self.cfg = cfg
        self.dtype = dtype
        self.gen_device = torch.device(gen_device)
        self.vae_device = torch.device(vae_device) if vae_device else self.gen_device

        pipeline = CausalInferencePipeline(cfg, device=self.gen_device)
        if checkpoint_path:
            sd = torch.load(checkpoint_path, map_location="cpu")
            try:
                gen_sd = sd["generator_ema"]
            except (KeyError, TypeError):
                gen_sd = sd["generator"]
            try:
                pipeline.generator.load_state_dict(gen_sd)
            except RuntimeError:
                fixed = {(k.replace("model._fsdp_wrapped_module.", "model.", 1)
                          if k.startswith("model._fsdp_wrapped_module.") else k): v
                         for k, v in gen_sd.items()}
                pipeline.generator.load_state_dict(fixed, strict=False)

        pipeline = pipeline.to(dtype=dtype)
        pipeline.text_encoder.to(self.gen_device)
        pipeline.generator.to(self.gen_device)
        pipeline.vae.to(self.vae_device)
        self.p = pipeline

        self.nfpb = int(cfg.num_frame_per_block)        # 4 frames per chunk
        self.fsl = pipeline.frame_seq_length            # 1560
        self.local_attn = pipeline.local_attn_size      # 20
        self.C, self.H, self.Wl = 16, 60, 104
        self.fps = 16
        self.denoise = pipeline.denoising_step_list     # warped 4-step schedule
        self.camera = CameraController(device=self.gen_device, dtype=dtype)
        self.cur_frame = 0
        self.cond = None
        self.prompt = None
        self.compiled = False

        if compile:
            try:
                # The repo path contains a space ("controllable world model"); inductor's
                # C++ wrapper passes the torch lib dir to ld unquoted and the link fails
                # ("cannot find world"). The Python wrapper avoids that host-side link.
                import torch._inductor.config as _ic
                _ic.cpp_wrapper = False
                self.p.generator.model = torch.compile(self.p.generator.model, dynamic=True)
                self.compiled = True
                print("[StreamingWorker] torch.compile enabled on generator (cpp_wrapper off)")
            except Exception as e:
                print(f"[StreamingWorker] torch.compile failed, continuing uncompiled: {e}")
        print(f"[StreamingWorker] ready: gen={self.gen_device} vae={self.vae_device} "
              f"nfpb={self.nfpb} local_attn={self.local_attn}")

    @torch.no_grad()
    def start(self, prompt):
        """Begin a fresh stream: (re)init all caches once, encode the seed prompt."""
        dev = self.gen_device
        self.p._initialize_kv_cache(batch_size=1, dtype=self.dtype, device=dev)
        self.p._initialize_crossattn_cache(batch_size=1, dtype=self.dtype, device=dev)
        self.p._initialize_prope_kv_cache(batch_size=1, dtype=self.dtype, device=dev)
        self.cond = self.p.text_encoder(text_prompts=[prompt])
        self.prompt = prompt
        self.cur_frame = 0
        self.camera.reset_global()
        self.p.vae.model.clear_cache()      # start a fresh streaming decode

    @torch.no_grad()
    def set_prompt(self, prompt):
        """Hard-cut prompt swap: re-encode text K/V on the next chunk."""
        if prompt and prompt != self.prompt:
            self.cond = self.p.text_encoder(text_prompts=[prompt])
            for c in self.p.crossattn_cache:
                c["is_init"] = False
            self.prompt = prompt

    @torch.no_grad()
    def gen_step(self, camera_state):
        """Generate one chunk (nfpb latent frames); returns latents on gen_device.

        KV/PRoPE caches persist; viewmats continue the global trajectory."""
        cur = self.nfpb
        motions = self.camera.velocity_to_motions(camera_state, cur)
        self.camera.extend_global(motions)
        vm, ks = self.camera.global_new_tensors(cur)
        noisy = torch.randn([1, cur, self.C, self.H, self.Wl],
                            dtype=self.dtype, device=self.gen_device)
        cs = self.cur_frame * self.fsl

        timestep = None
        den = None
        for i, ts in enumerate(self.denoise):
            timestep = torch.ones([1, cur], device=self.gen_device, dtype=torch.int64) * ts
            _, den = self.p.generator(
                noisy_image_or_video=noisy, conditional_dict=self.cond, timestep=timestep,
                kv_cache=self.p.kv_cache1, crossattn_cache=self.p.crossattn_cache,
                current_start=cs, viewmats=vm, Ks=ks, prope_kv_cache=self.p.prope_kv_cache1)
            if i < len(self.denoise) - 1:
                nt = self.denoise[i + 1]
                noisy = self.p.scheduler.add_noise(
                    den.flatten(0, 1), torch.randn_like(den.flatten(0, 1)),
                    nt * torch.ones([cur], device=self.gen_device, dtype=torch.long)
                ).unflatten(0, den.shape[:2])

        # clean-context pass: write this chunk's clean K/V into the caches
        ctx_t = torch.ones_like(timestep) * self.p.args.context_noise
        self.p.generator(
            noisy_image_or_video=den, conditional_dict=self.cond, timestep=ctx_t,
            kv_cache=self.p.kv_cache1, crossattn_cache=self.p.crossattn_cache,
            current_start=cs, viewmats=vm, Ks=ks, prope_kv_cache=self.p.prope_kv_cache1)
        self.cur_frame += cur
        return den

    @torch.no_grad()
    def decode_step(self, latents):
        """Decode ONLY this chunk's new frames (VAE temporal cache persists)."""
        z = latents.to(self.vae_device)
        video = self.p.vae.decode_to_pixel(z, use_cache=True)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        v = (video[0].float() * 255.0).round().to(torch.uint8)
        return v.permute(0, 2, 3, 1).contiguous().cpu().numpy()


# ----------------------------- headless latency / rollout test -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="Wan21/configs/causal_forcing_dmd_camera.yaml")
    ap.add_argument("--checkpoint_path", default="./ckpts/Wan21/Action2V/dmd/model.pt")
    ap.add_argument("--prompt", default="A first-person walk down a sunlit forest path, tall green "
                                        "trees, dappled light, cinematic, photorealistic, 4k")
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--gen_device", default="cuda:0")
    ap.add_argument("--vae_device", default=None, help="separate GPU for VAE (e.g. cuda:1)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default="./outputs/live/streaming.mp4")
    args = ap.parse_args()

    t0 = time.time()
    w = MinWMStreamingWorker(args.config_path, args.checkpoint_path, gen_device=args.gen_device,
                             vae_device=args.vae_device, compile=args.compile)
    print(f"[test] loaded in {time.time() - t0:.1f}s")
    w.start(args.prompt)

    # scripted timeline: forward, turn left, look up, stop (each held a few chunks)
    def cam(i):
        if i < 4:   return {"forward": 1.0, "speed": 1.0}
        if i < 8:   return {"turn": -1.0, "speed": 1.0}
        if i < 12:  return {"pitch": 1.0, "speed": 1.0}
        if i < 16:  return {"forward": 1.0, "turn": 0.5, "speed": 1.0}
        return {"speed": 0.0}

    frames = []
    gen_ms, dec_ms = [], []
    for i in range(args.chunks):
        torch.cuda.synchronize(w.gen_device)
        t = time.time(); lat = w.gen_step(cam(i)); torch.cuda.synchronize(w.gen_device); gen_ms.append((time.time() - t) * 1e3)
        t = time.time(); px = w.decode_step(lat); dec_ms.append((time.time() - t) * 1e3)
        frames.append(px)
        print(f"[test] chunk {i:2d}: gen {gen_ms[-1]:6.0f}ms  decode {dec_ms[-1]:6.0f}ms  "
              f"frames {px.shape[0]}  cam {cam(i)}")

    allf = np.concatenate(frames, axis=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    from torchvision.io import write_video
    write_video(args.out, torch.from_numpy(allf), fps=w.fps)
    # skip the first chunk (cold/compile) in the averages
    g = np.mean(gen_ms[1:]) if len(gen_ms) > 1 else gen_ms[0]
    d = np.mean(dec_ms[1:]) if len(dec_ms) > 1 else dec_ms[0]
    print(f"\n[test] wrote {args.out} ({allf.shape[0]} frames)")
    print(f"[test] steady-state: gen {g:.0f}ms + decode {d:.0f}ms = {g + d:.0f}ms/chunk "
          f"(sequential); pipelined ~= max = {max(g, d):.0f}ms/chunk")


if __name__ == "__main__":
    main()
