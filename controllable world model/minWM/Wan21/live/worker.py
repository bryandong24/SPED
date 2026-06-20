"""Persistent, chunk-by-chunk streaming worker for minWM Wan Action2V (4-step DMD).

Loads the camera-conditioned pipeline + DMD checkpoint ONCE and keeps it resident, then
generates video chunk-by-chunk under live camera/action control. Continuation uses the
pipeline's native `initial_latent` path (which re-warms the KV/PRoPE cache over the
context window every call — "recache every chunk", Option A from the plan), so no upstream
edits are needed.

Frame bookkeeping (verified against Wan21/pipeline/causal_inference.py):
  - inference(noise[1,K,...], initial_latent[1,W,...], viewmats[1,W+K,...], Ks[1,W+K,...])
    returns (video, latents) with latents [1, W+K, 16, 60, 104]; the new frames are
    latents[:, W:].
  - W % 4 == 0, K % 4 == 0, and W + K <= local_attn_size (20).
  - VAE is 4x temporal: L latents -> 1 + 4*(L-1) pixel frames, so the K new latents are
    exactly the LAST 4*K pixel frames of the full-window decode -> video[:, -(4*K):].
"""
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


class MinWMWorker:
    def __init__(self, config_path, checkpoint_path,
                 device="cuda", dtype=torch.bfloat16):
        torch.set_grad_enabled(False)
        cfg = OmegaConf.merge(OmegaConf.load(_DEFAULT_CONFIG), OmegaConf.load(config_path))
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype

        # Expensive: build generator (CausalWanModel) + umt5-xxl text encoder + Wan VAE.
        pipeline = CausalInferencePipeline(cfg, device=self.device)

        # Load the DMD student checkpoint with the FSDP-prefix fallback
        # (mirrors wan_inference.py:106-123).
        if checkpoint_path:
            sd = torch.load(checkpoint_path, map_location="cpu")
            try:
                gen_sd = sd["generator_ema"]
            except (KeyError, TypeError):
                gen_sd = sd["generator"]
            try:
                pipeline.generator.load_state_dict(gen_sd)
            except RuntimeError:
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                pipeline.generator.load_state_dict(fixed, strict=False)

        pipeline = pipeline.to(dtype=dtype)
        pipeline.text_encoder.to(device=self.device)
        pipeline.generator.to(device=self.device)
        pipeline.vae.to(device=self.device)
        self.pipeline = pipeline

        self.nfpb = int(cfg.num_frame_per_block)                  # 4
        self.local_attn = int(cfg.model_kwargs.local_attn_size)   # 20
        self.C, self.H, self.Wl = 16, 60, 104                     # latent dims
        self.fps = 16
        self.latents = None                                       # [1, F, 16, 60, 104]
        self.prompt = None
        self.camera = CameraController(device=self.device, dtype=dtype)
        print(f"[MinWMWorker] ready: nfpb={self.nfpb} local_attn={self.local_attn}")

    def _noise(self, k, seed=None):
        g = torch.Generator("cpu").manual_seed(seed) if seed is not None else None
        return torch.randn([1, k, self.C, self.H, self.Wl], generator=g,
                           dtype=self.dtype).to(self.device)

    @staticmethod
    def _to_uint8(video):
        # video: [1, F, 3, H, W] in [0,1] -> uint8 [F, H, W, 3] numpy
        v = (video[0].float().clamp(0, 1) * 255.0).round().to(torch.uint8)
        return v.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    @torch.no_grad()
    def bootstrap(self, prompt, n=16, seed=None, profile=False):
        """Generate the first chunk fresh (no context) under a static camera."""
        assert n % self.nfpb == 0 and n <= self.local_attn, f"bad n={n}"
        noise = self._noise(n, seed)
        viewmats, Ks = self.camera.bootstrap_tensors(n)
        video, latents = self.pipeline.inference(
            noise=noise, text_prompts=[prompt], initial_latent=None,
            return_latents=True, viewmats=viewmats, Ks=Ks, profile=profile)
        self.pipeline.vae.model.clear_cache()
        self.latents = latents
        self.prompt = prompt
        return self._to_uint8(video)

    @torch.no_grad()
    def step(self, camera_state, prompt=None, K=4, W=12, seed=None, profile=False):
        """Generate the next K latent frames continuing from the last W; return new pixels."""
        assert self.latents is not None, "call bootstrap() first"
        assert K % self.nfpb == 0 and W % self.nfpb == 0, f"K/W must be x{self.nfpb}"
        assert W + K <= self.local_attn, f"W+K={W + K} > local_attn={self.local_attn}"
        assert self.latents.shape[1] >= W, "not enough history for window W"

        prompt = prompt or self.prompt
        self.prompt = prompt

        context = self.latents[:, -W:].contiguous()              # [1, W, 16, 60, 104]
        motions = self.camera.velocity_to_motions(camera_state, K)
        self.camera.extend(motions)
        viewmats, Ks = self.camera.window_tensors(W, K)          # [1, W+K, ...]
        noise = self._noise(K, seed)

        video, latents = self.pipeline.inference(
            noise=noise, text_prompts=[prompt], initial_latent=context,
            return_latents=True, viewmats=viewmats, Ks=Ks, profile=profile)
        self.pipeline.vae.model.clear_cache()

        new_latents = latents[:, W:]                             # [1, K, ...]
        self.latents = torch.cat([self.latents, new_latents], dim=1)
        return self._to_uint8(video[:, -(4 * K):])              # last 4*K pixel frames
