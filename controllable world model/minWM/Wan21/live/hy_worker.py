"""Persistent streaming worker for HunyuanVideo-1.5 (HY) Action2V — camera-controlled I2V.

Lifts HY15/hy15_inference.py::run_inference_rollout into start()/gen_step()/decode_step()
so the live demo can stream chunk-by-chunk under live camera control. Differences from the
Wan worker:
  - I2V: a stream starts from an IMAGE + caption (not text alone). The image lands in global
    latent-frame 0; later chunks see it only through the KV cache.
  - ~8B model (HunyuanVideo-1.5, 54 double blocks) + Qwen2.5-VL / SigLIP / byT5 encoders.
  - KV cache: text K/V cached once; per chunk the denoised vision K/V are APPENDED. The
    reference never evicts (fixed 77-frame clip); for an open-ended live stream we cap the
    vision cache while pinning the first `sink_vision_frames` latent frame(s) as an anchor.
  - Incremental VAE decode via a persistent feat_cache (HunyuanVideo VAE is causal Conv3d).

Reuses hy15_inference's helpers (load_model_prope, prepare_sample_data, the encoders) and
the existing CameraController (with HY's intrinsics).
"""
import os
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_WAN21 = os.path.dirname(_HERE)
_MINWM = os.path.dirname(_WAN21)
for _p in (_HERE, os.path.join(_MINWM, "HY15"), os.path.join(_MINWM, "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.distributed as dist

from camera import CameraController

# HY intrinsics (hy15_inference.make_camera_tensors defaults)
HY_FX, HY_FY, HY_CX, HY_CY = 0.5050505, 0.89786756, 0.5, 0.5


def _setup_hy_dist():
    """gloo single-process group + infer_state, as hy15_inference.setup_dist('ar_rollout')."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 2000))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://", world_size=1, rank=0)
    from hyvideo.commons.infer_state import initialize_infer_state
    # fp8 GEMM and SageAttention are wired in HY15 but off by default. Expose them as
    # env toggles so the live (interactive) worker can use them without touching the
    # batch path:  HY_FP8=1  -> fp8-per-block GEMM on the DiT (~1.3-1.6x gen);
    #              HY_SAGE=1  -> int8 SageAttention (helps the small-q/large-KV stream shape).
    use_fp8 = os.getenv("HY_FP8", "0") == "1"
    use_sage = os.getenv("HY_SAGE", "0") == "1"
    initialize_infer_state(SimpleNamespace(
        sage_blocks_range=("0-53" if use_sage else "0-0"), use_sageattn=use_sage,
        enable_torch_compile=False,
        use_fp8_gemm=use_fp8, quant_type="fp8-per-block",
        include_patterns="double_blocks", use_vae_parallel=False))
    if use_fp8 or use_sage:
        print(f"[hy_worker] infer_state: fp8_gemm={use_fp8} sageattn={use_sage}")


class HYStreamingWorker:
    def __init__(self, transformer_dir="./ckpts/HY15/Action2V/dmd",
                 model_path="./ckpts/HunyuanVideo-1.5",
                 gen_device="cuda:0", vae_device=None, dtype=torch.bfloat16,
                 height=480, width=832, chunk_latent_frames=4, num_steps=4,
                 shift=5.0, stabilization_level=1, max_vision_frames=24,
                 sink_vision_frames=1):
        torch.set_grad_enabled(False)
        _setup_hy_dist()
        self.gen_device = torch.device(gen_device)
        self.vae_device = torch.device(vae_device) if vae_device else self.gen_device
        self.dtype = dtype
        self.h, self.w = height, width
        self.nfpb = chunk_latent_frames
        self.num_steps = num_steps
        self.shift = shift
        self.stab = stabilization_level
        self.max_vis = int(max_vision_frames) if max_vision_frames else None
        self.sink_vis = max(0, int(sink_vision_frames or 0))
        if self.max_vis:
            self.sink_vis = min(self.sink_vis, max(0, self.max_vis - 1))
        self.C = 32
        self.lat_h, self.lat_w = height // 16, width // 16   # 30, 52
        self.tokens_per_frame = self.lat_h * self.lat_w
        self.fps = 16
        self.compiled = False

        import hy15_inference as hy
        self.hy = hy
        from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
        self._Sched = FlowMatchDiscreteScheduler

        # ── load once ──
        from trainer.models.hyvideo.vae.hunyuanvideo_15_vae_w_cache import AutoencoderKLConv3D
        self.vae = AutoencoderKLConv3D.from_pretrained(os.path.join(model_path, "vae"),
                                                       torch_dtype=torch.float16)
        from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
        self.text_encoder, _ = HunyuanVideo_1_5_Pipeline._load_text_encoders(model_path, device="cpu")
        self.vision_encoder = HunyuanVideo_1_5_Pipeline._load_vision_encoder(model_path, device="cpu")
        byt5_kwargs, _ = HunyuanVideo_1_5_Pipeline._load_byt5(model_path, True, 256, device="cpu")
        self.byt5_model = byt5_kwargs["byt5_model"]
        self.byt5_tokenizer = byt5_kwargs["byt5_tokenizer"]
        self.model = hy.load_model_prope(transformer_dir).to(self.gen_device, dtype=dtype).eval()
        self.vae = self.vae.to(self.vae_device)
        self.num_layers = len(self.model.double_blocks)
        self.scaling = self.vae.config.scaling_factor
        self.shift_factor = getattr(self.vae.config, "shift_factor", None)

        self.camera = CameraController(fx=HY_FX, fy=HY_FY, cx=HY_CX, cy=HY_CY,
                                       device=self.gen_device, dtype=dtype)
        # streaming state
        self.kv_cache = None
        self.global_start = 0
        self.image_cond = None
        self.prompt = None
        self._dec_frame = 0
        # live prompt steering: text conditioning persists so set_prompt() can recache it;
        # _ramp drives a LERP from the old caption embedding to the new one over k chunks.
        self.ramp_chunks = int(os.getenv("HY_RAMP_CHUNKS", "8"))
        self.cur_prompt = None
        self.prompt_embed = None         # current (possibly interpolated) text embeds [1,L,4096]
        self.prompt_mask = None
        self.vision_states = None        # seed-image SigLIP features; fixed for the stream
        self.extra = None                # byt5 glyph states/mask; follows the caption
        self._ramp = None
        print(f"[HYStreamingWorker] ready: gen={self.gen_device} vae={self.vae_device} "
              f"layers={self.num_layers} cap={self.max_vis} sink={self.sink_vis} "
              f"latent={self.C}x{self.lat_h}x{self.lat_w}")

    # ------------------------------------------------------------------ start
    @torch.no_grad()
    def start(self, image_path, caption):
        """Encode the seed image+caption once and prime the text KV cache."""
        example = {"image": image_path, "caption": caption}
        # encode_byt5 does NOT self-offload, so move encoders to GPU first (as the
        # reference main() does), then back to CPU afterward.
        self.text_encoder.to(self.gen_device)
        self.vision_encoder.to(self.gen_device)
        if self.byt5_model is not None:
            self.byt5_model.to(self.gen_device)
        data = self.hy.prepare_sample_data(
            self.vae, self.text_encoder, self.vision_encoder,
            self.byt5_model, self.byt5_tokenizer, example,
            self.h, self.w, video_length=77, device=self.gen_device)
        self.text_encoder.cpu(); self.vision_encoder.cpu()
        if self.byt5_model is not None:
            self.byt5_model.cpu()
        # prepare_sample_data moved the VAE to GPU then back to CPU; restore it for decode.
        self.vae = self.vae.to(self.vae_device)
        torch.cuda.empty_cache()

        dev, dt = self.gen_device, self.dtype
        self.image_cond = data["image_cond"].to(dev, dt)            # [1,32,1,30,52]
        # Persist the text/vision/byt5 conditioning so set_prompt() can recompute the
        # text K/V mid-stream. vision_states are the *seed image* features — fixed for the
        # whole stream; only the caption (prompt_embed / byt5) changes when steered.
        self.prompt_embed = data["prompt_embeds"].to(dev, dt)
        self.prompt_mask = data["prompt_mask"].to(dev, dt)
        self.vision_states = data["vision_states"].to(dev, dt)
        self.extra = {"byt5_text_states": data["byt5_text_states"].to(dev, dt),
                      "byt5_text_mask": data["byt5_text_mask"].to(dev, dt)}

        # Prime a LOCAL cache fully (text K/V populated) before publishing it. A gen_step
        # from a prior session (Stop->Start) must never observe a half-initialized cache with
        # k_txt=None — that crashes the vision attention's torch.cat([k_txt, ...]). Publishing
        # only the finished cache makes the swap atomic.
        kv = [{"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None}
              for _ in range(self.num_layers)]
        self._recache_text(self.prompt_embed, self.prompt_mask, cache=kv)

        self.kv_cache = kv                       # atomic publish of a fully-primed cache
        self.global_start = 0
        self.camera.reset_global()
        self.vae.clear_cache()
        self._dec_frame = 0
        self.prompt = caption
        self.cur_prompt = caption
        self._ramp = None

    @torch.no_grad()
    def _recache_text(self, prompt_embed, prompt_mask, cache=None):
        """(Re)compute the text K/V in every layer from the given (interpolated) embeds and
        write k_txt/v_txt into `cache` (defaults to the live self.kv_cache). The accumulated
        vision K/V (the generated history) is left untouched — only the caption conditioning
        changes. vision_states (seed image) and self.extra (byt5) are reused as-is. This is the
        HY analogue of resetting the cross-attn cache in Causal-Forcing, except HY bakes the
        prompt into k_txt/v_txt. Validates the pass produced real tensors and only then commits,
        so a degenerate prompt/mask can never leave k_txt=None in a live cache."""
        cache = self.kv_cache if cache is None else cache
        dev, dt = self.gen_device, self.dtype
        t_txt = torch.tensor([0]).to(dev, dt)
        with torch.autocast("cuda", dtype=dt):
            fresh = self.model(
                bi_inference=False, ar_txt_inference=True, ar_vision_inference=False,
                timestep_txt=t_txt, text_states=prompt_embed, encoder_attention_mask=prompt_mask,
                vision_states=self.vision_states, mask_type="i2v", extra_kwargs=self.extra,
                kv_cache=cache, cache_txt=True)
        if fresh is None or any(fresh[j].get("k_txt") is None for j in range(self.num_layers)):
            raise RuntimeError("text KV recache produced None k_txt (degenerate prompt/mask)")
        for j in range(self.num_layers):
            cache[j]["k_txt"] = fresh[j]["k_txt"]
            cache[j]["v_txt"] = fresh[j]["v_txt"]

    @staticmethod
    def _pad_to(x, length):
        """Right-pad/truncate a [1, L, ...] tensor to length L along dim=1 (zeros)."""
        if x.shape[1] == length:
            return x
        if x.shape[1] > length:
            return x[:, :length]
        pad = torch.zeros(x.shape[0], length - x.shape[1], *x.shape[2:],
                          dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=1)

    def _trim_vision_cache(self, cache_entry):
        """Bound HY vision KV while preserving the seed frame as an attention sink."""
        if not self.max_vis:
            return
        cap = int(self.max_vis) * self.tokens_per_frame
        total = cache_entry["k_vision"].shape[2]
        if total <= cap:
            return

        sink = min(self.sink_vis * self.tokens_per_frame, cap)
        recent = cap - sink
        if sink <= 0:
            cache_entry["k_vision"] = cache_entry["k_vision"][:, :, -cap:].contiguous()
            cache_entry["v_vision"] = cache_entry["v_vision"][:, :, -cap:].contiguous()
            return
        if recent <= 0:
            cache_entry["k_vision"] = cache_entry["k_vision"][:, :, :sink].contiguous()
            cache_entry["v_vision"] = cache_entry["v_vision"][:, :, :sink].contiguous()
            return

        k_sink = cache_entry["k_vision"][:, :, :sink]
        v_sink = cache_entry["v_vision"][:, :, :sink]
        k_recent = cache_entry["k_vision"][:, :, -recent:]
        v_recent = cache_entry["v_vision"][:, :, -recent:]
        cache_entry["k_vision"] = torch.cat([k_sink, k_recent], dim=2).contiguous()
        cache_entry["v_vision"] = torch.cat([v_sink, v_recent], dim=2).contiguous()

    # ------------------------------------------------------------- generate
    @torch.no_grad()
    def gen_step(self, camera_state):
        """Generate one chunk (nfpb latent frames) under the live camera; returns latents."""
        self._advance_ramp()                  # apply one step of any in-progress prompt ramp
        cur = self.nfpb
        motions = self.camera.velocity_to_motions(camera_state, cur)
        self.camera.extend_global(motions)
        vm, ks = self.camera.global_new_tensors(cur)              # [1,cur,4,4]/[1,cur,3,3]

        cond_chunk = torch.zeros(1, self.C + 1, cur, self.lat_h, self.lat_w,
                                 device=self.gen_device, dtype=self.dtype)
        if self.global_start == 0:                                # image lands in frame 0 only
            cond_chunk[:, :self.C, 0:1] = self.image_cond
            cond_chunk[:, self.C, 0] = 1.0
        latent = torch.randn(1, self.C, cur, self.lat_h, self.lat_w,
                             device=self.gen_device, dtype=self.dtype)
        rope_total = self.global_start + cur

        sched = self._Sched(shift=self.shift, reverse=True, solver="euler")
        sched.set_timesteps(self.num_steps, device=self.gen_device)
        for t in sched.timesteps:
            ts_in = torch.full((cur,), t, device=self.gen_device, dtype=sched.timesteps.dtype)
            hidden = torch.cat([latent, cond_chunk], dim=1)       # [1,65,cur,30,52]
            with torch.autocast("cuda", dtype=self.dtype):
                pred = self.model(
                    bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                    hidden_states=hidden, timestep=ts_in, timestep_r=None, mask_type="i2v",
                    return_dict=False, kv_cache=self.kv_cache, cache_vision=False,
                    rope_temporal_size=rope_total, start_rope_start_idx=self.global_start,
                    viewmats=vm, Ks=ks)[0]
            latent = sched.step(pred, t, latent, return_dict=False)[0]

        # clean-context pass: append this chunk's denoised vision K/V
        ctx_ts = torch.full((cur,), self.stab - 1, device=self.gen_device, dtype=self.dtype)
        denoised_input = torch.cat([latent, cond_chunk], dim=1)
        with torch.autocast("cuda", dtype=self.dtype):
            new_kv = self.model(
                bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                hidden_states=denoised_input, timestep=ctx_ts, timestep_r=None, mask_type="i2v",
                return_dict=False, kv_cache=self.kv_cache, cache_vision=True,
                rope_temporal_size=rope_total, start_rope_start_idx=self.global_start,
                viewmats=vm, Ks=ks)
        for j in range(self.num_layers):
            c = self.kv_cache[j]
            if c["k_vision"] is None:
                c["k_vision"], c["v_vision"] = new_kv[j]["k_vision"], new_kv[j]["v_vision"]
            else:
                c["k_vision"] = torch.cat([c["k_vision"], new_kv[j]["k_vision"]], dim=2)
                c["v_vision"] = torch.cat([c["v_vision"], new_kv[j]["v_vision"]], dim=2)
            self._trim_vision_cache(c)
        self.global_start += cur
        return latent

    # --------------------------------------------------------------- decode
    @torch.no_grad()
    def decode_step(self, latent):
        """Incremental VAE decode of just this chunk (feat_cache persists across chunks)."""
        # VAE is fp16 (kept fp16 so the encode path matches the reference); decode in pure
        # fp16 — under autocast the conv bias stays fp16 and mismatches the bf16 input.
        z = latent.to(self.vae_device, dtype=torch.float16)
        if self.shift_factor:
            z = z / self.scaling + self.shift_factor
        else:
            z = z / self.scaling
        outs = []
        for i in range(z.shape[2]):
            self.vae._conv_idx = [0]
            first = (self._dec_frame == 0)
            out = self.vae.decoder(z[:, :, i:i + 1], feat_cache=self.vae._feat_map,
                                   feat_idx=self.vae._conv_idx, first_chunk=first)
            outs.append(out)
            self._dec_frame += 1
        video = torch.cat(outs, dim=2)                             # [1,3,P,H,W]
        video = (video.float().clamp(-1, 1) + 1) / 2
        v = (video[0] * 255.0).round().to(torch.uint8)             # [3,P,H,W]
        return v.permute(1, 2, 3, 0).contiguous().cpu().numpy()    # [P,H,W,3]

    # ----------------------------------------------------------- prompt steering
    @torch.no_grad()
    def set_prompt(self, prompt, k=None):
        """Steer the live stream to a new caption. Re-encodes the prompt and schedules a
        LERP ramp: over the next k chunks (default self.ramp_chunks=8) the text embedding is
        linearly interpolated old->new and the text K/V is recached each chunk, so the world
        morphs smoothly instead of hard-cutting. The vision K/V (already-generated frames)
        carries the motion through the transition. Mirrors Causal-Forcing's ramp_to().

        The server calls this every chunk with the composed caption; identical prompts are a
        no-op, so a ramp only starts when the user actually steers (voice/text)."""
        if not prompt or prompt == self.cur_prompt or self.kv_cache is None:
            return
        k = self.ramp_chunks if k is None else max(1, int(k))
        dev, dt = self.gen_device, self.dtype
        new_embed, new_mask = self.hy.encode_text(self.text_encoder, prompt, dev)
        new_embed = new_embed.to(dev, dt)
        new_mask = (new_mask.to(dev, dt) if new_mask is not None
                    else torch.ones(new_embed.shape[:2], device=dev, dtype=dt))
        # byt5 glyph stream follows the caption immediately (small; ~zeros for non-text prompts)
        if self.byt5_model is not None:
            self.byt5_model.to(dev)
            bs, bm = self.hy.encode_byt5(self.byt5_model, self.byt5_tokenizer, prompt, 256, dev)
            self.byt5_model.cpu()
            self.extra = {"byt5_text_states": bs.to(dev, dt), "byt5_text_mask": bm.to(dev, dt)}
        L = max(self.prompt_embed.shape[1], new_embed.shape[1])
        self._ramp = {"oe": self._pad_to(self.prompt_embed, L), "ne": self._pad_to(new_embed, L),
                      "om": self._pad_to(self.prompt_mask, L), "nm": self._pad_to(new_mask, L),
                      "i": 0, "k": k}
        self.cur_prompt = prompt
        self.prompt = prompt
        if k == 1:                       # k=1 == instant hard cut
            self._advance_ramp()

    @torch.no_grad()
    def _advance_ramp(self):
        """Advance an in-progress prompt ramp by one chunk and recache the text K/V."""
        if self._ramp is None:
            return
        r = self._ramp
        r["i"] += 1
        s = min(1.0, r["i"] / r["k"])                       # linear 0->1 over k chunks
        if r["i"] >= r["k"]:
            embed, mask = r["ne"], r["nm"]
            self._ramp = None
        else:
            embed = (1.0 - s) * r["oe"] + s * r["ne"]
            mask = torch.maximum(r["om"], r["nm"])          # union: keep both token sets live
        try:
            self._recache_text(embed, mask)                 # commits only if it produced real K/V
            self.prompt_embed, self.prompt_mask = embed, mask
        except Exception as e:                              # degenerate interp -> drop the ramp,
            print(f"[hy_worker] ramp recache failed ({e}); keeping current prompt")
            self._ramp = None                               # keep the last good text cache


# ----------------------------- headless latency / rollout test -----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer_dir", default="./ckpts/HY15/Action2V/dmd")
    ap.add_argument("--image", default=None, help="seed image (defaults to first example.json entry)")
    ap.add_argument("--caption", default=None)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--gen_device", default="cuda:0")
    ap.add_argument("--vae_device", default=None)
    ap.add_argument("--cap", type=int, default=24)
    ap.add_argument("--sink", type=int, default=1,
                    help="HY vision-KV sink in latent frames; pins the seed anchor during eviction")
    ap.add_argument("--out", default="./outputs/live/hy_streaming.mp4")
    ap.add_argument("--switch_caption", default=None,
                    help="if set, steer to this caption mid-rollout to test live prompt ramp")
    ap.add_argument("--switch_at", type=int, default=None,
                    help="chunk index to trigger the prompt switch (default: 1/3 through)")
    ap.add_argument("--ramp", type=int, default=8, help="LERP ramp length in chunks (1 = hard cut)")
    args = ap.parse_args()

    # default seed = first camera example
    if args.image is None:
        import json
        ex = json.load(open("assets/example.json"))
        ex = [e for e in ex if e.get("trajectory")][0]
        base = os.path.dirname(os.path.abspath("assets/example.json"))
        args.image = os.path.join(base, ex["image"])
        args.caption = args.caption or ex["caption"]
        print(f"[test] seed image={args.image}\n[test] caption={args.caption[:80]}")

    t0 = time.time()
    w = HYStreamingWorker(transformer_dir=args.transformer_dir, gen_device=args.gen_device,
                          vae_device=args.vae_device, max_vision_frames=args.cap,
                          sink_vision_frames=args.sink)
    print(f"[test] loaded in {time.time() - t0:.1f}s")
    w.start(args.image, args.caption)
    print(f"[test] start() done in {time.time() - t0:.1f}s")

    def cam(i):
        if i < 3:  return {"forward": 1.0, "speed": 1.0}
        if i < 6:  return {"turn": -1.0, "speed": 1.0}
        return {"pitch": 1.0, "speed": 1.0}

    switch_at = args.switch_at if args.switch_at is not None else max(1, args.chunks // 3)
    frames, gen_ms, dec_ms = [], [], []
    for i in range(args.chunks):
        if args.switch_caption and i == switch_at:
            t = time.time(); w.set_prompt(args.switch_caption, k=args.ramp)
            print(f"[test] SET_PROMPT at chunk {i} (ramp={args.ramp}, "
                  f"{(time.time() - t) * 1e3:.0f}ms): {args.switch_caption[:70]}", flush=True)
        torch.cuda.synchronize(w.gen_device)
        t = time.time(); lat = w.gen_step(cam(i)); torch.cuda.synchronize(w.gen_device); gen_ms.append((time.time() - t) * 1e3)
        t = time.time(); px = w.decode_step(lat); dec_ms.append((time.time() - t) * 1e3)
        frames.append(px)
        print(f"[test] chunk {i:2d}: gen {gen_ms[-1]:6.0f}ms decode {dec_ms[-1]:6.0f}ms "
              f"frames {px.shape[0]} cam {cam(i)}", flush=True)

    allf = np.concatenate(frames, axis=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    from torchvision.io import write_video
    write_video(args.out, torch.from_numpy(allf), fps=w.fps)
    g = np.mean(gen_ms[1:]) if len(gen_ms) > 1 else gen_ms[0]
    d = np.mean(dec_ms[1:]) if len(dec_ms) > 1 else dec_ms[0]
    print(f"\n[test] wrote {args.out} ({allf.shape[0]} frames)")
    print(f"[test] steady: gen {g:.0f}ms + decode {d:.0f}ms = {g + d:.0f}ms/chunk "
          f"(pipelined ~= max = {max(g, d):.0f}ms)")


if __name__ == "__main__":
    main()
