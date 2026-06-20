"""Driveable streaming generator for LongLive — dynamic prompt updates between chunks.

Unlike the repo's interactive inference (fixed `switch_frame_indices`), this polls a
thread-safe PromptBus between every chunk and recaches whenever the prompt changes.
This is the core enabler for audio-steered video: an ASR thread writes the latest
transcript into the bus; the generation loop picks it up at the next chunk boundary.

Phase 1: validate dynamic injection with a scripted prompt timeline (__main__).
"""
import threading, time, argparse, os, sys
import torch
import numpy as np
from einops import rearrange

sys.path.insert(0, os.path.dirname(__file__))
from load_longlive import load_pipeline, LONGLIVE_DIR


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


class StreamingLongLive:
    def __init__(self, pipeline, seed=1):
        self.p = pipeline
        self.device = next(pipeline.generator.parameters()).device
        self.nfpb = pipeline.num_frame_per_block
        self.fsl = pipeline.frame_seq_length
        self.local_attn = pipeline.local_attn_size
        self.seed = seed
        self.H, self.W, self.C = 60, 104, 16  # latent dims (=480x832 px, 16ch)

    @torch.no_grad()
    def start(self, prompt, total_frames):
        assert total_frames % self.nfpb == 0
        self.total = total_frames
        g = torch.Generator("cpu").manual_seed(self.seed)
        self.noise = torch.randn([1, total_frames, self.C, self.H, self.W],
                                 generator=g, dtype=torch.bfloat16).to(self.device)
        self.output = torch.zeros_like(self.noise)
        # cache sizing (matches the pipeline's inference())
        if self.local_attn != -1:
            kv_size = self.local_attn * self.fsl
        else:
            kv_size = total_frames * self.fsl
        self.p._initialize_kv_cache(1, dtype=self.noise.dtype, device=self.device,
                                    kv_cache_size_override=kv_size)
        self.p._initialize_crossattn_cache(batch_size=1, dtype=self.noise.dtype, device=self.device)
        self.p.generator.model.local_attn_size = self.local_attn
        self.p._set_all_modules_max_attention_size(self.local_attn)
        self.cond = self.p.text_encoder(text_prompts=[prompt])
        self.cur_prompt = prompt
        self.cur_frame = 0

    @torch.no_grad()
    def recache(self, new_prompt):
        """Swap to a new prompt mid-rollout via LongLive's trained recache."""
        cond_new = self.p.text_encoder(text_prompts=[new_prompt])
        self.p._recache_after_switch(self.output, self.cur_frame, cond_new)
        self.cond = cond_new
        self.cur_prompt = new_prompt

    @torch.no_grad()
    def step(self):
        """Generate one chunk; return its clean latents [1, nfpb, C, H, W]."""
        cur = self.nfpb
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
        self.output[:, self.cur_frame:self.cur_frame + cur] = den
        # clean-context pass (write this chunk's K/V)
        ctx_t = torch.ones_like(timestep) * self.p.args.context_noise
        self.p.generator(noisy_image_or_video=den, conditional_dict=self.cond, timestep=ctx_t,
                         kv_cache=self.p.kv_cache1, crossattn_cache=self.p.crossattn_cache, current_start=cs)
        self.cur_frame += cur
        return den

    @torch.no_grad()
    def run(self, bus: PromptBus, on_chunk=None):
        """Drive generation; between every chunk, recache if the bus prompt changed."""
        last_version = 0
        n_chunks = self.total // self.nfpb
        for c in range(n_chunks):
            prompt, version = bus.get()
            if version != last_version and prompt != self.cur_prompt:
                t0 = time.time()
                self.recache(prompt)
                last_version = version
                if on_chunk: on_chunk("RECACHE", c, self.cur_frame, prompt, (time.time() - t0) * 1e3)
            t0 = time.time()
            self.step()
            if on_chunk: on_chunk("CHUNK", c, self.cur_frame, self.cur_prompt, (time.time() - t0) * 1e3)

    @torch.no_grad()
    def decode_all(self):
        video = self.p.vae.decode_to_pixel(self.output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return (255 * rearrange(video, "b t c h w -> b t h w c")[0]).to(torch.uint8).cpu().numpy()


# ----------------------- Phase 1: scripted prompt timeline -----------------------
def main():
    import imageio
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=120)  # 120 latent ~ 30s
    ap.add_argument("--out", default="/data/SPED/gino/out/audio_stream/scripted.mp4")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    print("loading LongLive pipeline...")
    pipe, cfg = load_pipeline()
    print("loaded. local_attn_size=", pipe.local_attn_size, "nfpb=", pipe.num_frame_per_block)

    gen = StreamingLongLive(pipe, seed=args.seed)
    P1 = "A fluffy golden retriever sprinting through a sunlit meadow of orange wildflowers, warm golden daylight, lush green grass, cinematic, photorealistic, highly detailed, 4k"
    P2 = "A fluffy golden retriever running across a deep snowy field on a frigid winter night, full moon, falling snow, frosted pine trees, deep blue moonlight, cinematic, photorealistic, highly detailed, 4k"
    P3 = "A fluffy golden retriever running down a rain-soaked neon city street at night, vibrant pink and blue neon reflections on wet asphalt, skyscrapers, cinematic, photorealistic, highly detailed, 4k"
    bus = PromptBus(initial=P1)
    gen.start(P1, total_frames=args.total)

    # scripted timeline: switch prompt at chunk 13 and 27 (≈10s, 20s) — arbitrary points
    schedule = {13: P2, 27: P3}

    log = []
    def on_chunk(kind, c, frame, prompt, ms):
        if c in schedule and kind == "CHUNK":  # set the bus just before this chunk's neighbor
            pass
        log.append((kind, c, round(ms, 1)))
        print(f"  [{kind}] chunk {c:3d} frame {frame:3d} {ms:6.1f}ms :: {prompt[:40]}")

    # drive: emulate an external controller updating the bus at scheduled chunks
    n_chunks = args.total // gen.nfpb
    last_version = 0
    import time as _t
    t_start = _t.time()
    for c in range(n_chunks):
        if c in schedule:
            bus.set(schedule[c])
        prompt, version = bus.get()
        if version != last_version and prompt != gen.cur_prompt:
            t0 = _t.time(); gen.recache(prompt); last_version = version
            print(f"  [RECACHE] before chunk {c} ({(_t.time()-t0)*1e3:.0f}ms) -> {prompt[:40]}")
        t0 = _t.time(); gen.step()
        print(f"  chunk {c:3d} frame {gen.cur_frame:3d} {( _t.time()-t0)*1e3:6.1f}ms")
    gen_s = _t.time() - t_start

    frames = gen.decode_all()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=16, codec="libx264")
    # contact sheet
    idx = np.linspace(0, frames.shape[0]-1, 18).round().astype(int)
    sw_px = [k*gen.nfpb*4 for k in schedule]
    tiles=[]
    for i in idx:
        f=np.array(Image.fromarray(frames[i]).resize((166,96)))
        if any(0<=i-s<24 for s in sw_px): f[:4,:]=[255,0,0]
        tiles.append(f)
    Image.fromarray(np.concatenate(tiles,axis=1)).save(args.out.replace(".mp4","_sheet.png"))
    print(f"\n[done] {args.out}  ({frames.shape[0]} frames, gen {gen_s:.1f}s, {frames.shape[0]/gen_s:.1f} FPS)")
    print(f"switches scheduled at chunks {list(schedule)} (red stripe in sheet)")


if __name__ == "__main__":
    main()
