"""LIVE audio-steered continuous video demo.

Runs the entire pipeline concurrently:
  - AudioSource feeds audio into a rolling buffer in REAL TIME (FileMic = a wav
    streamed at 1x speed, standing in for a microphone on this headless box).
  - ASRWorker thread re-transcribes the rolling buffer (~1s cadence, faster-whisper
    on a SEPARATE gpu) and pushes the latest spoken command to a PromptBus.
  - The generation loop streams video CONTINUOUSLY from a start prompt, recaching
    whenever the PromptBus changes, decoding each chunk with the streaming VAE and
    writing frames to a growing mp4 — paced to real time so video-time≈audio-time.

Swap FileMic -> a real MicSource (sounddevice) or a websocket source for live use.
"""
import os, sys, time, threading, re, argparse, subprocess
import numpy as np
import torch
import soundfile as sf
import imageio

sys.path.insert(0, os.path.dirname(__file__))
from streaming_longlive import StreamingLongLive, PromptBus
from load_longlive import load_pipeline

SR = 16000
CHUNK_SECONDS = 0.75  # 3 latent frames -> ~12 px frames / 16 fps


class PromptDebouncer:
    """Filter noisy ASR output: require the candidate to be stable for N ticks,
    long enough, and a DIFFERENT scene (low word-overlap) before committing."""
    def __init__(self, min_len=18, jaccard=0.6, stable_ticks=2):
        self.min_len, self.jac, self.stable = min_len, jaccard, stable_ticks
        self.last_cand, self.cand_count, self.committed = None, 0, None

    @staticmethod
    def _sim(a, b):
        wa, wb = set(a.lower().split()), set(b.lower().split())
        return len(wa & wb) / max(1, len(wa | wb))

    def update(self, text):
        t = " ".join(text.split())
        if len(t) < self.min_len:
            return None
        if t == self.last_cand:
            self.cand_count += 1
        else:
            self.last_cand, self.cand_count = t, 1
        if self.cand_count < self.stable:
            return None
        if self.committed and self._sim(t, self.committed) > self.jac:
            return None
        self.committed = t
        return t


class RollingBuffer:
    def __init__(self, seconds=8.0):
        self.lock = threading.Lock()
        self.data = np.zeros(0, dtype=np.float32)
        self.max = int(seconds * SR)

    def append(self, a):
        with self.lock:
            self.data = np.concatenate([self.data, a.astype(np.float32)])[-self.max:]

    def get(self):
        with self.lock:
            return self.data.copy()


class FileMic(threading.Thread):
    """Stream a wav into the rolling buffer at real-time rate (stand-in for a mic)."""
    def __init__(self, wav, buf, step_s=0.25):
        super().__init__(daemon=True)
        self.audio, sr = sf.read(wav)
        assert sr == SR, f"resample {wav} to {SR}"
        self.audio = self.audio.astype(np.float32)
        self.buf = buf
        self.step = int(step_s * SR)
        self.done = False

    def run(self):
        for i in range(0, len(self.audio), self.step):
            self.buf.append(self.audio[i:i + self.step])
            time.sleep(self.step / SR)
        time.sleep(0.5)
        self.done = True


class ASRWorker(threading.Thread):
    """Re-transcribe the rolling buffer; push the latest spoken command to the bus."""
    def __init__(self, buf, bus, mic, device_index=1, cadence=1.0):
        super().__init__(daemon=True)
        from faster_whisper import WhisperModel
        # CPU ASR: avoids GPU contention + the current-device conflict with the
        # Wan text encoder (which places inputs on torch.cuda.current_device()).
        self.model = WhisperModel("base.en", device="cpu", compute_type="int8")
        self.buf, self.bus, self.mic, self.cadence = buf, bus, mic, cadence
        self.deb = PromptDebouncer()
        self.stop = False

    def run(self):
        while not self.stop:
            time.sleep(self.cadence)
            audio = self.buf.get()
            if len(audio) < SR * 0.6:
                if self.mic.done:
                    break
                continue
            segs, _ = self.model.transcribe(audio, language="en")
            text = " ".join(s.text.strip() for s in segs).strip()
            sents = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
            cand = sents[-1] if sents else text
            committed = self.deb.update(cand)
            if committed:
                self.bus.set(committed)
            if self.mic.done:
                break


def run_live(pipe, wav, start_prompt, out_path, asr_gpu=1, max_seconds=45, seed=1):
    bus = PromptBus(initial=start_prompt)
    buf = RollingBuffer(seconds=8.0)
    mic = FileMic(wav, buf)
    asr = ASRWorker(buf, bus, mic, device_index=asr_gpu, cadence=1.0)

    gen = StreamingLongLive(pipe, seed=seed)
    n_chunks = int(np.ceil(max_seconds / CHUNK_SECONDS))
    gen.start(start_prompt, total_frames=n_chunks * gen.nfpb)
    pipe.vae.model.clear_cache()  # streaming-decode cache

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=16, codec="libx264", macro_block_size=1)

    mic.start(); asr.start()
    cur = start_prompt
    timeline = []
    t0 = time.time()
    print(f"[live] start: {start_prompt[:50]}")
    for c in range(n_chunks):
        p, _ = bus.get()
        if p and p != cur:
            r0 = time.time(); gen.recache(p); cur = p
            timeline.append((round(time.time() - t0, 1), p))
            print(f"  [{time.time()-t0:5.1f}s] STEER -> {p[:55]}  (recache {(time.time()-r0)*1e3:.0f}ms)")
        den = gen.step()
        pix = pipe.vae.decode_to_pixel_chunk(den, use_cache=True)  # [1, nf, 3, H, W], [-1,1]
        frames = ((pix * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)[0].permute(0, 2, 3, 1).cpu().numpy()
        for f in frames:
            writer.append_data(f)
        # pace to real time so video-time ~ audio-time
        ahead = (c + 1) * CHUNK_SECONDS - (time.time() - t0)
        if ahead > 0:
            time.sleep(ahead)
        if mic.done and bus.get()[0] == cur and (time.time() - t0) > (len(mic.audio) / SR + 1.0):
            break
    writer.close()
    asr.stop = True
    gen_s = time.time() - t0
    print(f"[done] {out_path}  ({gen_s:.1f}s real-time)")
    # mux narration audio for review
    av = out_path.replace(".mp4", "_with_audio.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", out_path, "-i", wav, "-c:v", "copy",
                    "-c:a", "aac", "-shortest", av], capture_output=True)
    print(f"[done] {av} (with audio)")
    print("STEER timeline:", [(t, p[:30]) for t, p in timeline])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="/mnt/data/SPED/gino/out/audio_stream/narration.wav")
    ap.add_argument("--start_prompt", default="A fluffy golden retriever runs through a sunny green meadow full of orange wildflowers in bright daylight, cinematic, highly detailed")
    ap.add_argument("--out", default="/mnt/data/SPED/gino/out/audio_stream/live.mp4")
    ap.add_argument("--asr_gpu", type=int, default=1)
    ap.add_argument("--max_seconds", type=int, default=40)
    args = ap.parse_args()
    print("loading LongLive...")
    pipe, cfg = load_pipeline()
    run_live(pipe, args.wav, args.start_prompt, args.out, asr_gpu=args.asr_gpu, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
