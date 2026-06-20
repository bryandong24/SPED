"""Phase 2: audio file -> ASR -> prompt schedule -> driveable LongLive -> video.

Transcribe a narration with faster-whisper, group segments into utterances by
silence gaps, then drive StreamingLongLive: at each chunk (video time = audio
time, 1:1) the active prompt = the latest utterance started by that time. The
video steers to whatever scene is being described as the narration plays.
"""
import os, sys, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from streaming_longlive import StreamingLongLive
from load_longlive import load_pipeline

CHUNK_SECONDS = 0.75  # 3 latent frames * 4 (vae temporal) / 16 fps


def asr_utterances(wav, model_size="base.en", gap_thresh=1.5):
    from faster_whisper import WhisperModel
    m = WhisperModel(model_size, device="cuda", compute_type="float16")
    segs, _ = m.transcribe(wav, word_timestamps=False)
    utts, cur, cur_start, last_end = [], [], None, None
    for s in segs:
        txt = s.text.strip()
        if not txt:
            continue
        if last_end is not None and s.start - last_end > gap_thresh and cur:
            utts.append((cur_start, " ".join(cur)))
            cur, cur_start = [], None
        if cur_start is None:
            cur_start = s.start
        cur.append(txt)
        last_end = s.end
    if cur:
        utts.append((cur_start, " ".join(cur)))
    return utts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="/mnt/data/SPED/gino/out/audio_stream/narration.wav")
    ap.add_argument("--out", default="/mnt/data/SPED/gino/out/audio_stream/audio_driven.mp4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--asr", default="base.en")
    args = ap.parse_args()

    import soundfile as sf
    audio, sr = sf.read(args.wav)
    audio_dur = len(audio) / sr
    print(f"audio: {audio_dur:.1f}s")

    print("transcribing...")
    utts = asr_utterances(args.wav, args.asr)
    print("utterances (prompt schedule):")
    for st, t in utts:
        print(f"  @{st:5.1f}s -> {t[:70]}")

    print("loading LongLive...")
    pipe, cfg = load_pipeline()
    gen = StreamingLongLive(pipe, seed=args.seed)

    # video covers the audio duration
    n_chunks = int(np.ceil(audio_dur / CHUNK_SECONDS))
    total_frames = n_chunks * gen.nfpb
    init_prompt = utts[0][1] if utts else "a video"
    gen.start(init_prompt, total_frames=total_frames)

    def active_prompt(vtime):
        p = utts[0][1]
        for st, t in utts:
            if st <= vtime:
                p = t
        return p

    print(f"generating {n_chunks} chunks (~{total_frames* 4/16:.0f}s)...")
    cur_prompt = init_prompt
    switches = []
    t_start = time.time()
    for c in range(n_chunks):
        vtime = c * CHUNK_SECONDS
        want = active_prompt(vtime)
        if want != cur_prompt:
            t0 = time.time(); gen.recache(want); cur_prompt = want
            switches.append((c, vtime, want))
            print(f"  [RECACHE] chunk {c} (vtime {vtime:.1f}s, {(time.time()-t0)*1e3:.0f}ms) -> {want[:50]}")
        gen.step()
    gen_s = time.time() - t_start

    frames = gen.decode_all()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import imageio
    from PIL import Image
    imageio.mimwrite(args.out, frames, fps=16, codec="libx264")
    # mux the narration audio onto the video so they can watch+listen
    av_out = args.out.replace(".mp4", "_with_audio.mp4")
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-i", args.out, "-i", args.wav, "-c:v", "copy",
                    "-c:a", "aac", "-shortest", av_out], capture_output=True)
    idx = np.linspace(0, frames.shape[0]-1, 20).round().astype(int)
    sw_px = [int(v/0.0625*4/4) for _, v, _ in switches]  # vtime->px: vtime*16
    sw_px = [int(v*16) for _, v, _ in switches]
    tiles = []
    for i in idx:
        f = np.array(Image.fromarray(frames[i]).resize((166, 96)))
        if any(0 <= i - s < 24 for s in sw_px): f[:4, :] = [255, 0, 0]
        tiles.append(f)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(args.out.replace(".mp4", "_sheet.png"))
    print(f"\n[done] {args.out}  (+ {av_out} with narration audio)")
    print(f"  {frames.shape[0]} frames, gen {gen_s:.1f}s -> {frames.shape[0]/gen_s:.1f} FPS")
    print(f"  video switched scenes at: " + ", ".join(f"{v:.1f}s" for _, v, _ in switches))


if __name__ == "__main__":
    main()
