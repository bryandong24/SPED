"""Generate a test narration WAV: 3 scene commands separated by silence gaps.

Emulates a user speaking scene descriptions over time. gTTS per sentence ->
ffmpeg to 16kHz mono wav -> concatenate with controlled silence gaps so we know
roughly when each command starts.
"""
import os, subprocess, tempfile
import numpy as np
import soundfile as sf
from gtts import gTTS

OUT = "/data/SPED/gino/out/audio_stream/narration.wav"
SR = 16000
GAP_S = 4.0  # silence between commands

SENTENCES = [
    "A fluffy golden retriever runs through a sunny green meadow full of orange wildflowers in bright daylight.",
    "Now the same golden retriever runs across a deep snowy field on a cold winter night under a full moon.",
    "Now the golden retriever runs down a rain soaked neon city street at night with glowing pink and blue lights.",
]


def tts_to_wav(text, path):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3 = f.name
    gTTS(text=text, lang="en").save(mp3)
    subprocess.run(["ffmpeg", "-y", "-i", mp3, "-ar", str(SR), "-ac", "1", path],
                   check=True, capture_output=True)
    os.unlink(mp3)
    a, _ = sf.read(path)
    return a.astype(np.float32)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    gap = np.zeros(int(GAP_S * SR), dtype=np.float32)
    chunks, starts, t = [], [], 0.0
    for i, s in enumerate(SENTENCES):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wp = f.name
        a = tts_to_wav(s, wp); os.unlink(wp)
        starts.append(t)
        chunks.append(a); t += len(a) / SR
        if i < len(SENTENCES) - 1:
            chunks.append(gap); t += GAP_S
    audio = np.concatenate(chunks)
    sf.write(OUT, audio, SR)
    print(f"[narration] {OUT}  ({len(audio)/SR:.1f}s)")
    for i, (st, s) in enumerate(zip(starts, SENTENCES)):
        print(f"  ~{st:5.1f}s : {s[:60]}")


if __name__ == "__main__":
    main()
