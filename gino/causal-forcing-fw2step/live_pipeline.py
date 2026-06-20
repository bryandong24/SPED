"""Audio buffering + ASR debounce helpers for the live audio-steered demo.

Trimmed copy of gino/audio_stream/live_pipeline.py: only the pieces web_live_cf.py
needs (RollingBuffer, PromptDebouncer, SR, CHUNK_SECONDS). The LongLive-specific
FileMic/ASRWorker/run_live machinery is intentionally dropped so this folder has no
dependency on the LongLive pipeline.
"""
import threading
import numpy as np

SR = 16000
CHUNK_SECONDS = 0.75  # reference only; the gen loop derives chunk length from nfpb


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
