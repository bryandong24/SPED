"""LIVE audio-steered video demo on CAUSAL FORCING (hard-cut swapping).

Port of web_live.py from LongLive to Causal Forcing. Browser mic -> socket.io ->
faster-whisper ASR -> PromptBus -> continuous CF generation loop (StreamingCF),
streaming decoded frames back as JPEGs. Prompt switching is a plain HARD CUT
(re-encode prompt + reset cross-attn), per request — no recache.

Reach from a laptop:
    ssh -L 5009:localhost:5009 spycoder@<box>
    open http://localhost:5009
Launch (video=cuda:0, ASR=cuda:1):
    CUDA_VISIBLE_DEVICES=6,7 .venv/bin/python web_live_cf.py --port 5009
"""
import os, sys, time, re, base64, io, queue, threading
import numpy as np
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from cf_streaming import StreamingCF, load_cf_pipeline, PromptBus
from live_pipeline import RollingBuffer, PromptDebouncer, SR, CHUNK_SECONDS

MAX_SECONDS = 90
RAMP_CHUNKS = 4   # forward conditioning-ramp horizon (chunks) on a prompt change (snappier steer)
WINDOW = 12       # rolling self-attn window (frames); smaller = faster gen + quicker visible change
SINK = 3
ASR_RMS_GATE = 0.012   # skip transcription when the recent audio is ~silence (stops Whisper
                       # hallucinating on quiet and thrashing the GIL, which throttled gen to ~4fps)
DEFAULT_START = ("A fluffy golden retriever with a glossy honey-gold coat and a red leather collar "
                 "bounds through a vast sunlit meadow of orange and yellow wildflowers, captured in "
                 "crisp HDR 4K with razor-sharp fur detail. Warm golden afternoon light rakes across "
                 "the lush green grass and pollen drifts glowing in the air. The low camera tracks fast "
                 "alongside the dog at a full sprint, tongue out and ears flapping, petals scattering in "
                 "its wake, rolling green hills and a bright blue sky beyond, cinematic and photorealistic.")
TEMPLATES = os.path.join(os.path.dirname(__file__), "templates")

pipe = None
asr_model = None
bus = PromptBus(initial=DEFAULT_START)
buf = RollingBuffer(seconds=8.0)
state = {"active": False, "stop": False, "seed": DEFAULT_START, "current": "", "epoch": 0}
frame_q = queue.Queue(maxsize=32)
gen_lock = threading.Lock()   # only one gen_loop may touch pipe.kv_cache1 at a time

app = Flask(__name__, template_folder=TEMPLATES)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=32 * 1024 * 1024,
                    async_mode="threading")


def compose():
    # full-prompt replacement: a steer REPLACES the whole prompt (no fixed-seed anchoring)
    return state["current"] or state["seed"]


def init_models(asr_gpu=1, do_compile=False):
    global pipe, asr_model
    print("loading Causal Forcing (video, cuda:0)...")
    pipe = load_cf_pipeline(window=WINDOW, sink=SINK)
    if do_compile:
        try:
            torch.cuda.set_device(0)
            pipe.generator.model = torch.compile(pipe.generator.model, dynamic=True)
            g = StreamingCF(pipe, seed=0); g.start("a warmup scene, cinematic", total_frames=45)
            for _ in range(15):
                g.step()
            pipe.vae.model.clear_cache()
            print("torch.compile warmup done.")
        except Exception as e:
            print("torch.compile failed, continuing uncompiled:", e)
    from faster_whisper import WhisperModel
    print(f"loading faster-whisper distil-large-v3 (ASR, cuda:{asr_gpu})...")
    try:
        asr_model = WhisperModel("distil-large-v3", device="cuda", device_index=asr_gpu, compute_type="float16")
    except Exception as e:
        print("distil-large-v3 failed, trying base.en/cpu:", e)
        asr_model = WhisperModel("base.en", device="cpu", compute_type="int8")
    torch.cuda.set_device(0)
    print("models ready.")


@app.route("/")
def index():
    return render_template("live.html")


@socketio.on("audio")
def on_audio(data):
    try:
        pcm = np.frombuffer(base64.b64decode(data["pcm"]), dtype=np.int16).astype(np.float32) / 32768.0
        buf.append(pcm)
    except Exception as e:
        print("audio err", e)


def asr_loop(ep):
    # More responsive than the LongLive default: commit on the first stable tick and
    # only reject near-duplicates (jaccard 0.85), so same-subject scene changes
    # ("red car on coast" -> "red car in neon city") aren't suppressed.
    deb = PromptDebouncer(jaccard=0.85, stable_ticks=1)
    while state["active"] and state["epoch"] == ep:
        socketio.sleep(0.8)
        a = buf.get()
        if len(a) < SR * 0.6:
            continue
        # VAD gate: only transcribe if the most-recent ~1.5s actually has speech energy.
        # Otherwise Whisper hallucinates ("thank you", "I don't know") and the ~200ms calls
        # thrash the GIL every 0.8s, starving the generation thread.
        recent = a[-int(SR * 1.5):]
        if recent.size == 0 or float(np.sqrt(np.mean(recent ** 2))) < ASR_RMS_GATE:
            continue
        try:
            _ta = time.time()
            segs, _ = asr_model.transcribe(a, language="en")
            text = " ".join(s.text.strip() for s in segs).strip()
            _asr_ms = (time.time() - _ta) * 1e3
        except Exception as e:
            print("asr err", e); continue
        socketio.emit("asr_live", {"raw": text, "window_s": round(len(a) / SR, 1)})
        sents = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
        cand = sents[-1] if sents else text
        committed = deb.update(cand)
        print(f"[asr] {_asr_ms:.0f}ms win={len(a)/SR:.1f}s "
              f"{'COMMIT '+repr(committed[:50]) if committed else 'heard '+repr(text[:50])}", flush=True)
        if committed:
            state["current"] = committed
            bus.set(compose())


def gen_loop(start_prompt, ep):
    # Serialize: a new Start (which bumped state["epoch"]) signals any running loop to exit;
    # we wait for it to release the lock so we never reinitialize pipe.kv_cache1 underneath a
    # live generator (that race corrupts the rolling-window KV cache and crashes generation).
    with gen_lock:
        if state["epoch"] != ep:      # superseded by a newer Start while we waited
            return
        while not frame_q.empty():    # drop any stale frames/sentinel from a prior run
            try: frame_q.get_nowait()
            except queue.Empty: break
        torch.cuda.set_device(0)
        try:
            gen = StreamingCF(pipe, seed=int(time.time()) % 100000, window=WINDOW, sink=SINK)
            n_chunks = int(np.ceil(MAX_SECONDS / CHUNK_SECONDS))
            gen.start(start_prompt, total_frames=n_chunks * gen.nfpb)
            pipe.vae.model.clear_cache()
            cur = start_prompt
            socketio.emit("status", {"msg": "generating"})
            socketio.emit("embedded", {"prompt": compose(), "seed": state["seed"], "current": state["current"]})
            gen_ms = dec_ms = blk_ms = 0.0   # rolling sums over a print window
            win = 0
            for c in range(n_chunks):
                if state["stop"] or state["epoch"] != ep:
                    break
                p, _ = bus.get()
                if p and p != cur:
                    gen.ramp_to(p, k=RAMP_CHUNKS); cur = p   # FORWARD RAMP (no recache)
                    socketio.emit("embedded", {"prompt": p, "seed": state["seed"], "current": state["current"]})
                    socketio.emit("steer", {"text": p})
                    print(f"[gen] STEER applied at chunk {c}: {p[:60]!r}", flush=True)
                tb = time.time()
                tg = time.time(); den = gen.step(); gen_ms += (time.time() - tg) * 1e3
                td = time.time(); frames = gen.decode_chunk(den); dec_ms += (time.time() - td) * 1e3
                for f in frames:
                    while state["active"] and not state["stop"] and state["epoch"] == ep:
                        try:
                            frame_q.put(f, timeout=0.5); break
                        except queue.Full:
                            continue
                blk_ms += (time.time() - tb) * 1e3   # includes queue-backpressure wait (emit/ASR stalls)
                win += 1
                if win == 16:
                    print(f"[gen] chunk {c-15}-{c}: gen {gen_ms/16:.0f}ms decode {dec_ms/16:.0f}ms "
                          f"block {blk_ms/16:.0f}ms -> {1000*16/blk_ms:.1f} fps (qsize={frame_q.qsize()})",
                          flush=True)
                    gen_ms = dec_ms = blk_ms = 0.0; win = 0
        except Exception as e:
            import traceback; traceback.print_exc()
            socketio.emit("status", {"msg": f"generation error: {e} — press Start to retry"})
        finally:
            frame_q.put(None)
            if state["epoch"] == ep:
                state["active"] = False


def emitter_loop(ep):
    min_period = 1.0 / 16
    while state["epoch"] == ep:
        try:
            f = frame_q.get(timeout=1.0)
        except queue.Empty:
            if not state["active"]:
                break
            continue
        if f is None:
            break
        t0 = time.time()
        b = io.BytesIO()
        Image.fromarray(f).save(b, format="JPEG", quality=78)
        socketio.emit("frame", {"img": base64.b64encode(b.getvalue()).decode()})
        dt = min_period - (time.time() - t0)
        if dt > 0:
            socketio.sleep(dt)
    socketio.emit("status", {"msg": "done"})


@socketio.on("start")
def on_start(data):
    # Supersede any running generation instead of ignoring repeated Starts. Bumping the
    # epoch tells the old loops to exit; the new gen_loop blocks on gen_lock until the old
    # one has released it, so two generations never share pipe.kv_cache1.
    start_prompt = (data or {}).get("prompt") or DEFAULT_START
    state["epoch"] += 1
    ep = state["epoch"]
    state["seed"] = start_prompt
    state["current"] = ""
    state["stop"] = False
    state["active"] = True
    bus.set(compose())
    buf.data = np.zeros(0, dtype=np.float32)
    socketio.emit("status", {"msg": "starting…"})
    socketio.start_background_task(asr_loop, ep)
    socketio.start_background_task(gen_loop, start_prompt, ep)
    socketio.start_background_task(emitter_loop, ep)


@socketio.on("set_prompt")
def on_set_prompt(data):
    p = (data or {}).get("prompt", "").strip()
    if p:
        state["current"] = p
        bus.set(compose())
        socketio.emit("heard", {"text": "(typed) " + p})


@socketio.on("stop")
def on_stop():
    state["stop"] = True
    state["active"] = False
    state["epoch"] += 1   # invalidate the running loops so they exit promptly


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5009)
    ap.add_argument("--asr_gpu", type=int, default=1)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    init_models(asr_gpu=args.asr_gpu, do_compile=args.compile)
    print(f"\n>>> ssh -L {args.port}:localhost:{args.port} spycoder@<box>  then http://localhost:{args.port}\n")
    socketio.run(app, host="127.0.0.1", port=args.port, allow_unsafe_werkzeug=True)
