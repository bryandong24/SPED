"""Web server for the LIVE audio-steered video demo — usable from a laptop via SSH.

Laptop browser captures the mic (getUserMedia) and streams 16kHz PCM over socket.io
to this server; the server runs ASR (faster-whisper, separate GPU) -> PromptBus and
a continuous LongLive generation loop, streaming decoded frames back to the browser
as base64 JPEGs. Reach it from the laptop with:
    ssh -L 5008:localhost:5008 spycoder@<box>
then open http://localhost:5008

Launch on the box (2 GPUs: video=cuda:0, ASR=cuda:1):
    CUDA_VISIBLE_DEVICES=1,2 .venv/bin/python web_live.py
"""
import os, sys, time, threading, re, base64, io, queue
import numpy as np
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from streaming_longlive import StreamingLongLive, PromptBus
from load_longlive import load_pipeline
from live_pipeline import RollingBuffer, PromptDebouncer, SR, CHUNK_SECONDS

MAX_SECONDS = 120
DEFAULT_START = "A fluffy golden retriever runs through a sunny green meadow full of orange wildflowers in bright daylight, cinematic, highly detailed, 4k"

pipe = None
asr_model = None
bus = PromptBus(initial=DEFAULT_START)
buf = RollingBuffer(seconds=8.0)
state = {"active": False, "stop": False, "seed": DEFAULT_START, "current": ""}
frame_q = queue.Queue(maxsize=32)  # ~1s buffer; gen produces, emitter drains @16fps


def compose():
    """Embedded prompt = the persistent SEED + the current steering STATE, so the
    base subject/style stays fixed and steering only changes the 'current state'."""
    seed, cur = state["seed"], state["current"]
    if not cur:
        return seed
    return f"{seed} Current state: {cur}."

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=32 * 1024 * 1024,
                    async_mode="threading")


def _warmup_compile():
    print("warming up torch.compile (~2min, one-time)...")
    g = StreamingLongLive(pipe, seed=0)
    g.start("a warmup scene, cinematic", total_frames=45)
    for _ in range(15):
        g.step()
    pipe.vae.model.clear_cache()
    print("warmup done.")


def init_models(asr_gpu=1, do_compile=True):
    global pipe, asr_model
    print("loading LongLive (video, cuda:0)...")
    pipe, _ = load_pipeline()
    if do_compile:
        try:
            torch.cuda.set_device(0)
            pipe.generator.model = torch.compile(pipe.generator.model, dynamic=True)
            print("torch.compile enabled on generator (default).")
        except Exception as e:
            print("torch.compile failed, continuing uncompiled:", e)
    from faster_whisper import WhisperModel
    # distil-large-v3: near-large-v3 accuracy, ~6x faster. Same GPU as video
    # (cuda:0) -> no current-device mismatch; brief ASR calls barely dent the H100.
    # ASR on a SEPARATE gpu (cuda:1) so it never steals cycles from generation.
    # (Launch with CUDA_VISIBLE_DEVICES=1,2 -> video=cuda:0, ASR=cuda:1.)
    print(f"loading faster-whisper distil-large-v3 (ASR, cuda:{asr_gpu})...")
    try:
        asr_model = WhisperModel("distil-large-v3", device="cuda", device_index=asr_gpu, compute_type="float16")
    except Exception as e:
        print("distil-large-v3 load failed, falling back to large-v3:", e)
        asr_model = WhisperModel("large-v3", device="cuda", device_index=asr_gpu, compute_type="float16")
    torch.cuda.set_device(0)  # restore main device for the video pipeline
    if do_compile:
        _warmup_compile()
    print("models ready.")


UMT5_BUDGET = 512  # umT5 tokenizer seq_len (Wan/LongLive). Prompt padded/truncated to this.


def umt5_tokens(prompt):
    """How many umT5 tokens this prompt uses (of the 512 budget)."""
    try:
        ids, mask = pipe.text_encoder.tokenizer([prompt], return_mask=True, add_special_tokens=True)
        return int(mask.gt(0).sum().item())
    except Exception:
        return len(prompt.split())  # fallback


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


def asr_loop():
    deb = PromptDebouncer()
    while state["active"]:
        socketio.sleep(0.8)
        a = buf.get()
        if len(a) < SR * 0.6:
            continue
        try:
            segs, _ = asr_model.transcribe(a, language="en")
            text = " ".join(s.text.strip() for s in segs).strip()
        except Exception as e:
            print("asr err", e); continue
        # live: the full rolling-window transcript the ASR currently hears
        socketio.emit("asr_live", {"raw": text, "window_s": round(len(a) / SR, 1)})
        sents = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
        cand = sents[-1] if sents else text
        committed = deb.update(cand)
        if committed:
            state["current"] = committed
            bus.set(compose())


def gen_loop(start_prompt):
    """Producer: generate + decode back-to-back, push raw frames into frame_q.
    Blocks when the queue is full (backpressure keeps it ~real-time)."""
    torch.cuda.set_device(0)  # this thread drives the video on cuda:0
    gen = StreamingLongLive(pipe, seed=int(time.time()) % 100000)
    n_chunks = int(np.ceil(MAX_SECONDS / CHUNK_SECONDS))
    gen.start(start_prompt, total_frames=n_chunks * gen.nfpb)
    pipe.vae.model.clear_cache()
    cur = start_prompt
    socketio.emit("status", {"msg": "generating"})
    socketio.emit("embedded", {"prompt": compose(), "seed": state["seed"], "current": state["current"], "tokens": umt5_tokens(compose()), "budget": UMT5_BUDGET})
    for c in range(n_chunks):
        if state["stop"]:
            break
        p, _ = bus.get()
        if p and p != cur:
            gen.recache(p); cur = p
            socketio.emit("embedded", {"prompt": p, "seed": state["seed"], "current": state["current"], "tokens": umt5_tokens(p), "budget": UMT5_BUDGET})
            socketio.emit("steer", {"text": p})
        den = gen.step()
        pix = pipe.vae.decode_to_pixel_chunk(den, use_cache=True)
        frames = ((pix * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)[0].permute(0, 2, 3, 1).cpu().numpy()
        for f in frames:
            while state["active"] and not state["stop"]:
                try:
                    frame_q.put(f, timeout=0.5); break
                except queue.Full:
                    continue
    frame_q.put(None)  # sentinel
    state["active"] = False


def emitter_loop():
    """Consumer: emit each frame as it becomes available, CAPPED at 16fps. This
    paces playback to the sustainable generation rate (no fixed-16fps schedule that
    races ahead and starves) -> smooth, no stalls even if gen < 16fps."""
    min_period = 1.0 / 16
    while True:
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
            socketio.sleep(dt)  # cap at 16fps; if gen is slower, get() blocks naturally
    socketio.emit("status", {"msg": "done"})


@socketio.on("start")
def on_start(data):
    if state["active"]:
        return
    start_prompt = (data or {}).get("prompt") or DEFAULT_START
    state["seed"] = start_prompt
    state["current"] = ""
    bus.set(compose())
    buf.data = np.zeros(0, dtype=np.float32)
    while not frame_q.empty():
        try: frame_q.get_nowait()
        except queue.Empty: break
    state["active"] = True; state["stop"] = False
    socketio.start_background_task(asr_loop)
    socketio.start_background_task(gen_loop, start_prompt)
    socketio.start_background_task(emitter_loop)


@socketio.on("set_prompt")
def on_set_prompt(data):
    """Type-to-steer: set the prompt directly, bypassing ASR (no debounce)."""
    p = (data or {}).get("prompt", "").strip()
    if p:
        state["current"] = p
        bus.set(compose())
        socketio.emit("heard", {"text": "(typed) " + p})


@socketio.on("stop")
def on_stop():
    state["stop"] = True; state["active"] = False


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5008)
    ap.add_argument("--asr_gpu", type=int, default=1)
    ap.add_argument("--no_compile", action="store_true", help="disable torch.compile (on by default)")
    args = ap.parse_args()
    init_models(asr_gpu=args.asr_gpu, do_compile=not args.no_compile)
    print(f"\n>>> open via:  ssh -L {args.port}:localhost:{args.port} spycoder@<box>  then http://localhost:{args.port}\n")
    socketio.run(app, host="127.0.0.1", port=args.port, allow_unsafe_werkzeug=True)
