"""LIVE audio-steered video demo on CAUSAL FORCING++ (FRAME-WISE 2-STEP).

Lightweight sibling of gino/audio_stream/web_live_cf.py (chunk-wise 4-step). Browser mic
-> socket.io -> faster-whisper ASR -> Gemini prompt refinement -> PromptBus -> a continuous
StreamingCF rollout (frame-wise 2-step, 4-step first chunk), streaming decoded frames back as
JPEGs. Prompt switching is a smooth forward SLERP ramp (ramp_to), with Gemini-enriched prompts.

Reach from a laptop:
    ssh -L 5013:localhost:5013 <user>@<box>
    open http://localhost:5013
Launch (video=cuda:0, ASR=cuda:1 within CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=5,6 /data/SPED/gino/Self-Forcing/.venv/bin/python web_live_cf.py --port 5013 --asr_gpu 1
(or just run launch_live.sh)
"""
import os, sys, time, re, base64, io, queue, subprocess
import numpy as np
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from cf_streaming import StreamingCF, load_cf_pipeline, PromptBus
from live_pipeline import RollingBuffer, PromptDebouncer, SR

MAX_SECONDS = 90  # run length cap (note: CF RoPE drifts past ~30s — quality degrades late)
RAMP_CHUNKS = 4   # forward conditioning-ramp horizon (chunks) on a prompt change
PIX_PER_LATENT = 4  # Wan VAE temporal factor: 1 latent frame -> ~4 pixel frames @ 16 fps

GEM_PY = "/data/SPED/gemini/.venv/bin/python"
GEM_REFINE = "/data/SPED/gemini/refine_prompt.py"


def refine_prompt(text, timeout=8):
    """Expand a rough prompt into a rich cinematic one via Gemini 3.1 Flash-Lite
    (subprocess to the gemini venv). Falls back to the original on any failure."""
    text = (text or "").strip()
    if not text:
        return text
    try:
        r = subprocess.run([GEM_PY, GEM_REFINE], input=text, capture_output=True,
                           text=True, timeout=timeout)
        out = (r.stdout or "").strip()
        print(f"[refine] in={text[:45]!r} -> out={out[:60]!r}", flush=True)
        return out if out else text
    except Exception as e:
        print("refine err:", e); return text


DEFAULT_START = ("A fluffy golden retriever with a glossy honey-gold coat and a red leather collar "
                 "bounds through a vast sunlit meadow of orange and yellow wildflowers, captured in "
                 "crisp HDR 4K with razor-sharp fur detail. Warm golden afternoon light rakes across "
                 "the lush green grass and pollen drifts glowing in the air. The low camera tracks fast "
                 "alongside the dog at a full sprint, tongue out and ears flapping, petals scattering in "
                 "its wake, rolling green hills and a bright blue sky beyond, cinematic and photorealistic.")
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

pipe = None
asr_model = None
bus = PromptBus(initial=DEFAULT_START)
buf = RollingBuffer(seconds=8.0)
state = {"active": False, "stop": False, "seed": DEFAULT_START, "current": ""}
frame_q = queue.Queue(maxsize=32)

app = Flask(__name__, template_folder=TEMPLATES)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=32 * 1024 * 1024,
                    async_mode="threading")


def compose():
    # full-prompt replacement: a steer REPLACES the whole prompt (no fixed-seed anchoring)
    return state["current"] or state["seed"]


def init_models(asr_gpu=1, do_compile=False):
    global pipe, asr_model
    print("loading Causal Forcing++ framewise-2step (video, cuda:0)...")
    pipe = load_cf_pipeline(window=21, sink=3)
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


def asr_loop():
    # Commit on the first stable tick and only reject near-duplicates (jaccard 0.85),
    # so same-subject scene changes aren't suppressed.
    deb = PromptDebouncer(jaccard=0.85, stable_ticks=1)
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
        socketio.emit("asr_live", {"raw": text, "window_s": round(len(a) / SR, 1)})
        sents = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
        cand = sents[-1] if sents else text
        committed = deb.update(cand)
        if committed:
            rp = refine_prompt(committed)            # Gemini-enrich the spoken steer
            state["current"] = rp
            bus.set(compose())
            socketio.emit("embedded", {"prompt": rp, "seed": state["seed"], "current": rp})


def gen_loop(start_prompt):
    torch.cuda.set_device(0)
    gen = StreamingCF(pipe, seed=int(time.time()) % 100000)
    # frame-wise: each chunk is nfpb latent frames = nfpb/PIX_PER_LATENT seconds of video.
    chunk_seconds = gen.nfpb / float(PIX_PER_LATENT)
    n_chunks = int(np.ceil(MAX_SECONDS / chunk_seconds))
    gen.start(start_prompt, total_frames=n_chunks * gen.nfpb)
    pipe.vae.model.clear_cache()
    cur = start_prompt
    socketio.emit("status", {"msg": "generating"})
    socketio.emit("embedded", {"prompt": compose(), "seed": state["seed"], "current": state["current"]})
    for c in range(n_chunks):
        if state["stop"]:
            break
        p, _ = bus.get()
        if p and p != cur:
            gen.ramp_to(p, k=RAMP_CHUNKS); cur = p   # FORWARD RAMP (no recache)
            socketio.emit("embedded", {"prompt": p, "seed": state["seed"], "current": state["current"]})
            socketio.emit("steer", {"text": p})
        den = gen.step()
        frames = gen.decode_chunk(den)
        for f in frames:
            while state["active"] and not state["stop"]:
                try:
                    frame_q.put(f, timeout=0.5); break
                except queue.Full:
                    continue
    frame_q.put(None)
    state["active"] = False


def emitter_loop():
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
        Image.fromarray(f).save(b, format="JPEG", quality=92)
        socketio.emit("frame", {"img": base64.b64encode(b.getvalue()).decode()})
        dt = min_period - (time.time() - t0)
        if dt > 0:
            socketio.sleep(dt)
    socketio.emit("status", {"msg": "done"})


@socketio.on("start")
def on_start(data):
    if state["active"]:
        return
    raw = (data or {}).get("prompt") or DEFAULT_START
    socketio.emit("status", {"msg": "refining prompt with Gemini…"})
    start_prompt = refine_prompt(raw)               # Gemini-enrich the initial prompt
    state["seed"] = start_prompt
    state["current"] = ""
    bus.set(compose())
    socketio.emit("embedded", {"prompt": start_prompt, "seed": start_prompt, "current": ""})
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
    p = (data or {}).get("prompt", "").strip()
    if p:
        rp = refine_prompt(p)                        # Gemini-enrich the typed steer
        state["current"] = rp
        bus.set(compose())
        socketio.emit("heard", {"text": "(typed) " + p})
        socketio.emit("embedded", {"prompt": rp, "seed": state["seed"], "current": rp})


@socketio.on("stop")
def on_stop():
    state["stop"] = True; state["active"] = False


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5013)
    ap.add_argument("--asr_gpu", type=int, default=1)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    init_models(asr_gpu=args.asr_gpu, do_compile=args.compile)
    print(f"\n>>> ssh -L {args.port}:localhost:{args.port} <user>@<box>  then http://localhost:{args.port}\n")
    socketio.run(app, host="127.0.0.1", port=args.port, allow_unsafe_werkzeug=True)
