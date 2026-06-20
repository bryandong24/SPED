"""Live browser demo — voice (push-to-talk) + text -> Gemini -> camera control of minWM.

Architecture (adapted from gino/audio_stream/web_live.py):
  browser (live.html): video <img>, push-to-talk mic, text steer box
    --socket.io--> server.py
      - "start"      -> bootstrap + launch gen_loop + emitter_loop
      - "utterance"  -> 16kHz PCM blob -> wav -> VoicePlanner(audio) -> CommandBus
      - "command"    -> text          ->        VoicePlanner(text)  -> CommandBus
      - gen_loop     -> each chunk: CommandBus.snapshot() -> worker.step() -> frame_q
      - emitter_loop -> frame_q -> base64 JPEG -> emit "frame" @<=16fps (backpressure)

Run from the minWM repo root with the launcher env (single GPU, no torchrun):
  cd "/mnt/data/SPED/controllable world model/minWM"
  export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python Wan21/live/server.py --port 5008
Reach it from a laptop:  ssh -L 5008:localhost:5008 <box>  then http://localhost:5008
"""
import argparse
import base64
import io
import os
import queue
import sys
import tempfile
import threading
import time
import wave

_HERE = os.path.dirname(os.path.abspath(__file__))
_WAN21 = os.path.dirname(_HERE)
for _p in (_HERE, _WAN21):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
from PIL import Image

from planner import VoicePlanner   # workers imported lazily in __main__ (heavy)

DEFAULT_PROMPT = ("A first-person walk down a sunlit forest path, tall green trees, "
                  "dappled light, cinematic, photorealistic, highly detailed, 4k")
MAX_STEPS = int(os.environ.get("MAX_STEPS", 2000))

BACKBONE = "wan"        # "wan" (T2V) or "hy" (I2V, image+caption seed) — set in __main__
EXAMPLES = []           # HY seed examples loaded from assets/example.json
worker = None
planner = VoicePlanner()


class CommandBus:
    """Thread-safe latest-wins camera target + prompt delta."""

    def __init__(self):
        self._lock = threading.Lock()
        self.camera = {"forward": 0.0, "strafe": 0.0, "turn": 0.0, "pitch": 0.0, "up": 0.0, "speed": 1.0}
        self.prompt_delta = ""
        self.version = 0

    def update(self, cmd):
        with self._lock:
            self.camera = dict(cmd.get("camera", self.camera))
            if cmd.get("prompt_delta"):
                self.prompt_delta = cmd["prompt_delta"]
            self.version += 1

    def reset(self):
        with self._lock:
            self.camera = {"forward": 0.0, "strafe": 0.0, "turn": 0.0, "pitch": 0.0, "up": 0.0, "speed": 1.0}
            self.prompt_delta = ""
            self.version = 0

    def snapshot(self):
        with self._lock:
            return dict(self.camera), self.prompt_delta, self.version


bus = CommandBus()
state = {"active": False, "stop": False, "seed": DEFAULT_PROMPT}
frame_q = queue.Queue(maxsize=64)       # decode -> emitter
latent_q = queue.Queue(maxsize=4)       # gen -> decode (2-GPU pipeline backpressure)

app = Flask(__name__, template_folder=os.path.join(_HERE, "templates"))
socketio = SocketIO(app, cors_allowed_origins="*",
                    max_http_buffer_size=64 * 1024 * 1024, async_mode="threading")


def compose(prompt_delta):
    return state["seed"] if not prompt_delta else f"{state['seed']}. {prompt_delta}"


@app.route("/")
def index():
    return render_template("live.html")


def _push_frames(frames):
    for f in frames:
        while state["active"] and not state["stop"]:
            try:
                frame_q.put(f, timeout=0.5)
                break
            except queue.Full:
                continue


def gen_loop():
    """Producer on the generator GPU: snapshot the command bus, generate one chunk's
    latents, hand them to the decode pipeline. Caches persist across chunks.

    Wrapped so any failure resets state["active"] and tells the browser — otherwise a crash
    here leaves active=True with no producer, which freezes the last frame and makes every
    later Start a silent no-op (the "stuck on the old scene" zombie)."""
    try:
        socketio.emit("status", {"msg": "starting scene… (first chunk warms up)"})
        if BACKBONE == "hy":
            worker.start(state["image"], state["seed"])   # I2V: seed image + caption
        else:
            worker.start(state["seed"])                   # T2V: text prompt
        socketio.emit("status", {"msg": "live — steer with keys, voice, or text"})
        step = 0
        while state["active"] and not state["stop"] and step < MAX_STEPS:
            cam, pd, _ = bus.snapshot()
            worker.set_prompt(compose(pd))
            t0 = time.time()
            lat = worker.gen_step(cam)
            socketio.emit("tick", {"camera": cam, "prompt": worker.prompt, "step": step,
                                   "sec": round(time.time() - t0, 2)})
            while state["active"] and not state["stop"]:
                try:
                    latent_q.put(lat, timeout=0.5)
                    break
                except queue.Full:
                    continue
            step += 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        socketio.emit("status", {"msg": f"generation error: {e} — press Start to retry"})
    finally:
        latent_q.put(None)
        state["active"] = False
        state["stop"] = False


def decode_loop():
    """Consumer on the VAE GPU: decode each chunk's new frames (overlaps with gen)."""
    while True:
        try:
            lat = latent_q.get(timeout=1.0)
        except queue.Empty:
            if not state["active"]:
                break
            continue
        if lat is None:
            break
        _push_frames(worker.decode_step(lat))
    frame_q.put(None)


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
        Image.fromarray(f).save(b, format="JPEG", quality=80)
        socketio.emit("frame", {"img": base64.b64encode(b.getvalue()).decode()})
        dt = min_period - (time.time() - t0)
        if dt > 0:
            socketio.sleep(dt)
    socketio.emit("status", {"msg": "done"})


@socketio.on("connect")
def on_connect():
    """Tell the browser which backbone is live (and HY seed images, if any)."""
    socketio.emit("config", {"backbone": BACKBONE, "default_prompt": DEFAULT_PROMPT,
                             "examples": EXAMPLES})


@socketio.on("start")
def on_start(data):
    if state["active"]:
        return
    data = data or {}
    if BACKBONE == "hy":
        idx = int(data.get("example_idx", 0))
        idx = max(0, min(idx, len(EXAMPLES) - 1))
        state["image"] = EXAMPLES[idx]["image_path"]
        state["seed"] = (data.get("caption") or EXAMPLES[idx]["caption"]).strip()
    else:
        state["seed"] = (data.get("prompt") or DEFAULT_PROMPT).strip()
    bus.reset()
    for q in (frame_q, latent_q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
    state["active"] = True
    state["stop"] = False
    socketio.emit("parsed", {"src": "seed", "camera": bus.snapshot()[0], "intent": "none",
                             "prompt_delta": "", "transcript": ""})
    socketio.start_background_task(gen_loop)      # generator GPU
    socketio.start_background_task(decode_loop)   # VAE GPU (overlaps with gen)
    socketio.start_background_task(emitter_loop)  # frames -> browser


@socketio.on("command")
def on_command(data):
    text = (data or {}).get("text", "").strip()
    if not text:
        return
    cmd = planner.plan(text=text, state=bus.snapshot()[0])
    bus.update(cmd)
    socketio.emit("parsed", {"src": "text: " + text, **cmd})


@socketio.on("direct")
def on_direct(data):
    """Keyboard / direct camera control — bypasses Gemini entirely.

    The browser sends the raw velocity state from held keys; we clamp and push it
    straight to the command bus (same path the worker samples each chunk)."""
    src = (data or {}).get("camera") or {}
    cam = {"forward": 0.0, "strafe": 0.0, "turn": 0.0, "pitch": 0.0, "up": 0.0, "speed": 1.0}
    for k in ("forward", "strafe", "turn", "pitch", "up"):
        try:
            cam[k] = max(-1.0, min(1.0, float(src.get(k, 0.0))))
        except (TypeError, ValueError):
            pass
    try:
        cam["speed"] = max(0.0, min(2.0, float(src.get("speed", 1.0))))
    except (TypeError, ValueError):
        pass
    bus.update({"camera": cam})
    socketio.emit("parsed", {"src": "keyboard " + ((data or {}).get("keys") or "(idle)"),
                             "camera": cam, "intent": "move", "prompt_delta": "", "transcript": ""})


@socketio.on("utterance")
def on_utterance(data):
    try:
        pcm = base64.b64decode(data["pcm"])
        if len(pcm) < 3200:  # < ~0.1s of 16kHz int16 -> ignore stray taps
            return
        path = os.path.join(tempfile.gettempdir(), f"utt_{int(time.time() * 1000)}.wav")
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm)
        cmd = planner.plan(audio_path=path, state=bus.snapshot()[0])
        bus.update(cmd)
        socketio.emit("parsed", {"src": "voice: " + (cmd.get("transcript") or "(heard)"), **cmd})
    except Exception as e:
        socketio.emit("parsed", {"src": f"voice error: {e}", "camera": bus.snapshot()[0],
                                 "intent": "none", "prompt_delta": "", "transcript": ""})


@socketio.on("stop")
def on_stop():
    state["stop"] = True
    state["active"] = False


def _load_hy_examples(example_json="assets/example.json"):
    import json
    base = os.path.dirname(os.path.abspath(example_json))
    out = []
    for i, e in enumerate(json.load(open(example_json))):
        if not e.get("trajectory"):
            continue
        img = e["image"]
        out.append({"idx": i, "caption": e["caption"],
                    "image": img,
                    "image_path": img if os.path.isabs(img) else os.path.join(base, img)})
    return out


if __name__ == "__main__":
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["wan", "hy"], default="wan",
                    help="wan = Wan2.1 T2V (text seed); hy = HunyuanVideo-1.5 I2V (image+caption seed)")
    ap.add_argument("--port", type=int, default=5008)
    ap.add_argument("--gen_device", default=os.environ.get("GEN_DEVICE", "cuda:0"))
    ap.add_argument("--vae_device", default=os.environ.get("VAE_DEVICE", "cuda:1"),
                    help="separate GPU for VAE decode (overlaps with gen); == gen_device for 1 GPU")
    # Wan-only
    ap.add_argument("--config_path", default="Wan21/configs/causal_forcing_dmd_camera.yaml")
    ap.add_argument("--checkpoint_path", default="./ckpts/Wan21/Action2V/dmd/model.pt")
    ap.add_argument("--compile", action="store_true")
    # HY-only
    ap.add_argument("--transformer_dir", default="./ckpts/HY15/Action2V/dmd")
    ap.add_argument("--cap", type=int, default=24, help="HY vision-KV window (latent frames); "
                    "smaller = faster steady-state gen, shorter temporal memory")
    ap.add_argument("--sink", type=int, default=1,
                    help="HY seed-anchor sink kept inside the bounded vision-KV window")
    args = ap.parse_args()

    BACKBONE = args.backbone
    vae_device = args.vae_device
    if torch.cuda.device_count() < 2 or vae_device in ("", "none", args.gen_device):
        vae_device = args.gen_device  # single-GPU fallback

    if BACKBONE == "hy":
        EXAMPLES = _load_hy_examples()
        DEFAULT_PROMPT = EXAMPLES[0]["caption"] if EXAMPLES else DEFAULT_PROMPT
        from hy_worker import HYStreamingWorker
        print(f"loading HY (HunyuanVideo-1.5) worker (gen={args.gen_device} vae={vae_device})…")
        worker = HYStreamingWorker(transformer_dir=args.transformer_dir,
                                   gen_device=args.gen_device, vae_device=vae_device,
                                   max_vision_frames=args.cap,
                                   sink_vision_frames=args.sink)
    else:
        from streaming_worker import MinWMStreamingWorker
        print(f"loading Wan worker (gen={args.gen_device} vae={vae_device}, compile={args.compile})…")
        worker = MinWMStreamingWorker(args.config_path, args.checkpoint_path,
                                      gen_device=args.gen_device, vae_device=vae_device,
                                      compile=args.compile)
    print(f"\n>>> backbone={BACKBONE}  open via:  ssh -L {args.port}:localhost:{args.port} <box>  "
          f"then http://localhost:{args.port}\n")
    socketio.run(app, host="127.0.0.1", port=args.port, allow_unsafe_werkzeug=True)
