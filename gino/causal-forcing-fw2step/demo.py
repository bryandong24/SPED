"""Click-to-generate streaming UI for CAUSAL FORCING++ (FRAME-WISE 2-STEP).

A Flask + Socket.IO server matching the Self-Forcing / Causal-Forcing demo.html contract
(start_generation -> frame_ready / progress / generation_complete), but its generation is
backed by the proven frame-wise StreamingCF rollout (cf_streaming.py) + the full Wan VAE
streaming decode. This avoids the chunk-wise VAEDecoderWrapper block path in the upstream
demo.py, which is hardcoded for 3-frame blocks and does not adapt cleanly to nfpb=1.

The browser buffers frames and plays them back at the chosen FPS. First chunk uses the
4-step ASD schedule, later chunks use 2 steps (handled inside StreamingCF.step).

Launch (one free GPU):
    CUDA_VISIBLE_DEVICES=5 /data/SPED/gino/Self-Forcing/.venv/bin/python demo.py --port 5003
(or run launch_ui.sh)
"""
import os, sys, time, base64, random, argparse
from io import BytesIO
from threading import Event
from PIL import Image
import torch
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf_streaming import StreamingCF, load_cf_pipeline

PIX_PER_LATENT = 4  # Wan VAE temporal factor: 1 latent frame -> ~4 pixel frames @ 16 fps
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5003)
parser.add_argument("--host", type=str, default="0.0.0.0")
args = parser.parse_args()

print("loading Causal Forcing++ framewise-2step (video, cuda:0)...")
pipe = load_cf_pipeline(window=21, sink=3)  # also os.chdir's into the shared CF repo

app = Flask(__name__, template_folder=TEMPLATES)
app.config["SECRET_KEY"] = "cf-fw2step-demo"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

state = {"active": False, "stop": Event()}


def _free_vram_gb():
    try:
        free, _ = torch.cuda.mem_get_info()
        return round(free / 1e9, 1)
    except Exception:
        return 0.0


def _jpeg_data_url(frame_hwc_uint8):
    img = Image.fromarray(frame_hwc_uint8, "RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def generate_video_stream(prompt, seed, duration):
    job_id = str(int(time.time() * 1000))
    try:
        state["active"] = True
        state["stop"].clear()

        def progress(msg, pct):
            socketio.emit("progress", {"message": msg, "progress": pct, "job_id": job_id})

        progress("Encoding prompt...", 5)
        torch.cuda.set_device(0)
        gen = StreamingCF(pipe, seed=seed)
        total = max(gen.nfpb, int(round(duration * PIX_PER_LATENT)))
        total -= total % gen.nfpb
        gen.start(prompt, total_frames=total)
        pipe.vae.model.clear_cache()
        n_chunks = total // gen.nfpb

        progress("Generating frames...", 12)
        frame_index = 0
        t0 = time.time()
        for c in range(n_chunks):
            if state["stop"].is_set():
                break
            den = gen.step()
            frames = gen.decode_chunk(den)
            for f in frames:
                if state["stop"].is_set():
                    break
                socketio.emit("frame_ready", {
                    "data": _jpeg_data_url(f),
                    "frame_index": frame_index,
                    "block_index": c,
                    "job_id": job_id,
                })
                frame_index += 1
            pct = int(12 + 83 * (c + 1) / n_chunks)
            progress(f"Block {c + 1}/{n_chunks}...", pct)
            socketio.sleep(0)  # yield so frames flush to the client promptly

        gen_time = time.time() - t0
        socketio.emit("generation_complete", {
            "message": "Video generation completed!",
            "total_frames": frame_index,
            "generation_time": f"{gen_time:.2f}s",
            "job_id": job_id,
        })
        print(f"[demo] {frame_index} frames in {gen_time:.1f}s = {frame_index/max(gen_time,1e-6):.1f} FPS")
    except Exception as e:
        print(f"generation failed: {e}")
        socketio.emit("error", {"message": f"Generation failed: {e}", "job_id": job_id})
    finally:
        state["active"] = False
        state["stop"].set()


@app.route("/")
def index():
    return render_template("demo.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "generation_active": state["active"],
        "free_vram_gb": _free_vram_gb(),
        "fp8_applied": False,
        "torch_compile_applied": False,
        "current_use_taehv": False,
    })


@socketio.on("connect")
def on_connect():
    emit("status", {"message": "Connected to CF++ framewise-2step demo"})


@socketio.on("start_generation")
def on_start(data):
    if state["active"]:
        emit("error", {"message": "Generation already in progress"})
        return
    prompt = (data or {}).get("prompt", "").strip()
    if not prompt:
        emit("error", {"message": "Prompt is required"})
        return
    seed = int((data or {}).get("seed", -1))
    if seed == -1:
        seed = random.randint(0, 2**31 - 1)
    duration = float((data or {}).get("duration", 5))
    emit("status", {"message": "Generation started"})
    socketio.start_background_task(generate_video_stream, prompt, seed, duration)


@socketio.on("stop_generation")
def on_stop():
    state["stop"].set()
    state["active"] = False
    emit("status", {"message": "Generation stopped"})


if __name__ == "__main__":
    print(f"🚀 CF++ framewise-2step UI on http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
