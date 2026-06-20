"""
Realtime(-ish) interactive GameCraft web demo.

Launched under torchrun (N ranks = N-way sequence parallel). ALL ranks run the
autoregressive chunk loop in lockstep; RANK 0 additionally runs a Flask-SocketIO
server that (a) takes keyboard input from the browser, (b) broadcasts the current
action+prompt to every rank before each chunk, (c) streams the decoded frames back
to the browser. Generation continues continuously; the held key chooses the next
chunk's camera action; the prompt box does a mid-rollout prompt swap.

Run via realtime/run_play.sh (sets the model args + GPUS/H/W).
"""
import os, io, time, base64, threading, queue, random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from PIL import Image
from loguru import logger

from hymm_sp.config import parse_args
from hymm_sp.sample_inference import HunyuanVideoSampler
from hymm_sp.modules.parallel_states import initialize_distributed, nccl_info

# ----- valid GameCraft camera actions (see ACTION_MAP in the gradio app) -----
VALID_ACTIONS = {"w", "a", "s", "d", "up_rot", "down_rot", "left_rot", "right_rot"}


class CropResize:
    def __init__(self, size):
        self.target_h, self.target_w = size

    def __call__(self, img):
        w, h = img.size
        scale = max(self.target_w / w, self.target_h / h)
        img = transforms.Resize((int(h * scale), int(w * scale)),
                                interpolation=transforms.InterpolationMode.BILINEAR)(img)
        return transforms.CenterCrop((self.target_h, self.target_w))(img)


class ActionBus:
    """Thread-safe holder for the currently-held action + pending prompt swap."""
    def __init__(self, default_action, default_speed, prompt):
        self._lock = threading.Lock()
        self.action = default_action
        self.speed = default_speed
        self.prompt = prompt
        self.prompt_version = 0
        self.stop = False
        # history-attenuation levers (1.0 = stock; lower = weaken old-scene anchor so swaps take)
        self.hist = 1.0
        self.gt = 1.0
        self.mask = 1.0

    def set_levers(self, hist=None, gt=None, mask=None):
        with self._lock:
            if hist is not None: self.hist = float(hist)
            if gt is not None: self.gt = float(gt)
            if mask is not None: self.mask = float(mask)

    def set_action(self, action, speed):
        with self._lock:
            if action in VALID_ACTIONS:
                self.action = action
            if speed is not None:
                self.speed = float(speed)

    def set_prompt(self, prompt):
        with self._lock:
            if prompt and prompt.strip():
                self.prompt = prompt.strip()
                self.prompt_version += 1

    def request_stop(self):
        with self._lock:
            self.stop = True

    def snapshot(self):
        with self._lock:
            return dict(action=self.action, speed=self.speed,
                        prompt=self.prompt, pv=self.prompt_version, stop=self.stop,
                        hist=self.hist, gt=self.gt, mask=self.mask)


def main():
    args = parse_args()
    initialize_distributed(args.seed)
    rank = int(os.getenv("RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', '0'))}")
    torch.cuda.set_device(device)

    H, W = int(args.video_size[0]), int(args.video_size[1])
    seed = args.seed if args.seed else 250160
    init_prompt = args.prompt
    neg_prompt = args.add_neg_prompt
    default_action = os.environ.get("DEFAULT_ACTION", "w")
    default_speed = float(os.environ.get("DEFAULT_SPEED", "0.2"))
    port = int(os.environ.get("PORT", "8080"))

    logger.info(f"[rank{rank}/{world}] loading GameCraft from {args.ckpt} @ {H}x{W}")
    sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)
    args = sampler.args

    # ---- encode the start image -> initial last_latents / ref_latents ----
    tf = transforms.Compose([
        CropResize((H, W)), transforms.CenterCrop((H, W)),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    raw_img = Image.open(args.image_path).convert("RGB")
    px = tf(raw_img).unsqueeze(0).unsqueeze(2).to(device)  # (1,C,1,H,W)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        sampler.pipeline.vae.enable_tiling()
        last_latents = sampler.vae.encode(px).latent_dist.sample().to(dtype=torch.float16)
        last_latents.mul_(sampler.vae.config.scaling_factor)
        ref_latents = last_latents.clone()
        sampler.pipeline.vae.disable_tiling()
    # keep all ranks identical
    if world > 1:
        dist.broadcast(last_latents, src=0)
        dist.broadcast(ref_latents, src=0)
    ref_images = [raw_img]

    bus = ActionBus(default_action, default_speed, init_prompt) if rank == 0 else None
    frame_q = queue.Queue(maxsize=256) if rank == 0 else None
    state = {"socketio": None, "n_chunks": 0}

    # ---------------- rank-0 web server ----------------
    if rank == 0:
        from flask import Flask, render_template
        from flask_socketio import SocketIO
        app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
        socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                            max_http_buffer_size=64 * 1024 * 1024)
        state["socketio"] = socketio

        @app.route("/")
        def index():
            return render_template("play.html", W=W, H=H, prompt=init_prompt)

        @socketio.on("action")
        def on_action(data):
            data = data or {}
            bus.set_action(data.get("id"), data.get("speed"))

        @socketio.on("set_prompt")
        def on_set_prompt(data):
            p = (data or {}).get("prompt", "")
            bus.set_prompt(p)
            socketio.emit("swapped", {"prompt": p})

        @socketio.on("levers")
        def on_levers(data):
            data = data or {}
            bus.set_levers(data.get("hist"), data.get("gt"), data.get("mask"))
            socketio.emit("levers_set", {"hist": bus.hist, "gt": bus.gt, "mask": bus.mask})

        @socketio.on("stop")
        def on_stop(_=None):
            bus.request_stop()

        def emitter():
            while True:
                item = frame_q.get()
                if item is None:
                    break
                socketio.emit(item[0], item[1])
        threading.Thread(target=emitter, daemon=True).start()
        threading.Thread(
            target=lambda: socketio.run(app, host="0.0.0.0", port=port,
                                        allow_unsafe_werkzeug=True),
            daemon=True).start()
        logger.info(f"[rank0] web UI on http://0.0.0.0:{port}")

    # ---------------- shared AR generation loop ----------------
    chunk_idx = 0
    while True:
        # rank0 decides the control for this chunk; broadcast to all ranks
        if rank == 0:
            ctrl = bus.snapshot()
        else:
            ctrl = None
        if world > 1:
            box = [ctrl]
            dist.broadcast_object_list(box, src=0)
            ctrl = box[0]
        if ctrl["stop"]:
            break

        # apply history-attenuation levers on every rank before the chunk
        os.environ["GC_HIST_SCALE"] = str(ctrl.get("hist", 1.0))
        os.environ["GC_GT_SCALE"] = str(ctrl.get("gt", 1.0))
        os.environ["GC_MASK_SCALE"] = str(ctrl.get("mask", 1.0))

        t0 = time.time()
        outputs = sampler.predict(
            prompt=ctrl["prompt"],
            action_id=ctrl["action"],
            action_speed=ctrl["speed"],
            is_image=(chunk_idx == 0),
            size=(H, W),
            seed=seed,
            last_latents=last_latents,
            ref_latents=ref_latents,
            video_length=args.sample_n_frames,
            guidance_scale=args.cfg_scale,
            num_images_per_prompt=args.num_images,
            negative_prompt=neg_prompt,
            infer_steps=args.infer_steps,
            flow_shift=args.flow_shift_eval_video,
            use_linear_quadratic_schedule=args.use_linear_quadratic_schedule,
            linear_schedule_end=args.linear_schedule_end,
            use_deepcache=args.use_deepcache,
            cpu_offload=False,
            ref_images=ref_images,
            output_dir="/tmp/gc_play",
            return_latents=True,
            use_sage=args.use_sage,
        )
        last_latents = outputs["last_latents"]
        ref_latents = outputs["ref_latents"]
        dt = time.time() - t0

        if rank == 0:
            vid = outputs["samples"][0][0]  # (C,F,H,W) in [0,1]
            nf = vid.shape[1]
            arr = (vid.permute(1, 2, 3, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)  # (F,H,W,C)
            for i in range(nf):
                b = io.BytesIO()
                Image.fromarray(arr[i]).save(b, format="JPEG", quality=80)
                frame_q.put(("frame", {"img": base64.b64encode(b.getvalue()).decode()}))
            gen_fps = nf / dt
            frame_q.put(("chunk_done", {
                "chunk": chunk_idx, "gen_s": round(dt, 2), "gen_fps": round(gen_fps, 2),
                "nframes": nf, "action": ctrl["action"], "speed": ctrl["speed"],
                "prompt": ctrl["prompt"]}))
            logger.info(f"chunk {chunk_idx}: {nf}f in {dt:.1f}s = {gen_fps:.2f} gen-FPS | "
                        f"action={ctrl['action']} speed={ctrl['speed']}")
        chunk_idx += 1


if __name__ == "__main__":
    main()
