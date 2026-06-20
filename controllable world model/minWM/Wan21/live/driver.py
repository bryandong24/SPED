"""Headless scripted driver for the minWM live worker (Stages 0-2).

Bootstraps a scene, then steps through a scripted camera timeline (and, with --plan,
through Gemini-parsed text/voice commands), writes a concatenated mp4 + contact sheet,
and prints per-step latency. Run from the minWM repo root with the launcher env:

  cd "/mnt/data/SPED/controllable world model/minWM"
  export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python Wan21/live/driver.py --steps 6
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_WAN21 = os.path.dirname(_HERE)
for _p in (_HERE, _WAN21):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from torchvision.io import write_video

from worker import MinWMWorker

# A scripted camera timeline exercising each axis (one entry applied per step).
TIMELINE = [
    {"forward": 1.0, "speed": 1.0},                 # dolly forward
    {"forward": 1.0, "speed": 1.0},
    {"turn": -1.0, "speed": 1.0},                   # yaw left
    {"turn": -1.0, "speed": 1.0},
    {"pitch": 1.0, "speed": 1.0},                   # look up
    {"strafe": 1.0, "speed": 1.0},                  # strafe right
    {"forward": 0.0, "speed": 0.0},                 # stop / hold
    {"forward": 0.0, "speed": 0.0},
]

DEFAULT_PROMPT = ("A first-person walk down a sunlit forest path, tall green trees, "
                  "dappled light, cinematic, photorealistic, highly detailed, 4k")


def contact_sheet(frames, path, cols=12):
    try:
        from PIL import Image
    except Exception:
        return
    n = frames.shape[0]
    idx = np.linspace(0, n - 1, cols).round().astype(int)
    tiles = [np.array(Image.fromarray(frames[i]).resize((166, 96))) for i in idx]
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="Wan21/configs/causal_forcing_dmd_camera.yaml")
    ap.add_argument("--checkpoint_path", default="./ckpts/Wan21/Action2V/dmd/model.pt")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out", default="./outputs/live/driver.mp4")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--n", type=int, default=16, help="bootstrap latent frames")
    ap.add_argument("--K", type=int, default=4, help="new latent frames per step")
    ap.add_argument("--W", type=int, default=12, help="context window latent frames")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    w = MinWMWorker(args.config_path, args.checkpoint_path)
    print(f"[driver] model loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    frames = [w.bootstrap(args.prompt, n=args.n, seed=args.seed, profile=True)]
    print(f"[driver] bootstrap: {frames[0].shape} in {time.time() - t0:.1f}s "
          f"(chunk0_latency={w.pipeline.last_chunk0_latency})")

    for i in range(args.steps):
        cs = TIMELINE[i % len(TIMELINE)]
        t0 = time.time()
        f = w.step(cs, K=args.K, W=args.W, profile=(i == 0))
        print(f"[driver] step {i}: cs={cs} -> {f.shape} in {time.time() - t0:.2f}s "
              f"(chunk0_latency={w.pipeline.last_chunk0_latency:.3f}s)")
        frames.append(f)

    allf = np.concatenate(frames, axis=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_video(args.out, torch.from_numpy(allf), fps=w.fps)
    contact_sheet(allf, args.out.replace(".mp4", "_sheet.png"))
    print(f"[driver] wrote {args.out} ({allf.shape[0]} frames @ {w.fps}fps)")


if __name__ == "__main__":
    main()
