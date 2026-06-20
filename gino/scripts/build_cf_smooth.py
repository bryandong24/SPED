"""Build per-example comparison grids for the Tier-1 smooth-transition ablation.

Rows (top->bottom): hardcut, recache, coral, kalman, ramp, smooth_all — so the
contribution of each method (and the full stack) is visible at a glance. Red stripe
marks post-swap frames; each row labelled with warmth pre->final and a luma-jerk
smoothness proxy (lower = smoother). Run after run_cf_smooth.sh (CPU-only, no GPU)."""
import os
import numpy as np, imageio.v3 as iio
from PIL import Image, ImageDraw

OUT = "/data/SPED/gino/out/cf_smooth"
EXAMPLES = ["dog", "car", "jungle"]
ORDER = ["hardcut", "recache", "coral", "kalman", "ramp", "smooth_all"]
SWAP_PX = 12 * 4   # swap_frame 12 -> px 48
NCOL = 8


def strip(v, n=NCOL, w=200, h=116):
    idx = np.linspace(0, len(v) - 1, n).round().astype(int)
    tiles = []
    for i in idx:
        f = np.array(Image.fromarray(v[i]).resize((w, h)))
        if i >= SWAP_PX:
            f[:4, :] = [255, 60, 60]
        tiles.append(f)
    return np.concatenate(tiles, axis=1)


def row(path, label):
    if not os.path.exists(path):
        print("MISSING", path); return None
    v = iio.imread(path)
    warm = (v[..., 0].astype(float) - v[..., 2].astype(float)).mean(axis=(1, 2))
    luma = v.reshape(len(v), -1).mean(axis=1)
    jerk = float(np.abs(np.diff(luma, n=2)).mean()) if len(luma) > 2 else 0.0
    s = strip(v)
    canvas = Image.new("RGB", (s.shape[1], s.shape[0] + 22), (20, 20, 20))
    canvas.paste(Image.fromarray(s), (0, 22))
    ImageDraw.Draw(canvas).text((6, 5), f"{label}   warmth {warm[:SWAP_PX].mean():.0f}->{warm[-10:].mean():.0f}   jerk {jerk:.2f}", fill=(255, 255, 255))
    return np.array(canvas)


for ex in EXAMPLES:
    rows = [row(f"{OUT}/{ex}_{m}.mp4", f"{ex}  {m}") for m in ORDER]
    rows = [r for r in rows if r is not None]
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0)).save(f"{OUT}/compare_{ex}.png")
        print("saved", f"{OUT}/compare_{ex}.png")
print("DONE")
