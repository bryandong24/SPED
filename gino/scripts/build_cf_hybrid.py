"""Build idea-1 (delay) and idea-2 (lookback sweep) comparison grids per example."""
import numpy as np, imageio.v3 as iio, os
from PIL import Image, ImageDraw

HY = "/data/SPED/gino/out/cf_hybrid"
RC = "/data/SPED/gino/out/cf_recache"
EXAMPLES = ["dog", "car", "jungle"]
SWAP_PX = 12 * 4  # 48
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


def warm_label(v):
    w = (v[..., 0].astype(float) - v[..., 2].astype(float)).mean(axis=(1, 2))
    return w[:SWAP_PX].mean(), w[-10:].mean()


def row(path, text):
    if not os.path.exists(path):
        print("MISSING", path); return None
    v = iio.imread(path)
    pre, fin = warm_label(v)
    s = strip(v)
    canvas = Image.new("RGB", (s.shape[1], s.shape[0] + 22), (20, 20, 20))
    canvas.paste(Image.fromarray(s), (0, 22))
    ImageDraw.Draw(canvas).text((6, 5), f"{text}   warmth {pre:.0f} -> {fin:.0f}", fill=(255, 255, 255))
    return np.array(canvas)


for ex in EXAMPLES:
    # Idea 1 — delay spectrum (hardcut = delay infinity ... rc9 = delay 0)
    rows = [
        row(f"{RC}/{ex}_hardcut.mp4", f"{ex}  HARD-CUT (no recache)"),
        row(f"{HY}/{ex}_delay6.mp4",  f"{ex}  hard-cut 6f -> recache"),
        row(f"{HY}/{ex}_delay4.mp4",  f"{ex}  hard-cut 4f -> recache"),
        row(f"{HY}/{ex}_delay2.mp4",  f"{ex}  hard-cut 2f -> recache"),
        row(f"{HY}/{ex}_rc9.mp4",     f"{ex}  IMMEDIATE recache (delay 0)"),
    ]
    rows = [r for r in rows if r is not None]
    Image.fromarray(np.concatenate(rows, axis=0)).save(f"{HY}/idea1_delay_{ex}.png")
    print("saved", f"idea1_delay_{ex}.png")

    # Idea 2 — recache lookback sweep
    rows = [row(f"{RC}/{ex}_hardcut.mp4", f"{ex}  HARD-CUT (ref)")]
    for n in (3, 6, 9, 15, 21):
        rows.append(row(f"{HY}/{ex}_rc{n}.mp4", f"{ex}  recache N={n} frames"))
    rows = [r for r in rows if r is not None]
    Image.fromarray(np.concatenate(rows, axis=0)).save(f"{HY}/idea2_sweep_{ex}.png")
    print("saved", f"idea2_sweep_{ex}.png")

print("DONE")
