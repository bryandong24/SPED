"""Build hardcut-vs-recache comparison grids for the 3 CF examples."""
import numpy as np, imageio.v3 as iio
from PIL import Image
import os

OUT = "/data/SPED/gino/out/cf_recache"
EXAMPLES = ["dog", "car", "jungle"]
SWAP_PX = 12 * 4  # swap_frame 12 -> px 48
NCOL = 8


def strip(frames, n=NCOL, w=200, h=116):
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    tiles = []
    for i in idx:
        f = np.array(Image.fromarray(frames[i]).resize((w, h)))
        if i >= SWAP_PX:
            f[:4, :] = [255, 60, 60]  # red stripe = post-swap
        tiles.append(f)
    return np.concatenate(tiles, axis=1)


def label(img, text, h=22):
    from PIL import ImageDraw
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + h), (20, 20, 20))
    canvas.paste(Image.fromarray(img), (0, h))
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), text, fill=(255, 255, 255))
    return np.array(canvas)


for ex in EXAMPLES:
    rows = []
    for mode in ["hardcut", "recache"]:
        mp4 = f"{OUT}/{ex}_{mode}.mp4"
        if not os.path.exists(mp4):
            print("MISSING", mp4); continue
        v = iio.imread(mp4)
        warm = (v[..., 0].astype(float) - v[..., 2].astype(float)).mean(axis=(1, 2))
        pre = warm[:SWAP_PX].mean(); post = warm[SWAP_PX:].mean(); final = warm[-10:].mean()
        print(f"{ex:7s} {mode:8s}: warmth pre={pre:6.1f} post={post:6.1f} final={final:6.1f}")
        rows.append(label(strip(v), f"{ex}  {mode.upper()}  (red=after swap @3s)  warmth {pre:.0f}->{final:.0f}"))
    if rows:
        grid = np.concatenate(rows, axis=0)
        Image.fromarray(grid).save(f"{OUT}/compare_{ex}.png")
        print("  saved", f"{OUT}/compare_{ex}.png")

# also a combined 3-example recache-only and hardcut-only montage? keep per-example.
print("DONE")
