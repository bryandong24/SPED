"""Per-chunk warmth/brightness for a GameCraft swap video + contact sheet.
Usage: analyze_mp4.py <mp4> <swap_action_idx> <frames_per_chunk> <out_sheet.png>"""
import sys, math
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

mp4, swap_at, fpc, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rd = imageio.get_reader(mp4)
frames = [f for f in rd]
n = len(frames)
print(f"{mp4}: {n} frames, ~{fpc}/chunk, swap at action {swap_at} (~frame {swap_at*fpc})")
print("chunk | warmth | bright")
cells = []
nch = math.ceil(n / fpc)
for c in range(nch):
    seg = np.stack(frames[c*fpc:(c+1)*fpc]).astype(np.float32)
    if seg.size == 0: continue
    warm = seg[..., 0].mean() - seg[..., 2].mean()
    bright = seg.mean()
    mark = " <-- swap" if c == swap_at else ""
    print(f"  {c:3d} | {warm:6.1f} | {bright:6.1f}{mark}")
    cells.append((c, frames[min((c+1)*fpc-1, n-1)], warm))
# contact sheet
h, w, _ = cells[0][1].shape
cols = min(5, len(cells)); rows = math.ceil(len(cells)/cols)
sheet = Image.new("RGB", (cols*w, rows*h), (0,0,0))
for k,(c,f,warm) in enumerate(cells):
    im = Image.fromarray(np.asarray(f).astype(np.uint8)); d = ImageDraw.Draw(im)
    d.rectangle([0,0,70,14], fill=(0,0,0))
    d.text((2,2), f"c{c} w{warm:.0f}"+(" SWAP" if c==swap_at else ""), fill=(255,210,74))
    sheet.paste(im, ((k%cols)*w, (k//cols)*h))
sheet.save(out); print("saved", out)
