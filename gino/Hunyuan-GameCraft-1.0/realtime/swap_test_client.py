"""
Headless driver for the live GameCraft demo: connect, capture the stream, fire a
mid-rollout prompt swap, keep capturing, then save a per-chunk contact sheet +
warmth(mean R-B) curve so we can see whether/how the scene transitions.

Usage:
  .venv/bin/python realtime/swap_test_client.py \
     --url http://localhost:8080 --swap-after 4 --stop-after 12 \
     --new-prompt "a ruined medieval village at night, snow falling, dark eerie moonlight" \
     --out experiments/22_gamecraft_swap/22b_live_swap
"""
import argparse, base64, io, os, time, threading
import numpy as np
from PIL import Image
import socketio

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8080")
ap.add_argument("--swap-after", type=int, default=4, help="emit swap after this many chunks")
ap.add_argument("--stop-after", type=int, default=12, help="stop after this many chunks")
ap.add_argument("--new-prompt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--hist", type=float, default=1.0, help="GC_HIST_SCALE applied at swap")
ap.add_argument("--gt", type=float, default=1.0, help="GC_GT_SCALE applied at swap")
ap.add_argument("--mask", type=float, default=1.0, help="GC_MASK_SCALE applied at swap")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

sio = socketio.Client()
frames = []            # (chunk_idx, np_frame)
chunk_meta = []        # chunk_done dicts
cur_chunk = [0]
swapped = {"done": False}
done = threading.Event()

@sio.on("frame")
def on_frame(d):
    img = np.array(Image.open(io.BytesIO(base64.b64decode(d["img"]))).convert("RGB"))
    frames.append((cur_chunk[0], img))

n_recv = [0]   # chunks received since connect (relative)
@sio.on("chunk_done")
def on_chunk(d):
    chunk_meta.append(d)
    c = d["chunk"]
    cur_chunk[0] = c + 1
    n_recv[0] += 1
    rel = n_recv[0]            # 1-based count since connect
    warm = None
    cf = [f for ci, f in frames if ci == c]
    if cf:
        f = cf[-1].astype(np.float32)
        warm = float(f[..., 0].mean() - f[..., 2].mean())
    print(f"recv {rel} (server chunk {c}): gen {d['gen_s']}s ({d['gen_fps']} FPS) "
          f"action={d['action']} warmth={warm if warm is None else round(warm,1)}")
    if rel == args.swap_after and not swapped["done"]:
        print(f">>> SWAP -> {args.new_prompt}  (levers hist={args.hist} gt={args.gt} mask={args.mask})")
        if (args.hist, args.gt, args.mask) != (1.0, 1.0, 1.0):
            sio.emit("levers", {"hist": args.hist, "gt": args.gt, "mask": args.mask})
        sio.emit("set_prompt", {"prompt": args.new_prompt})
        swapped["done"] = True
        swapped["at_chunk"] = c + 1     # server chunk where new prompt first applies
    if rel >= args.stop_after:
        done.set()

@sio.on("swapped")
def on_swapped(d):
    print(f"    (server confirmed swap: {d['prompt'][:50]})")

sio.connect(args.url)
print(f"connected to {args.url}; capturing {args.stop_after} chunks, swap after {args.swap_after}")
done.wait(timeout=400)
time.sleep(1)
sio.disconnect()

# ---- analysis ----
print(f"\ncaptured {len(frames)} frames across {len(chunk_meta)} chunks")
# warmth per chunk (mean over the chunk's frames)
import collections
bychunk = collections.defaultdict(list)
for ci, f in frames:
    bychunk[ci].append(f)
rows = []
for ci in sorted(bychunk):
    fs = np.stack(bychunk[ci]).astype(np.float32)
    warm = float(fs[..., 0].mean() - fs[..., 2].mean())
    bright = float(fs.mean())
    rows.append((ci, warm, bright, len(bychunk[ci])))
swap_at = swapped.get("at_chunk", args.swap_after)
print(f"\nchunk |  warmth | bright | nframes   (swap before chunk {swap_at})")
for ci, warm, bright, n in rows:
    mark = " <-- SWAP HERE" if ci == swap_at else ""
    print(f"  {ci:3d} | {warm:7.1f} | {bright:6.1f} | {n:4d}{mark}")

# contact sheet: last frame of each chunk
import math
cells = [(ci, bychunk[ci][-1]) for ci in sorted(bychunk)]
if cells:
    h, w, _ = cells[0][1].shape
    cols = min(6, len(cells)); rows_n = math.ceil(len(cells) / cols)
    sheet = Image.new("RGB", (cols * w, rows_n * h), (0, 0, 0))
    from PIL import ImageDraw
    for k, (ci, f) in enumerate(cells):
        im = Image.fromarray(f.astype(np.uint8))
        d = ImageDraw.Draw(im)
        tag = f"c{ci}" + (" SWAP" if ci == swap_at else "")
        d.rectangle([0, 0, 46, 14], fill=(0, 0, 0))
        d.text((2, 2), tag, fill=(255, 210, 74))
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
    sheet.save(os.path.join(args.out, "contact_sheet.png"))
    print(f"\nsaved {os.path.join(args.out, 'contact_sheet.png')}")

# also dump a short mp4 of the whole capture (best-effort)
try:
    import imageio
    allf = [f for _, f in frames]
    imageio.mimsave(os.path.join(args.out, "capture.mp4"), allf, fps=25)
    print(f"saved {os.path.join(args.out, 'capture.mp4')} ({len(allf)} frames)")
except Exception as e:
    print("mp4 save skipped:", e)
