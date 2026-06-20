# minWM — local run notes (this machine)

Set up for the **Wan Action2V (4-step DMD)** pipeline only. Hardware: 8× H100 80GB,
CUDA 12.9, Python 3.10.

## Environment
- venv: `./.venv` (created with `uv`, Python 3.10). `requirements.txt` installed
  (torch 2.9.1+cu128 — sees all 8 GPUs).
- Activate implicitly via the launcher, or: `source .venv/bin/activate`.
- The repo packages need `PYTHONPATH` (the upstream scripts assume it):
  `export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared:$PYTHONPATH"`

## Weights (public, no token needed) — fully self-contained, no symlinks
- Base (DiT + Wan VAE + umt5-xxl T5): a **real 17 GB copy** at
  `Wan21/wan_models/Wan2.1-T2V-1.3B/` (the path the code loads). Byte-identical to the HF
  release (verified by size + md5). No external dependency — safe even if other project
  folders change or are deleted. To re-fetch from scratch if ever needed:
  `hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir Wan21/wan_models/Wan2.1-T2V-1.3B`.
- DMD checkpoint: `./ckpts/Wan21/Action2V/dmd/model.pt` (4-step student, camera-conditioned
  Action2V). minWM-specific — NOT the same as Self-Forcing's `self_forcing_dmd.pt`
  (that one has no camera/action conditioning), so it can't be reused from other projects.

## Known pre-existing dangling symlinks (upstream, HY15 only — harmless here)
`HY15/trainer/models/wan/{causal_model,model,attention}.py` point at a `Causal-Forcing2/`
sibling that isn't present. They belong to the HunyuanVideo **training** path, are not
referenced by `Wan21/`, and do not affect the Wan demo. Left as-is so a future HY setup
(which would clone Causal-Forcing2) still works.

## Run
```bash
bash run_wan_demo.sh                       # all 30 demo prompts -> ./outputs/wan_action2v
# subset / custom:
DATA_PATH=/path/prompts.txt TRAJECTORY_PATH=/path/traj.txt \
  OUTPUT_FOLDER=./outputs/my_run bash run_wan_demo.sh
```
Output: one `.mp4` per prompt, 480×832 @ 16fps. Trajectories use `w/s/a/d`/`i/j/k/l`
keys with `*N` repeats, comma-separated (e.g. `a*4,w*8,s*7`), one line per prompt.

## Two environment-specific changes from upstream

1. **SDPA attention fallback** — `Wan21/wan/modules/attention.py`
   The prebuilt `flash-attn` PyPI wheel is ABI-incompatible with torch 2.9.1
   (`undefined symbol: ...SymInt...`), and a source build is a ~1100-file CUDA compile.
   `attention()` already had a PyTorch SDPA fallback, but `flash_attention()` (used by
   cross-attention) did not — it just `assert FLASH_ATTN_2_AVAILABLE`. Added an SDPA
   fallback there that honors the `k_lens` text-padding mask, GQA head expansion,
   `softmax_scale`/`q_scale` and `causal`. It is numerically equivalent for this
   inference path (exact softmax attention) and is **guarded** — if you later install a
   torch-2.9-compatible flash-attn, the original FA2/FA3 path is used automatically.

   To install real flash-attn instead (slow, optional): from the repo root with the
   venv active and `CUDA_HOME=/usr/local/cuda`, `TORCH_CUDA_ARCH_LIST=9.0`:
   `pip install flash-attn --no-build-isolation --no-binary :all: --no-deps`.

2. **Single-GPU = direct `python`, not `torchrun`**
   `torchrun` always sets `LOCAL_RANK`, so `wan_inference.py` initializes NCCL even for
   one GPU, and NCCL fails here ("Failed to initialize any NET plugin"). `run_wan_demo.sh`
   calls `python` directly → non-distributed branch → no NCCL. For multi-GPU (SP /
   sequence parallel) you'd use the upstream torchrun scripts and would first need to fix
   NCCL networking (e.g. `NCCL_SOCKET_IFNAME=<iface>`, `NCCL_P2P_LEVEL`, or
   `NCCL_NET=Socket`).

## HunyuanVideo-1.5 (HY) pipelines — installed & verified

Both **HY Action2V** (camera/ProPE control) and **HY TI2V** (text+image→video) run, 4-step
DMD, single-GPU. Verified: 480×832 @ 16fps, ~11–18 s/clip on one H100.

### Run
```bash
bash run_hy_action2v.sh     # camera control -> ./outputs/hy_action2v   (uses ckpts/HY15/Action2V/dmd)
bash run_hy_ti2v.sh         # text+image      -> ./outputs/hy_ti2v       (uses ckpts/HY15/TI2V/dmd)
# overridable: EXAMPLE_JSON=..., OUTPUT_DIR=..., TRANSFORMER_DIR=..., CUDA_VISIBLE_DEVICES=N
```
Inputs come from `assets/example.json` (18 entries: id, image, caption, trajectory).
`assets/example_smoke.json` is a 1-entry copy for quick tests.

### Weights (in ckpts/)
- HunyuanVideo-1.5 base: `transformer/480p_i2v` (downloaded; unused by ar_rollout), `vae/`,
  `scheduler/` (downloaded; scheduler built in-memory so unused).
- Text encoders: `text_encoder/{llm (Qwen2.5-VL-7B), byt5-small, Glyph-SDXL-v2}`.
- DMD transformers: `HY15/Action2V/dmd` (16 G), `HY15/TI2V/dmd` (32 G) — loaded via
  `--transformer_dir`.

### HY-specific notes (differ from Wan)
1. **No attention patch needed.** HY's ar_rollout path defaults to `torch_causal` (PyTorch
   SDPA); its flash-attn imports are lazy/guarded and never reached. (Only `--mode
   bidirectional` would hit flash/flex paths — out of scope, would need hardening.)
2. **No NCCL issue.** HY uses a **gloo** single-process group (`hy15_inference.py:164-166`),
   so plain `python` works; the launchers avoid torchrun anyway.
3. **TI2V transformer path:** upstream `run_infer_causal.sh` defaults to `TI2V/causal_cd`,
   but the released 4-step checkpoint is `TI2V/dmd` — `run_hy_ti2v.sh` uses `dmd`.
4. **SigLIP vision encoder = public substitute (no FLUX license).** Code loads
   `SiglipVisionModel.from_pretrained(siglip, subfolder="image_encoder")` +
   `SiglipImageProcessor.from_pretrained(siglip, subfolder="feature_extractor")`. The gated
   `FLUX.1-Redux-dev` 403'd, so `vision_encoder/siglip/image_encoder/` is filled with the
   public, byte-equivalent `google/siglip-so400m-patch14-384` (hidden 1152, 384px, patch14),
   and `feature_extractor/preprocessor_config.json` is a copy of the same preprocessor.
   If HY outputs ever look off, accepting the FLUX license and re-downloading is the
   exact-match fallback.
