"""
HY15 unified inference script.

Supports two inference modes:
  - bidirectional: full-sequence denoising with flash attention
  - ar_rollout: chunk-by-chunk autoregressive denoising with KV cache

When --trajectory is provided, uses ProPE (camera-conditioned) model and injects
viewmats/Ks into model calls. Without --trajectory, uses the standard action model.

Usage (standard):
    torchrun --nproc_per_node=1 HY15/hy15_inference.py \
        --mode bidirectional \
        --transformer_dir <ckpt_dir> \
        --example_json assets/example.json \
        --output_dir ./outputs/eval_bidir

Usage (camera/ProPE):
    torchrun --nproc_per_node=1 HY15/hy15_inference.py \
        --mode bidirectional \
        --transformer_dir <ckpt_dir> \
        --example_json assets/example.json \
        --output_dir ./outputs/eval_camera \
        --trajectory "w*19"
"""

import argparse
import json
import os
import re
import sys
import time
from types import SimpleNamespace

import imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler


# ---------------------------------------------------------------------------
# Camera trajectory (ProPE) utilities
# ---------------------------------------------------------------------------

_STEP = 0.08
_ROT_STEP = np.radians(3.0)

_MOTIONS = {
    "w":  {"forward":  _STEP},
    "s":  {"forward": -_STEP},
    "d":  {"right":    _STEP},
    "a":  {"right":   -_STEP},
    "u":  {"up":       _STEP},
    "dn": {"up":      -_STEP},
    "j":  {"yaw":     -_ROT_STEP},
    "l":  {"yaw":      _ROT_STEP},
    "i":  {"pitch":    _ROT_STEP},
    "k":  {"pitch":   -_ROT_STEP},
    # Aliases for verbose direction names
    "left":  {"yaw":     -_ROT_STEP},
    "right": {"yaw":      _ROT_STEP},
    "up":    {"pitch":    _ROT_STEP},
    "down":  {"pitch":   -_ROT_STEP},
}


def parse_trajectory(traj_str):
    segments = traj_str.strip().split(",")
    motions = []
    for seg in segments:
        seg = seg.strip()
        m = re.fullmatch(r"([a-z]+)\*(\d+)", seg)
        if m is None:
            raise ValueError(f"Cannot parse trajectory segment: '{seg}'.'.")
        key, n = m.group(1), int(m.group(2))
        if key not in _MOTIONS:
            raise ValueError(f"Unknown direction '{key}'. Valid: {list(_MOTIONS.keys())}")
        motions.extend([_MOTIONS[key]] * n)
    return motions


def make_camera_tensors(traj_str, fx=0.5050505, fy=0.89786756, cx=0.5, cy=0.5):
    from hyvideo.generate_custom_trajectory import generate_camera_trajectory_local
    motions = parse_trajectory(traj_str)
    c2w_list = generate_camera_trajectory_local(motions)
    T = len(c2w_list)
    viewmats = np.zeros((T, 4, 4), dtype=np.float32)
    for i, c2w in enumerate(c2w_list):
        viewmats[i] = np.linalg.inv(c2w)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    Ks = np.tile(K, (T, 1, 1))
    return torch.from_numpy(viewmats).unsqueeze(0), torch.from_numpy(Ks).unsqueeze(0)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["bidirectional", "ar_rollout"])

    # Model paths
    parser.add_argument("--transformer_dir", required=True,
                        help="Transformer checkpoint dir (contains config.json + diffusion_pytorch_model.safetensors)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="HunyuanVideo-1.5 base model dir (contains vae/, text_encoder/, vision_encoder/). "
                             "Auto-detected from HF cache if not specified.")
    parser.add_argument("--action_ckpt", type=str, default=None,
                        help="Path to action model safetensors (overrides transformer_dir weights)")

    # Data
    parser.add_argument("--example_json", default=None,
                        help="JSON file with list of {image, caption} entries (relative paths resolved from JSON dir)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate outputs even when the target mp4 already exists.")
    parser.add_argument("--profile_timing", action="store_true",
                        help="Print CUDA-synchronized per-phase timing and FPS metrics.")
    parser.add_argument("--serve_stdin", action="store_true",
                        help="Keep the model resident and read one JSON request per line from stdin on rank 0.")
    parser.add_argument("--max_requests", type=int, default=0,
                        help="Maximum stdin requests to serve. 0 means run until EOF or {\"stop\": true}.")
    parser.add_argument("--conditioning_cache_size", type=int, default=8,
                        help="Number of encoded image/text conditionings to keep resident on CPU. 0 disables.")
    parser.add_argument("--vae_decode_mode", choices=["leader", "tile_parallel", "none"], default="tile_parallel",
                        help="tile_parallel = all SP ranks decode VAE tiles in parallel (default, ~Nx faster decode; "
                             "falls back to single-rank decode when sp_size==1); leader = rank 0 decodes/saves; "
                             "none = skip decode/write for hot inference benchmarks.")

    # Camera trajectory (ProPE) - optional
    parser.add_argument("--trajectory", type=str, default=None,
                        help="Camera trajectory string, e.g. 'w*19', 'd*10,w*9'. "
                             "Overrides per-sample trajectory from JSON.")
    parser.add_argument("--use_camera", action="store_true",
                        help="Enable camera mode: read trajectory from JSON per sample. "
                             "Samples without trajectory field are skipped.")

    # Inference hyperparameters
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                        help="CFG guidance scale. 1.0 disables CFG.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--video_length", type=int, default=77,
                        help="Number of output video frames (must satisfy (n-1)//4+1 divisible by 4)")

    # Discrete action conditioning
    parser.add_argument("--use_discrete_action", action="store_true",
                        help="Pass discrete action labels to model (requires action_in module in ckpt)")

    # AR-specific
    parser.add_argument("--stabilization_level", type=int, default=1,
                        help="Timestep for clean context frame modulation (ar_rollout only)")
    parser.add_argument("--chunk_latent_frames", type=int, default=4,
                        help="Frames per chunk for ar_rollout")

    # Multi-GPU execution
    parser.add_argument("--parallel_mode", choices=["data", "sp", "xdit_usp"], default="data",
                        help="data = one full model per rank over different samples; "
                             "sp = repo sequence-parallel token sharding within rank groups; "
                             "xdit_usp = xDiT hybrid Ulysses/Ring sequence parallel.")
    parser.add_argument("--sp_size", type=int, default=0,
                        help="Ranks per sequence-parallel group. 0 means all ranks when "
                             "--parallel_mode sp is used.")
    parser.add_argument("--ulysses_degree", type=int, default=0,
                        help="xDiT USP Ulysses degree. 0 means --sp_size for pure Ulysses.")
    parser.add_argument("--ring_degree", type=int, default=1,
                        help="xDiT USP Ring degree.")
    parser.add_argument("--xdit_attention_backend", type=str, default="auto",
                        help="xDiT attention backend name, e.g. auto, sdpa, sdpa_flash, flash.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed. SP requires a non-negative seed so every rank "
                             "starts from identical conditioning/noise before token sharding.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def _ensure_single_process_env():
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"


def setup_dist(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)

    sp_size = 1
    uses_sequence_parallel = False
    uses_model_parallel = False
    uses_xdit_usp = False
    ulysses_degree = 1
    ring_degree = 1

    if args.parallel_mode in {"sp", "xdit_usp"}:
        sp_size = args.sp_size if args.sp_size > 0 else world_size
        if sp_size < 1:
            raise ValueError(f"Invalid --sp_size {sp_size}")
        if world_size % sp_size != 0:
            raise ValueError(
                f"WORLD_SIZE={world_size} must be divisible by --sp_size={sp_size}")
        uses_sequence_parallel = sp_size > 1
        uses_xdit_usp = args.parallel_mode == "xdit_usp"

    if uses_xdit_usp:
        if world_size != sp_size:
            raise ValueError(
                "--parallel_mode xdit_usp currently supports one SP worker group, "
                f"but WORLD_SIZE={world_size} and --sp_size={sp_size}.")
        ring_degree = args.ring_degree
        ulysses_degree = args.ulysses_degree if args.ulysses_degree > 0 else sp_size
        if ring_degree < 1 or ulysses_degree < 1:
            raise ValueError("--ulysses_degree and --ring_degree must be >= 1")
        if ulysses_degree * ring_degree != sp_size:
            raise ValueError(
                f"xDiT USP requires ulysses_degree * ring_degree == sp_size, got "
                f"{ulysses_degree} * {ring_degree} != {sp_size}")

    if uses_sequence_parallel:
        if args.seed < 0:
            raise ValueError("--parallel_mode sp/xdit_usp requires --seed >= 0")
        from sp.parallel_state import maybe_init_distributed_environment_and_model_parallel

        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=sp_size)
        uses_model_parallel = True
        if uses_xdit_usp:
            from xfuser.core.distributed import (
                init_distributed_environment as xdit_init_distributed_environment,
                initialize_model_parallel as xdit_initialize_model_parallel,
            )

            xdit_init_distributed_environment(
                world_size=world_size,
                rank=rank,
                local_rank=local_rank,
                backend="nccl",
            )
            xdit_initialize_model_parallel(
                data_parallel_degree=1,
                classifier_free_guidance_degree=1,
                sequence_parallel_degree=sp_size,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                tensor_parallel_degree=1,
                pipeline_parallel_degree=1,
                fully_shard_degree=1,
                vae_parallel_size=0,
                use_parallel_vae=False,
                backend="nccl",
            )
    else:
        if world_size == 1:
            _ensure_single_process_env()
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", init_method="env://",
                                    world_size=world_size, rank=rank)

    if args.mode == "bidirectional" and not uses_sequence_parallel:
        import sp.parallel_state as ps

        _orig_init_mp = ps.init_model_parallel_group
        def _patched_init_mp(group_ranks, local_rank, backend, **kwargs):
            return ps.GroupCoordinator(
                group_ranks=group_ranks,
                local_rank=local_rank,
                torch_distributed_backend=backend,
                use_device_communicator=False,
                group_name=kwargs.get("group_name"),
            )
        ps.init_model_parallel_group = _patched_init_mp

        if ps._WORLD is None:
            ps._WORLD = ps.GroupCoordinator(
                group_ranks=[list(range(world_size))],
                local_rank=local_rank,
                torch_distributed_backend="gloo",
                use_device_communicator=False,
                group_name="world",
            )
        ps.initialize_model_parallel(tensor_model_parallel_size=1,
                                     sequence_model_parallel_size=1)
        ps.init_model_parallel_group = _orig_init_mp
        uses_model_parallel = True

    if args.mode == "ar_rollout":
        from hyvideo.commons.infer_state import initialize_infer_state
        initialize_infer_state(SimpleNamespace(
            sage_blocks_range="0-0",
            use_sageattn=False,
            enable_torch_compile=False,
            use_xdit_usp=uses_xdit_usp,
            xdit_attention_backend=args.xdit_attention_backend,
            use_fp8_gemm=False,
            quant_type="fp8-per-block",
            include_patterns="double_blocks",
            use_vae_parallel=False,
        ))

    if uses_sequence_parallel:
        worker_group_id = rank // sp_size
        worker_group_rank = rank % sp_size
        num_worker_groups = world_size // sp_size
    else:
        worker_group_id = rank
        worker_group_rank = 0
        num_worker_groups = world_size

    return SimpleNamespace(
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        sp_size=sp_size,
        uses_sequence_parallel=uses_sequence_parallel,
        uses_model_parallel=uses_model_parallel,
        uses_xdit_usp=uses_xdit_usp,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        worker_group_id=worker_group_id,
        worker_group_rank=worker_group_rank,
        num_worker_groups=num_worker_groups,
        is_worker_leader=worker_group_rank == 0,
        worker_leader_rank=worker_group_id * sp_size,
    )


def cleanup_dist(dist_info):
    if getattr(dist_info, "uses_xdit_usp", False):
        from xfuser.core.distributed.parallel_state import (
            destroy_model_parallel as xdit_destroy_model_parallel,
        )
        xdit_destroy_model_parallel()
    if dist_info.uses_model_parallel:
        from sp.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory()
    elif dist.is_initialized():
        dist.destroy_process_group()


def set_sample_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_cuda_generator(device, seed):
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def sync_cuda(device):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def elapsed_since(t0, device):
    sync_cuda(device)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Shared utilities: on-the-fly encoding
# ---------------------------------------------------------------------------

def load_examples(example_json):
    """Load example list from JSON file. Returns list of {image, caption} dicts with absolute paths."""
    base_dir = os.path.dirname(os.path.abspath(example_json))
    with open(example_json) as f:
        examples = json.load(f)
    for ex in examples:
        img_path = ex["image"]
        if not os.path.isabs(img_path):
            ex["image"] = os.path.join(base_dir, img_path)
    return examples


def find_hunyuanvideo_model_path():
    """Auto-detect HunyuanVideo-1.5 model path under ./ckpts/."""
    c = "./ckpts/HunyuanVideo-1.5"
    if os.path.isdir(c):
        return c
    return None


# Keep the VAE / text / vision encoders resident on the GPU instead of shuffling
# them back to CPU after every encode. The shuffle is a low-VRAM strategy; on an
# 80GB H100 (DiT ~17GB + llava ~16GB + VAE ~3GB + SigLIP/byt5 ~2GB ~= 38GB) there
# is ample room, and the repeated ~20GB host<->device copies dominated the "encode"
# phase. Set HY_RESIDENT=0 to restore offloading on memory-constrained GPUs.
_KEEP_RESIDENT = os.getenv("HY_RESIDENT", "1") == "1"


def _maybe_offload(module):
    """Move a sub-model back to CPU only when resident mode is disabled."""
    if not _KEEP_RESIDENT:
        module.cpu()


def encode_image_to_cond_latent(vae, image_path, height, width, device):
    """Load image, resize, encode through VAE to get conditional latent."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = transform(image).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
    img_tensor = img_tensor.to(device, dtype=torch.float16)
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()
    scaling_factor = vae.config.scaling_factor
    shift_factor = getattr(vae.config, "shift_factor", None)
    if shift_factor:
        latent = (latent - shift_factor) * scaling_factor
    else:
        latent = latent * scaling_factor
    return latent.to(dtype=torch.bfloat16)


def encode_text(text_encoder, caption, device):
    """Encode text using the LLM text encoder."""
    text_encoder = text_encoder.to(device)
    with torch.no_grad():
        outputs = text_encoder([caption])
    _maybe_offload(text_encoder)
    prompt_embeds = outputs.hidden_state.to(device)
    prompt_mask = outputs.attention_mask.to(device) if outputs.attention_mask is not None else None
    return prompt_embeds, prompt_mask


def encode_vision(vision_encoder, image_path, height, width, device):
    """Encode image using SigLIP vision encoder."""
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image.resize((width, height)))  # [H, W, 3] uint8
    image_np = image_np[np.newaxis, ...]  # [1, H, W, 3]
    vision_encoder = vision_encoder.to(device)
    with torch.no_grad():
        outputs = vision_encoder.encode_images(image_np)
    _maybe_offload(vision_encoder)
    return outputs.last_hidden_state


def encode_byt5(byt5_model, byt5_tokenizer, caption, byt5_max_length, device):
    """Encode text using byT5 model."""
    if byt5_model is None:
        return torch.zeros((1, byt5_max_length, 1472), device=device), \
               torch.zeros((1, byt5_max_length), device=device, dtype=torch.int64)

    with torch.no_grad():
        byt5_text_inputs = byt5_tokenizer(
            caption, return_tensors="pt", padding="max_length",
            max_length=byt5_max_length, truncation=True,
        )
        text_ids = byt5_text_inputs.input_ids.to(device)
        text_mask = byt5_text_inputs.attention_mask.to(device)
        byt5_outputs = byt5_model(text_ids, attention_mask=text_mask.float())
        byt5_embeddings = byt5_outputs[0]
        byt5_mask = text_mask

    return byt5_embeddings, byt5_mask


def encode_negative_prompt(text_encoder, byt5_model, byt5_tokenizer, byt5_max_length, device):
    """Encode empty/negative prompt for CFG."""
    neg_prompt = ""
    neg_embeds, neg_mask = encode_text(text_encoder, neg_prompt, device)
    neg_byt5_states = torch.zeros((1, byt5_max_length, 1472), device=device)
    neg_byt5_mask = torch.zeros((1, byt5_max_length), device=device, dtype=torch.int64)
    return {
        "prompt_embeds": neg_embeds,
        "prompt_mask": neg_mask,
        "byt5_text_states": neg_byt5_states,
        "byt5_text_mask": neg_byt5_mask,
    }


def prepare_sample_data(vae, text_encoder, vision_encoder, byt5_model, byt5_tokenizer,
                        example, height, width, video_length, device):
    """
    Encode a single example (image + caption) into the data dict expected by inference functions.
    Returns dict with keys: image_cond, prompt_embeds, prompt_mask, vision_states,
                            byt5_text_states, byt5_text_mask, latent_shape
    """
    image_path = example["image"]
    caption = example["caption"]

    # Encode image -> VAE conditional latent
    vae = vae.to(device)
    image_cond = encode_image_to_cond_latent(vae, image_path, height, width, device)
    _maybe_offload(vae)
    # image_cond: [1, C, 1, h, w]

    # Compute latent spatial dims
    C = image_cond.shape[1]
    h = image_cond.shape[3]
    w = image_cond.shape[4]
    T = (video_length - 1) // 4 + 1  # temporal compression factor = 4

    # Encode text
    prompt_embeds, prompt_mask = encode_text(text_encoder, caption, device)

    # Encode vision (SigLIP)
    vision_states = encode_vision(vision_encoder, image_path, height, width, device)

    # Encode byT5
    byt5_max_length = 256
    byt5_states, byt5_mask = encode_byt5(byt5_model, byt5_tokenizer, caption, byt5_max_length, device)

    return {
        "image_cond": image_cond.cpu(),  # [1, C, 1, h, w]
        "prompt_embeds": prompt_embeds.cpu(),
        "prompt_mask": prompt_mask.cpu(),
        "vision_states": vision_states.cpu(),
        "byt5_text_states": byt5_states.cpu(),
        "byt5_text_mask": byt5_mask.cpu(),
        "latent_shape": (1, C, T, h, w),
    }


def decode_and_save(x, vae, device, output_path, fps, decode_mode="leader", dist_info=None):
    if decode_mode == "none":
        return False
    if decode_mode == "leader" and dist_info is not None and not dist_info.is_worker_leader:
        return False

    use_tile_parallel = (
        decode_mode == "tile_parallel"
        and dist_info is not None
        and dist_info.uses_sequence_parallel
        and dist_info.sp_size > 1
    )

    vae = vae.to(device)
    if use_tile_parallel:
        vae.enable_tiling(True)
        vae.enable_tile_parallelism()
    else:
        vae.disable_tile_parallelism()

    scaling_factor = vae.config.scaling_factor
    shift_factor = getattr(vae.config, "shift_factor", None)
    x_decoded = x / scaling_factor + shift_factor if shift_factor else x / scaling_factor
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        video = vae.decode(x_decoded).sample

    saved = False
    should_save = dist_info is None or dist_info.is_worker_leader
    if should_save and video.numel() > 0:
        video = (video.float().clamp(-1, 1) + 1) / 2
        frames = rearrange(video[0], "c t h w -> t h w c")
        frames = (frames.cpu().numpy() * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, frames, fps=fps)
        saved = True

    if use_tile_parallel:
        vae.disable_tile_parallelism()
    _maybe_offload(vae)
    return saved


def conditioning_cache_key(example, height, width, video_length):
    return (
        os.path.abspath(example["image"]),
        example["caption"],
        int(height),
        int(width),
        int(video_length),
    )


def iter_stdin_requests(args, dist_info):
    if dist_info.num_worker_groups != 1:
        raise ValueError("--serve_stdin expects one worker group; use SP/xDiT with all ranks for one prompt.")

    if dist_info.rank == 0:
        print(
            "[serve] Ready for JSONL requests on stdin. "
            "Each line needs image, caption, and optional trajectory/output_path/seed. "
            "Send {\"stop\": true} or EOF to exit.",
            flush=True,
        )

    request_idx = 0
    while args.max_requests <= 0 or request_idx < args.max_requests:
        payload = [None]
        if dist_info.rank == 0:
            line = sys.stdin.readline()
            if line == "":
                payload[0] = {"stop": True}
            else:
                try:
                    payload[0] = json.loads(line)
                except json.JSONDecodeError as exc:
                    payload[0] = {"error": str(exc)}

        dist.broadcast_object_list(payload, src=0)
        request = payload[0]

        if request.get("stop"):
            break
        if request.get("error"):
            if dist_info.rank == 0:
                print(f"[serve] Invalid JSON request: {request['error']}", flush=True)
            continue
        if "image" not in request or "caption" not in request:
            raise ValueError("stdin requests must include at least 'image' and 'caption'.")

        image_path = request["image"]
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)

        example = {
            "image": image_path,
            "caption": request["caption"],
        }
        if request.get("trajectory"):
            example["trajectory"] = request["trajectory"]
        if request.get("output_path"):
            output_path = request["output_path"]
            if not os.path.isabs(output_path):
                output_path = os.path.join(args.output_dir, output_path)
            example["_output_path"] = output_path
        if "seed" in request:
            example["_seed"] = int(request["seed"])

        sample_idx = int(request.get("sample_idx", request_idx))
        yield request_idx, sample_idx, example
        request_idx += 1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_bidirectional(transformer_dir):
    from trainer.models.hyvideo.transformer.ar_action_hunyuanvideo_1_5_transformer import (
        ARHunyuanVideo_1_5_DiffusionTransformer,
    )
    return ARHunyuanVideo_1_5_DiffusionTransformer.from_pretrained(transformer_dir)


def load_model_ar_rollout(transformer_dir):
    from hyvideo.models.transformers.worldplay_1_5_transformer import (
        HunyuanVideo_1_5_DiffusionTransformer,
    )
    return HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
        transformer_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )


def load_model_prope(transformer_dir, use_discrete_action=False):
    """Load ProPE camera-conditioned transformer (used for both modes when trajectory is provided)."""
    from trainer.models.hyvideo.transformer.ar_action_hunyuanvideo_1_5_prope_transformer import \
        ARHunyuanVideo_1_5_DiffusionTransformer
    config = ARHunyuanVideo_1_5_DiffusionTransformer.load_config(transformer_dir)
    model = ARHunyuanVideo_1_5_DiffusionTransformer.from_config(config)
    model.add_prope_parameters()
    if use_discrete_action:
        model.add_discrete_action_parameters()
    ckpt_path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
    state_dict = load_file(ckpt_path, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_model_prope] Missing keys: {missing}")
    if unexpected:
        print(f"[load_model_prope] Unexpected keys: {unexpected}")
    return model


# ---------------------------------------------------------------------------
# Bidirectional inference
# ---------------------------------------------------------------------------

def run_inference_bidirectional(model, data, neg_prompts, device, num_steps, shift,
                                guidance_scale, viewmats=None, Ks=None, action=None,
                                seed=None):
    """
    Bidirectional full-sequence denoising.
    When viewmats/Ks are provided, passes them to model (ProPE mode).
    """
    image_cond = data["image_cond"].to(device, dtype=torch.bfloat16)
    prompt_embed = data["prompt_embeds"].to(device, dtype=torch.bfloat16)
    prompt_mask = data["prompt_mask"].to(device, dtype=torch.bfloat16)
    vision_states = data["vision_states"].to(device, dtype=torch.bfloat16)
    byt5_text_states = data["byt5_text_states"].to(device, dtype=torch.bfloat16)
    byt5_text_mask = data["byt5_text_mask"].to(device, dtype=torch.bfloat16)

    B, C, T, H, W = data["latent_shape"]
    generator = make_cuda_generator(device, seed)
    randn_kwargs = {"generator": generator} if generator is not None else {}
    x = torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16,
                    **randn_kwargs)

    # i2v conditioning: [B, C+1, T, H, W]
    cond_latents = image_cond.repeat(1, 1, T, 1, 1)
    cond_latents[:, :, 1:, :, :] = 0.0
    mask = torch.zeros(B, 1, T, H, W, device=device)
    mask[:, :, 0, :, :] = 1.0
    cond_input = torch.cat([cond_latents, mask], dim=1)

    # ProPE camera kwargs (empty dict if no trajectory)
    prope_kwargs = {}
    if viewmats is not None:
        prope_kwargs = {"viewmats": viewmats.to(device, dtype=torch.bfloat16),
                        "Ks": Ks.to(device, dtype=torch.bfloat16)}
    if action is not None:
        prope_kwargs["action"] = action.to(device, dtype=torch.int64)

    use_cfg = guidance_scale > 1.0
    if use_cfg:
        neg_embed = neg_prompts["prompt_embeds"].to(device, dtype=torch.bfloat16)
        neg_mask = neg_prompts["prompt_mask"].to(device, dtype=torch.bfloat16)
        neg_byt5_states = neg_prompts["byt5_text_states"].to(device, dtype=torch.bfloat16)
        neg_byt5_mask = neg_prompts["byt5_text_mask"].to(device, dtype=torch.bfloat16)

    scheduler = FlowMatchDiscreteScheduler(shift=shift)
    scheduler.set_timesteps(num_steps, device=device)

    extra_kwargs = {"byt5_text_states": byt5_text_states, "byt5_text_mask": byt5_text_mask}
    timestep_txt = torch.tensor(0).unsqueeze(0).to(device, dtype=torch.bfloat16)

    total_steps = len(scheduler.timesteps)
    for i, t in enumerate(scheduler.timesteps):
        if dist.get_rank() == 0:
            print(f"  denoising {i+1}/{total_steps}, t={t.item():.1f}", flush=True)
        timesteps_in = t.unsqueeze(0).expand(B * T).to(device, dtype=torch.bfloat16)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cond_pred = model(
                hidden_states=torch.cat([x, cond_input], dim=1),
                timestep=timesteps_in,
                timestep_txt=timestep_txt,
                text_states=prompt_embed,
                text_states_2=None,
                encoder_attention_mask=prompt_mask,
                timestep_r=None,
                vision_states=vision_states,
                mask_type="i2v",
                guidance=None,
                extra_kwargs=extra_kwargs,
                return_dict=False,
                **prope_kwargs,
            )[0]

            if use_cfg:
                uncond_pred = model(
                    hidden_states=torch.cat([x, cond_input], dim=1),
                    timestep=timesteps_in,
                    timestep_txt=timestep_txt,
                    text_states=neg_embed,
                    text_states_2=None,
                    encoder_attention_mask=neg_mask,
                    timestep_r=None,
                    vision_states=vision_states,
                    mask_type="i2v",
                    guidance=None,
                    extra_kwargs={"byt5_text_states": neg_byt5_states, "byt5_text_mask": neg_byt5_mask},
                    return_dict=False,
                    **prope_kwargs,
                )[0]
                pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                pred = cond_pred

        x = scheduler.step(pred, t, x).prev_sample

    return x


# ---------------------------------------------------------------------------
# AR rollout inference (chunk-by-chunk with KV cache)
# ---------------------------------------------------------------------------

def _init_kv_cache(num_layers):
    return [{"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None}
            for _ in range(num_layers)]


def run_inference_rollout(model, data, neg_prompts, device, num_steps, shift,
                          guidance_scale, stabilization_level, chunk_latent_frames=4,
                          viewmats=None, Ks=None, action=None, seed=None,
                          profile_timing=False):
    """
    AR rollout chunk-by-chunk denoising with KV cache.
    When viewmats/Ks are provided, passes per-chunk camera tensors to model (ProPE mode).
    """
    sync_cuda(device)
    total_t0 = time.perf_counter()
    chunk0_t0 = total_t0
    chunk0_latency = None
    timings = {
        "input_to_gpu": 0.0,
        "latent_init": 0.0,
        "text_kv": 0.0,
        "denoise": 0.0,
        "vision_cache": 0.0,
        "total": 0.0,
    }

    t0 = time.perf_counter()
    image_cond = data["image_cond"].to(device, dtype=torch.bfloat16)
    prompt_embed = data["prompt_embeds"].to(device, dtype=torch.bfloat16)
    prompt_mask = data["prompt_mask"].to(device, dtype=torch.bfloat16)
    vision_states = data["vision_states"].to(device, dtype=torch.bfloat16)
    byt5_text_states = data["byt5_text_states"].to(device, dtype=torch.bfloat16)
    byt5_text_mask = data["byt5_text_mask"].to(device, dtype=torch.bfloat16)
    timings["input_to_gpu"] = elapsed_since(t0, device)

    B, C, T, H, W = data["latent_shape"]
    use_cfg = guidance_scale > 1.0
    chunk_num = T // chunk_latent_frames

    t0 = time.perf_counter()
    # i2v conditioning
    cond_latents = image_cond.repeat(1, 1, T, 1, 1)
    cond_latents[:, :, 1:, :, :] = 0.0
    mask = torch.zeros(B, 1, T, H, W, device=device)
    mask[:, :, 0, :, :] = 1.0
    cond_input = torch.cat([cond_latents, mask], dim=1)

    generator = make_cuda_generator(device, seed)
    randn_kwargs = {"generator": generator} if generator is not None else {}
    latents = torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16,
                          **randn_kwargs)
    timings["latent_init"] = elapsed_since(t0, device)

    num_layers = len(model.double_blocks)
    kv_cache = _init_kv_cache(num_layers)
    kv_cache_neg = _init_kv_cache(num_layers) if use_cfg else None

    extra_kwargs = {"byt5_text_states": byt5_text_states, "byt5_text_mask": byt5_text_mask}
    t_txt = torch.tensor([0]).to(device, dtype=torch.bfloat16)

    # Phase 1: cache text KV
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        kv_cache = model(
            bi_inference=False, ar_txt_inference=True, ar_vision_inference=False,
            timestep_txt=t_txt, text_states=prompt_embed,
            encoder_attention_mask=prompt_mask, vision_states=vision_states,
            mask_type="i2v", extra_kwargs=extra_kwargs,
            kv_cache=kv_cache, cache_txt=True,
        )
        if use_cfg:
            neg_embed = neg_prompts["prompt_embeds"].to(device, dtype=torch.bfloat16)
            neg_mask = neg_prompts["prompt_mask"].to(device, dtype=torch.bfloat16)
            neg_byt5 = neg_prompts["byt5_text_states"].to(device, dtype=torch.bfloat16)
            neg_byt5_mask = neg_prompts["byt5_text_mask"].to(device, dtype=torch.bfloat16)
            neg_extra = {"byt5_text_states": neg_byt5, "byt5_text_mask": neg_byt5_mask}
            kv_cache_neg = model(
                bi_inference=False, ar_txt_inference=True, ar_vision_inference=False,
                timestep_txt=t_txt, text_states=neg_embed,
                encoder_attention_mask=neg_mask, vision_states=vision_states,
                mask_type="i2v", extra_kwargs=neg_extra,
                kv_cache=kv_cache_neg, cache_txt=True,
            )
    timings["text_kv"] = elapsed_since(t0, device)

    # Phase 2: chunk-by-chunk denoising
    scheduler = FlowMatchDiscreteScheduler(shift=shift, reverse=True, solver="euler")

    for chunk_i in range(chunk_num):
        start_idx = chunk_i * chunk_latent_frames
        end_idx = start_idx + chunk_latent_frames
        rope_total = end_idx

        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps

        if dist.get_rank() == 0:
            print(f"  Chunk {chunk_i+1}/{chunk_num} frames[{start_idx}:{end_idx})", flush=True)

        # ProPE: per-chunk camera tensors
        prope_kwargs = {}
        if viewmats is not None:
            vm_chunk = viewmats[:, start_idx:end_idx].to(device, dtype=torch.bfloat16)
            Ks_chunk = Ks[:, start_idx:end_idx].to(device, dtype=torch.bfloat16)
            prope_kwargs = {"viewmats": vm_chunk, "Ks": Ks_chunk}
        if action is not None:
            prope_kwargs["action"] = action[start_idx:end_idx].to(device, dtype=torch.int64)

        denoise_t0 = time.perf_counter()
        for i, t in enumerate(timesteps):
            if dist.get_rank() == 0:
                print("timesteps", t.item(), flush=True)
            ts_in = torch.full((chunk_latent_frames,), t, device=device, dtype=timesteps.dtype)
            latent_chunk = latents[:, :, start_idx:end_idx]
            cond_chunk = cond_input[:, :, start_idx:end_idx]
            hidden = torch.cat([latent_chunk, cond_chunk], dim=1)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cond_pred = model(
                    bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                    hidden_states=hidden, timestep=ts_in, timestep_r=None,
                    mask_type="i2v", return_dict=False,
                    kv_cache=kv_cache, cache_vision=False,
                    rope_temporal_size=rope_total, start_rope_start_idx=start_idx,
                    **prope_kwargs,
                )[0]
                if use_cfg:
                    uncond_pred = model(
                        bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                        hidden_states=hidden, timestep=ts_in, timestep_r=None,
                        mask_type="i2v", return_dict=False,
                        kv_cache=kv_cache_neg, cache_vision=False,
                        rope_temporal_size=rope_total, start_rope_start_idx=start_idx,
                        **prope_kwargs,
                    )[0]

            if use_cfg:
                pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                pred = cond_pred

            latent_chunk = scheduler.step(pred, t, latent_chunk, return_dict=False)[0]
            latents[:, :, start_idx:end_idx] = latent_chunk
        timings["denoise"] += elapsed_since(denoise_t0, device)

        if chunk_i == 0:
            sync_cuda(device)
            chunk0_latency = time.perf_counter() - chunk0_t0

        # Phase 3: cache denoised chunk vision KV
        cache_t0 = time.perf_counter()
        denoised_chunk = latents[:, :, start_idx:end_idx]
        denoised_cond = cond_input[:, :, start_idx:end_idx]
        denoised_input = torch.cat([denoised_chunk, denoised_cond], dim=1)
        ctx_ts = torch.full((chunk_latent_frames,), stabilization_level - 1,
                            device=device, dtype=torch.bfloat16)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            new_kv = model(
                bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                hidden_states=denoised_input, timestep=ctx_ts, timestep_r=None,
                mask_type="i2v", return_dict=False,
                kv_cache=kv_cache, cache_vision=True,
                rope_temporal_size=rope_total, start_rope_start_idx=start_idx,
                **prope_kwargs,
            )
            for j in range(num_layers):
                if kv_cache[j]["k_vision"] is None:
                    kv_cache[j]["k_vision"] = new_kv[j]["k_vision"]
                    kv_cache[j]["v_vision"] = new_kv[j]["v_vision"]
                else:
                    kv_cache[j]["k_vision"] = torch.cat(
                        [kv_cache[j]["k_vision"], new_kv[j]["k_vision"]], dim=2)
                    kv_cache[j]["v_vision"] = torch.cat(
                        [kv_cache[j]["v_vision"], new_kv[j]["v_vision"]], dim=2)

            if use_cfg:
                new_kv_neg = model(
                    bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                    hidden_states=denoised_input, timestep=ctx_ts, timestep_r=None,
                    mask_type="i2v", return_dict=False,
                    kv_cache=kv_cache_neg, cache_vision=True,
                    rope_temporal_size=rope_total, start_rope_start_idx=start_idx,
                    **prope_kwargs,
                )
                for j in range(num_layers):
                    if kv_cache_neg[j]["k_vision"] is None:
                        kv_cache_neg[j]["k_vision"] = new_kv_neg[j]["k_vision"]
                        kv_cache_neg[j]["v_vision"] = new_kv_neg[j]["v_vision"]
                    else:
                        kv_cache_neg[j]["k_vision"] = torch.cat(
                            [kv_cache_neg[j]["k_vision"], new_kv_neg[j]["k_vision"]], dim=2)
                        kv_cache_neg[j]["v_vision"] = torch.cat(
                            [kv_cache_neg[j]["v_vision"], new_kv_neg[j]["v_vision"]], dim=2)
        timings["vision_cache"] += elapsed_since(cache_t0, device)

    timings["chunk0_latency"] = chunk0_latency
    timings["total"] = elapsed_since(total_t0, device)
    return latents, timings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dist_info = setup_dist(args)
    rank = dist_info.rank
    world_size = dist_info.world_size
    device = torch.device(f"cuda:{dist_info.local_rank}")

    if rank == 0:
        print(f"Mode: {args.mode}")
        print(f"Transformer: {args.transformer_dir}")
        print(f"CFG guidance_scale={args.guidance_scale}, shift={args.shift}")
        print(f"World size: {world_size}")
        print(f"Parallel mode: {args.parallel_mode}")
        if dist_info.uses_sequence_parallel:
            print(f"Sequence parallel size: {dist_info.sp_size} "
                  f"({dist_info.num_worker_groups} worker group(s))")
        if dist_info.uses_xdit_usp:
            print(f"xDiT USP: ulysses_degree={dist_info.ulysses_degree}, "
                  f"ring_degree={dist_info.ring_degree}, "
                  f"backend={args.xdit_attention_backend}")

    # Auto-detect model_path if not provided
    model_path = args.model_path
    if model_path is None:
        model_path = find_hunyuanvideo_model_path()
        if model_path is None:
            raise RuntimeError(
                "Cannot auto-detect HunyuanVideo-1.5 model path. "
                "Please specify --model_path explicitly.")
    if rank == 0:
        print(f"Model path: {model_path}")

    if not args.serve_stdin and args.example_json is None:
        raise ValueError("--example_json is required unless --serve_stdin is set.")

    # Load all examples and assign to this rank or SP worker group.
    all_examples = load_examples(args.example_json) if args.example_json else []

    # Camera mode: --use_camera or --trajectory CLI arg
    camera_mode = args.use_camera or args.trajectory is not None

    if args.serve_stdin:
        target_examples = []
        my_examples = []
        camera_mode = True
    else:
        if camera_mode and args.trajectory is None:
            # Camera mode reading per-sample trajectory: skip samples without trajectory
            target_examples = [(i, ex) for i, ex in enumerate(all_examples)
                               if ex.get("trajectory")]
        elif not camera_mode:
            # Ti2v mode: skip samples that have trajectory (they're camera-only)
            target_examples = [(i, ex) for i, ex in enumerate(all_examples)
                               if not ex.get("trajectory")]
            if not target_examples:
                # All samples have trajectory but we're in ti2v mode — use all, ignore trajectory
                target_examples = list(enumerate(all_examples))
        else:
            # --trajectory CLI override: apply same trajectory to all samples
            target_examples = list(enumerate(all_examples))

        # Data mode assigns one sample stream per rank. SP mode assigns one sample
        # stream per SP group; every rank in the group must run the same sample.
        my_examples = [(idx, ex) for j, (idx, ex) in enumerate(target_examples)
                       if j % dist_info.num_worker_groups == dist_info.worker_group_id]

    if dist_info.is_worker_leader:
        if args.serve_stdin:
            print(f"[rank {rank}] Serving stdin requests with one resident worker group.")
        else:
            print(f"[rank {rank}] Total examples: {len(all_examples)}, "
                  f"target examples: {len(target_examples)}, assigned: {len(my_examples)}")

    if not args.serve_stdin and not my_examples:
        if dist_info.is_worker_leader:
            print(f"[rank {rank}] No examples assigned, idle.")
        cleanup_dist(dist_info)
        return

    # ── Load encoders ONCE ──
    startup_t0 = time.perf_counter()
    from trainer.models.hyvideo.vae.hunyuanvideo_15_vae_w_cache import AutoencoderKLConv3D
    vae_path = os.path.join(model_path, "vae")
    vae = AutoencoderKLConv3D.from_pretrained(vae_path, torch_dtype=torch.float16).cpu()

    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
    text_encoder, _ = HunyuanVideo_1_5_Pipeline._load_text_encoders(model_path, device="cpu")
    vision_encoder = HunyuanVideo_1_5_Pipeline._load_vision_encoder(model_path, device="cpu")
    byt5_kwargs, _ = HunyuanVideo_1_5_Pipeline._load_byt5(model_path, True, 256, device="cpu")
    byt5_model = byt5_kwargs["byt5_model"]
    byt5_tokenizer = byt5_kwargs["byt5_tokenizer"]

    # ── Load diffusion model ONCE ──
    use_prope = camera_mode

    if use_prope:
        model = load_model_prope(args.transformer_dir, use_discrete_action=args.use_discrete_action)
    elif args.mode == "bidirectional":
        model = load_model_bidirectional(args.transformer_dir)
    else:
        model = load_model_ar_rollout(args.transformer_dir)

    if args.action_ckpt and not use_prope:
        state_dict = load_file(args.action_ckpt)
        model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(f"Loaded action ckpt: {args.action_ckpt}")

    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    startup_elapsed = elapsed_since(startup_t0, device)

    if args.mode == "bidirectional" and hasattr(model, "set_attn_mode"):
        model.set_attn_mode("flash")

    if dist_info.is_worker_leader:
        work_label = "stdin requests" if args.serve_stdin else f"{len(my_examples)} samples"
        print(f"Model loaded in {startup_elapsed:.1f}s. Processing {work_label}...")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Process each assigned sample ──
    chunk0_latencies = []  # worker leader: collect from 2nd prompt onward
    conditioning_cache = {}
    negative_prompt_cache = None
    request_iter = (
        iter_stdin_requests(args, dist_info)
        if args.serve_stdin
        else ((task_idx, sample_idx, example)
              for task_idx, (sample_idx, example) in enumerate(my_examples))
    )
    total_label = "stream" if args.serve_stdin else str(len(my_examples))

    for task_idx, sample_idx, example in request_iter:
        trajectory = (example.get("trajectory") or args.trajectory) if camera_mode else None
        if "_seed" in example:
            sample_seed = example["_seed"]
        else:
            sample_seed = None if args.seed < 0 else args.seed + sample_idx

        # Determine output path
        if example.get("_output_path"):
            output_path = example["_output_path"]
        elif trajectory:
            traj_safe = trajectory.replace("*", "").replace(",", "_")
            output_path = os.path.join(args.output_dir, f"sample_{sample_idx:03d}_{traj_safe}.mp4")
        else:
            output_path = os.path.join(args.output_dir, f"sample_{sample_idx:03d}.mp4")

        # Skip if already exists. In SP mode, synchronize the decision inside
        # the SP process group so ranks do not diverge before collectives.
        skip_existing = (not args.overwrite) and os.path.isfile(output_path)
        if dist_info.uses_sequence_parallel:
            from hyvideo.commons.parallel_states import get_parallel_state
            skip_tensor = torch.tensor([int(skip_existing)], device=device)
            dist.broadcast(skip_tensor, src=dist_info.worker_leader_rank,
                           group=get_parallel_state().sp_group)
            skip_existing = bool(skip_tensor.item())
        if skip_existing:
            if dist_info.is_worker_leader:
                print(f"[rank {rank}] Already exists, skipping: {output_path}")
            continue

        if dist_info.is_worker_leader:
            print(f"[rank {rank}] ({task_idx+1}/{total_label}) "
                  f"sample {sample_idx}: {example['caption'][:60]}...")

        # Encode inputs
        set_sample_seed(sample_seed)
        cache_key = conditioning_cache_key(example, args.height, args.width, args.video_length)
        cache_hit = args.conditioning_cache_size != 0 and cache_key in conditioning_cache
        encode_t0 = time.perf_counter()
        if cache_hit:
            data = conditioning_cache.pop(cache_key)
            conditioning_cache[cache_key] = data
            encode_elapsed = 0.0
        else:
            text_encoder.to(device)
            vision_encoder.to(device)
            if byt5_model is not None:
                byt5_model.to(device)

            data = prepare_sample_data(
                vae, text_encoder, vision_encoder, byt5_model, byt5_tokenizer,
                example, args.height, args.width, args.video_length, device,
            )

            if args.conditioning_cache_size > 0:
                conditioning_cache[cache_key] = data
                while len(conditioning_cache) > args.conditioning_cache_size:
                    conditioning_cache.pop(next(iter(conditioning_cache)))

            # Free encoders from GPU
            text_encoder.cpu()
            vision_encoder.cpu()
            if byt5_model is not None:
                byt5_model.cpu()
            torch.cuda.empty_cache()
            encode_elapsed = elapsed_since(encode_t0, device)

        neg_prompts = None
        if args.guidance_scale > 1.0:
            if negative_prompt_cache is None:
                neg_prompts = encode_negative_prompt(text_encoder, byt5_model, byt5_tokenizer, 256, device)
                negative_prompt_cache = {
                    key: value.cpu() if torch.is_tensor(value) else value
                    for key, value in neg_prompts.items()
                }
                text_encoder.cpu()
                if byt5_model is not None:
                    byt5_model.cpu()
                torch.cuda.empty_cache()
            else:
                neg_prompts = negative_prompt_cache

        # Build camera tensors if ProPE
        camera_t0 = time.perf_counter()
        viewmats, Ks = None, None
        if trajectory:
            T_lat = data["latent_shape"][2]
            viewmats, Ks = make_camera_tensors(trajectory)
            if viewmats.shape[1] > T_lat:
                viewmats = viewmats[:, :T_lat]
                Ks = Ks[:, :T_lat]
            elif viewmats.shape[1] < T_lat:
                pad_n = T_lat - viewmats.shape[1]
                viewmats = torch.cat([viewmats, viewmats[:, -1:].expand(-1, pad_n, -1, -1)], dim=1)
                Ks = torch.cat([Ks, Ks[:, -1:].expand(-1, pad_n, -1, -1)], dim=1)

        # Build discrete action labels if requested
        action = None
        if args.use_discrete_action and trajectory:
            from trainer.dataset_camera.action_utils import trajectory_str_to_action_labels
            T_lat = data["latent_shape"][2]
            action = trajectory_str_to_action_labels(trajectory, T_lat)
        camera_elapsed = time.perf_counter() - camera_t0

        # Run inference
        infer_breakdown = {}
        sync_cuda(device)
        infer_t0 = time.perf_counter()
        if args.mode == "bidirectional":
            x = run_inference_bidirectional(
                model, data, neg_prompts, device,
                args.num_inference_steps, args.shift, args.guidance_scale,
                viewmats, Ks, action, seed=None if sample_seed is None else sample_seed + 1,
            )
            infer_elapsed = elapsed_since(infer_t0, device)
            infer_breakdown["total"] = infer_elapsed
            if dist_info.is_worker_leader:
                if len(chunk0_latencies) >= 1:
                    chunk0_latencies.append(infer_elapsed)
                else:
                    chunk0_latencies.append(None)
        else:
            x, infer_breakdown = run_inference_rollout(
                model, data, neg_prompts, device,
                args.num_inference_steps, args.shift, args.guidance_scale,
                args.stabilization_level, args.chunk_latent_frames,
                viewmats, Ks, action, seed=None if sample_seed is None else sample_seed + 1,
                profile_timing=args.profile_timing,
            )
            infer_elapsed = elapsed_since(infer_t0, device)
            chunk0_lat = infer_breakdown.get("chunk0_latency")
            if dist_info.is_worker_leader:
                if len(chunk0_latencies) >= 1:
                    chunk0_latencies.append(chunk0_lat)
                else:
                    chunk0_latencies.append(None)

        if dist_info.is_worker_leader:
            if args.vae_decode_mode == "none":
                print(f"[rank {rank}] Inference done in {infer_elapsed:.3f}s, decode skipped.")
            else:
                print(f"[rank {rank}] Inference done in {infer_elapsed:.3f}s, decoding...")

        decode_elapsed = 0.0
        decode_saved = False
        if args.vae_decode_mode == "tile_parallel":
            sync_cuda(device)
            decode_t0 = time.perf_counter()
            decode_saved = decode_and_save(
                x, vae, device, output_path, args.fps,
                decode_mode=args.vae_decode_mode,
                dist_info=dist_info,
            )
            decode_elapsed = elapsed_since(decode_t0, device)
        elif dist_info.is_worker_leader and args.vae_decode_mode == "leader":
            sync_cuda(device)
            decode_t0 = time.perf_counter()
            decode_saved = decode_and_save(
                x, vae, device, output_path, args.fps,
                decode_mode=args.vae_decode_mode,
                dist_info=dist_info,
            )
            decode_elapsed = elapsed_since(decode_t0, device)

        if dist_info.is_worker_leader:
            if decode_saved:
                print(f"[rank {rank}] Saved {output_path}")

            frames = args.video_length
            playback_seconds = frames / float(args.fps)
            hot_fps = frames / infer_elapsed if infer_elapsed > 0 else float("inf")
            hot_x = playback_seconds / infer_elapsed if infer_elapsed > 0 else float("inf")
            e2e_elapsed = encode_elapsed + camera_elapsed + infer_elapsed + decode_elapsed
            e2e_fps = frames / e2e_elapsed if e2e_elapsed > 0 else float("inf")
            e2e_x = playback_seconds / e2e_elapsed if e2e_elapsed > 0 else float("inf")
            print(
                f"[benchmark] sample={sample_idx} frames={frames} "
                f"hot_inference={infer_elapsed:.3f}s "
                f"({hot_fps:.2f} output_fps, {hot_x:.2f}x playback) "
                f"sample_e2e={e2e_elapsed:.3f}s "
                f"({e2e_fps:.2f} output_fps, {e2e_x:.2f}x playback)",
                flush=True,
            )
            if args.profile_timing:
                parts = [
                    f"encode={encode_elapsed:.3f}s",
                    f"conditioning_cache={'hit' if cache_hit else 'miss'}",
                    f"camera={camera_elapsed:.3f}s",
                    f"decode_write={decode_elapsed:.3f}s",
                    f"vae_decode_mode={args.vae_decode_mode}",
                ]
                if infer_breakdown:
                    for key in ("input_to_gpu", "latent_init", "text_kv",
                                "denoise", "vision_cache", "total"):
                        if key in infer_breakdown:
                            parts.append(f"{key}={infer_breakdown[key]:.3f}s")
                print(f"[timing] sample={sample_idx} " + " ".join(parts), flush=True)

        # Clean up for next iteration
        del x, data, neg_prompts, viewmats, Ks
        torch.cuda.empty_cache()

    # All ranks done
    if dist_info.is_worker_leader:
        valid = [v for v in chunk0_latencies[1:] if v is not None]
        if valid:
            label = "full inference" if args.mode == "bidirectional" else "chunk0"
            print(f"[timing] rank {rank} {label} latency (from 2nd prompt): "
                  f"avg={sum(valid)/len(valid):.3f}s over {len(valid)} samples")

    cleanup_dist(dist_info)
    if rank == 0:
        print("All done.")


if __name__ == "__main__":
    main()
