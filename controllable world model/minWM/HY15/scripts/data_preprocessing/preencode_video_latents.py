"""
Preencode video latents for HunyuanVideo training.

IMPORTANT FIX:
- image_cond is extracted from video_latents[:, :, 0:1, :, :] (first frame)
- vision_states is encoded from video's first frame
This ensures consistency with official training code and avoids first-frame blur.

Usage:
    python scripts/data_preprocessing/preencode_video_latents.py \
        --input_json /path/to/videos.json \
        --output_dir /path/to/output \
        --hunyuan_checkpoint_path /path/to/hunyuanvideo_1_5

Input JSON format:
    [
        {"video_path": "/path/to/video1.mp4", "caption": "A cat playing"},
        {"video_path": "/path/to/video2.mp4", "caption": "A sunset"}
    ]

Output:
    - {output_dir}/latents/{item_id}.pt
    - {output_dir}/train_index.json
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

import argparse
import json
import re
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Use decord for video loading (no OpenGL dependency)
try:
    import decord
    decord.bridge.set_bridge('torch')
    USE_DECORD = True
except ImportError:
    USE_DECORD = False
    import cv2

from hyvideo.models.autoencoders.hunyuanvideo_15_vae_w_cache import AutoencoderKLConv3D
from hyvideo.models.vision_encoder import VisionEncoder
from hyvideo.models.text_encoders import PROMPT_TEMPLATE, TextEncoder
from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2
from hyvideo.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True,
                        help="Input JSON with video_path and caption")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for latent files")
    parser.add_argument("--hunyuan_checkpoint_path", type=str, required=True,
                        help="Path to HunyuanVideo checkpoint")
    parser.add_argument("--target_height", type=int, default=480)
    parser.add_argument("--target_width", type=int, default=832)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames to encode (None = all frames)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip if output file exists")
    return parser.parse_args()


def load_video(video_path, target_height, target_width, max_frames=None):
    """Load video and return as tensor [C, F, H, W] normalized to [-1, 1].

    Args:
        video_path: Path to video file
        target_height: Target height
        target_width: Target width
        max_frames: Maximum frames to load (None = all)

    Returns:
        video_tensor: [3, F, H, W] in range [-1, 1]
    """
    if USE_DECORD:
        # Use decord (no OpenGL dependency)
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        if max_frames:
            num_frames = min(max_frames, total_frames)
        else:
            num_frames = total_frames

        # Load frames
        indices = list(range(num_frames))
        frames = vr.get_batch(indices)  # Returns torch tensor when bridge='torch'

        # Convert to numpy if needed
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()

        # Convert to tensor
        video_tensor = torch.from_numpy(frames).float()  # [F, H, W, C]
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # [C, F, H, W]
    else:
        # Fallback to cv2
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if max_frames and len(frames) >= max_frames:
                break

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {video_path}")

        # Stack frames: [F, H, W, C]
        video_np = np.stack(frames, axis=0)

        # Convert to tensor and normalize
        video_tensor = torch.from_numpy(video_np).float()  # [F, H, W, C]
        video_tensor = video_tensor.permute(3, 0, 1, 2)    # [C, F, H, W]

    # Resize
    video_tensor = F.interpolate(
        video_tensor.unsqueeze(0),  # [1, C, F, H, W]
        size=(video_tensor.shape[1], target_height, target_width),
        mode='trilinear',
        align_corners=False
    ).squeeze(0)  # [C, F, H, W]

    # Normalize to [-1, 1]
    video_tensor = (video_tensor / 255.0 - 0.5) * 2.0

    return video_tensor


class VideoLatentExtractor:
    """Extract latents from videos and text using HunyuanVideo encoders."""

    def __init__(self, checkpoint_path, device, target_size=(480, 832)):
        self.device = device
        self.target_size = target_size  # (height, width)
        self._load_models(checkpoint_path)

    def _load_models(self, checkpoint_path):
        """Load all encoder models."""
        print(f"Loading models from {checkpoint_path}...")

        # VAE
        self.vae = AutoencoderKLConv3D.from_pretrained(
            os.path.join(checkpoint_path, "vae"),
            torch_dtype=torch.float32
        ).to(self.device).eval()

        # Vision Encoder (SigLIP)
        self.vision_encoder = VisionEncoder(
            vision_encoder_type="siglip",
            vision_encoder_precision="fp16",
            vision_encoder_path=os.path.join(checkpoint_path, "vision_encoder/siglip"),
            processor_type=None,
            processor_path=None,
            output_key=None,
            logger=None,
            device=self.device,
        )

        # Text Encoder (LLM)
        self.text_encoder = TextEncoder(
            text_encoder_type="llm",
            tokenizer_type="llm",
            text_encoder_path=os.path.join(checkpoint_path, "text_encoder/llm"),
            max_length=1000,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=None,
            device=self.device,
        )
        self.text_len = self.text_encoder.max_length

        # byT5
        load_from = os.path.join(checkpoint_path, "text_encoder")
        glyph_root = os.path.join(load_from, "Glyph-SDXL-v2")
        byt5_args = dict(
            byT5_google_path=os.path.join(load_from, "byt5-small"),
            byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
            multilingual_prompt_format_color_path=os.path.join(glyph_root, "assets/color_idx.json"),
            multilingual_prompt_format_font_path=os.path.join(glyph_root, "assets/multilingual_10-lang_idx.json"),
            byt5_max_length=256,
        )
        byt5_kwargs = load_glyph_byT5_v2(
            byt5_args,
            device=f"cuda:{self.device}" if isinstance(self.device, int) else str(self.device),
        )
        self.prompt_format = MultilingualPromptFormat(
            font_path=byt5_args["multilingual_prompt_format_font_path"],
            color_path=byt5_args["multilingual_prompt_format_color_path"],
        )
        self.byt5_model = byt5_kwargs["byt5_model"]
        self.byt5_tokenizer = byt5_kwargs["byt5_tokenizer"]
        self.byt5_max_length = byt5_kwargs["byt5_max_length"]

        print("All models loaded.")

    def _process_byt5_prompt(self, prompt_text):
        """Process text prompt through byT5 encoder."""
        byt5_embeddings = torch.zeros(
            (1, self.byt5_max_length, 1472), device=self.device
        )
        byt5_mask = torch.zeros(
            (1, self.byt5_max_length), device=self.device, dtype=torch.int64
        )

        # Extract quoted text (glyph text)
        pattern = r'\"(.*?)\"|"(.*?)"'
        matches = re.findall(pattern, prompt_text)
        glyph_texts = [m[0] or m[1] for m in matches]
        glyph_texts = list(dict.fromkeys(glyph_texts)) if len(glyph_texts) > 1 else glyph_texts

        if glyph_texts:
            text_styles = [{"color": None, "font-family": None} for _ in glyph_texts]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)
            inputs = self.byt5_tokenizer(
                formatted_text,
                padding="max_length",
                max_length=self.byt5_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_ids = inputs.input_ids.to(self.device)
            text_mask = inputs.attention_mask.to(self.device)
            byt5_embeddings = self.byt5_model(text_ids, attention_mask=text_mask.float())[0]
            byt5_mask = text_mask

        return byt5_embeddings, byt5_mask

    @torch.no_grad()
    def extract(self, video_path, caption, max_frames=None):
        """Extract all latents from a video and its caption.

        IMPORTANT: image_cond is extracted from video_latents first frame,
        NOT from a separate image file. This matches official training code.

        Args:
            video_path: Path to video file
            caption: Text caption for the video
            max_frames: Max frames to encode

        Returns:
            dict containing all latents
        """
        # Step 1: Load video
        video_tensor = load_video(
            video_path,
            self.target_size[0],
            self.target_size[1],
            max_frames
        )  # [C, F, H, W]
        video_tensor = video_tensor.to(self.device)

        # Step 2: VAE encode video
        # Input: [B, C, F, H, W]
        video_input = video_tensor.unsqueeze(0)  # [1, 3, F, H, W]
        video_latents = self.vae.encode(video_input).latent_dist.mode()
        video_latents = video_latents * self.vae.config.scaling_factor
        # Output: [1, 32, F_latent, H_latent, W_latent]

        # Step 3: Extract image_cond from video_latents first frame
        # THIS IS THE KEY FIX: use video latents, not separate image encoding
        image_cond = video_latents[:, :, 0:1, :, :]  # [1, 32, 1, H, W]

        # Step 4: Vision encoder from video's first frame
        # Extract first frame from video tensor
        first_frame = video_tensor[:, 0, :, :]  # [C, H, W]
        # Convert to [0, 255] for vision encoder
        first_frame_uint8 = ((first_frame + 1) * 127.5).clamp(0, 255)
        first_frame_np = first_frame_uint8.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

        vision_states = self.vision_encoder.encode_images(first_frame_np)
        vision_states = vision_states.last_hidden_state.to(
            device=self.device, dtype=torch.bfloat16
        )

        # Step 5: Text encoder (LLM)
        text_inputs = self.text_encoder.text2tokens(
            caption, data_type="video", max_length=self.text_len
        )
        prompt_outputs = self.text_encoder.encode(
            text_inputs, data_type="video", device=self.device
        )
        prompt_embeds = prompt_outputs.hidden_state.to(
            dtype=self.text_encoder.dtype, device=self.device
        )
        attention_mask = (
            prompt_outputs.attention_mask.to(self.device)
            if prompt_outputs.attention_mask is not None
            else None
        )

        # Step 6: byT5 encoder
        byt5_embeddings, byt5_masks = self._process_byt5_prompt(caption)

        return {
            "latent": video_latents.to(torch.bfloat16),
            "image_cond": image_cond.to(torch.bfloat16),
            "prompt_embeds": prompt_embeds,
            "prompt_mask": attention_mask,
            "vision_states": vision_states,
            "byt5_text_states": byt5_embeddings,
            "byt5_text_mask": byt5_masks,
        }


def main():
    args = parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize distributed process group if multi-GPU
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        global_rank = torch.distributed.get_rank()
    else:
        global_rank = 0

    device = torch.cuda.current_device()

    # Create output directories (same structure as reference)
    latents_dir = os.path.join(args.output_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)

    # Load input data
    with open(args.input_json, "r") as f:
        data_list = json.load(f)

    total_items = len(data_list)

    # Distribute data across GPUs
    if world_size > 1:
        per_gpu = (total_items + world_size - 1) // world_size
        start_idx = global_rank * per_gpu
        end_idx = min(start_idx + per_gpu, total_items)
        data_list = data_list[start_idx:end_idx]
    else:
        start_idx = 0

    print(f"GPU {local_rank}: Processing {len(data_list)} items")

    # Initialize extractor
    extractor = VideoLatentExtractor(
        args.hunyuan_checkpoint_path,
        device,
        (args.target_height, args.target_width),
    )

    # Process each item
    output_items = []
    for idx, item in enumerate(tqdm(data_list, desc=f"GPU {local_rank}")):
        video_path = item.get("video_path")
        caption = item.get("caption", "")

        if not video_path or not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        # Calculate global index for consistent naming across GPUs
        global_idx = start_idx + idx

        # Use 6-digit zero-padded naming: 000000.pt, 000001.pt, etc.
        output_filename = f"{global_idx:06d}.pt"
        output_path = os.path.join(latents_dir, output_filename)

        # Convert to absolute path for train_index.json
        output_path_abs = os.path.abspath(output_path)

        # Skip if already processed
        if args.skip_existing and os.path.exists(output_path):
            output_items.append({
                "latent_path": output_path_abs,
            })
            continue

        try:
            latents = extractor.extract(video_path, caption, args.max_frames)
            torch.save(latents, output_path)
            output_items.append({
                "latent_path": output_path_abs,
            })

            if idx == 0 and local_rank == 0:
                # Print shapes for first item
                print(f"\nFirst item shapes:")
                for key, value in latents.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {value.shape}")

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save index file per rank
    output_json = os.path.join(args.output_dir, f"train_index_rank{global_rank}.json")
    with open(output_json, "w") as f:
        json.dump(output_items, f, indent=2)

    print(f"GPU {global_rank}: Saved {len(output_items)} items to {output_json}")

    # Merge index files if multi-GPU
    if world_size > 1:
        import torch.distributed as dist

        # Wait for all GPUs to finish processing (even if some finish early)
        print(f"GPU {global_rank}: Waiting for all GPUs to complete...")
        dist.barrier()

        if global_rank == 0:
            print("Rank 0: Merging index files...")
            all_items = []
            for rank in range(world_size):
                rank_json = os.path.join(args.output_dir, f"train_index_rank{rank}.json")
                if os.path.exists(rank_json):
                    with open(rank_json, "r") as f:
                        all_items.extend(json.load(f))
                    os.remove(rank_json)

            # Sort by latent_path to ensure consistent ordering
            all_items.sort(key=lambda x: x["latent_path"])

            final_json = os.path.join(args.output_dir, "train_index.json")
            with open(final_json, "w") as f:
                json.dump(all_items, f, indent=2)
            print(f"Merged {len(all_items)} items to {final_json}")

        # Wait for rank 0 to finish merging before destroying process group
        dist.barrier()

        # Clean up distributed process group
        dist.destroy_process_group()

    print(f"GPU {local_rank}: Done!")


if __name__ == "__main__":
    main()
