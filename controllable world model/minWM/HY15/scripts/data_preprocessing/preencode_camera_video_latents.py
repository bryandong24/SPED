#!/usr/bin/env python3
"""
Preencode video latents with camera data for HunyuanVideo training.

Based on preencode_video_latents.py, adds raw camera data to .pt output.
Camera data is stored as-is for plucker embedding construction in the dataloader.

Input JSON format (from prepare_camera_json.py):
    [{"video_path": ..., "caption": ..., "intrinsics_path": ...,
      "poses_path": ..., "camera_indices": [...], "num_frames": ..., ...}]

Output .pt contains:
    latent, image_cond, prompt_embeds, prompt_mask, vision_states,
    byt5_text_states, byt5_text_mask,
    intrinsics (4,), poses (N_cam, 7), camera_indices (N_cam,), num_frames (int)

Usage:
    torchrun --nproc_per_node=8 scripts/data_preprocessing/preencode_camera_video_latents.py \
        --input_json /path/to/train_camera.json \
        --output_dir /path/to/output \
        --hunyuan_checkpoint_path /path/to/hunyuanvideo_1_5 \
        --max_frames 77
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

import argparse
import datetime
import json
import re
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
                        help="Path to comprehensive_analysis.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hunyuan_checkpoint_path", type=str, required=True)
    parser.add_argument("--target_height", type=int, default=480)
    parser.add_argument("--target_width", type=int, default=832)
    parser.add_argument("--max_frames", type=int, default=77)
    parser.add_argument("--min_frames", type=int, default=81)
    parser.add_argument("--max_cam_diff", type=int, default=6)
    parser.add_argument("--min_n_cam", type=int, default=15)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def sample_frame_indices(camera_indices, max_frames=77):
    """
    Sample max_frames video frame indices aligned to VAE 4x temporal downsampling.

    Strategy:
    - Select n_select = (max_frames-1)//4 + 1 camera frames (evenly from camera_indices)
    - First camera frame: [c_0]
    - For each subsequent camera frame c_i: 3 frames uniformly between c_{i-1} and c_i, then c_i
    - Total: 1 + (n_select-1)*4 = max_frames
    - Each group of 4 frames encodes to 1 latent; camera frame is the last of each group.

    Returns list of integer frame indices (length == max_frames).
    """
    cam = sorted(set(int(x) for x in camera_indices))
    n_cam = len(cam)
    n_select = (max_frames - 1) // 4 + 1  # 20 for max_frames=77

    if n_cam < n_select:
        return None  # not enough camera frames, skip this sample

    start = (n_cam - n_select) // 2
    selected = cam[start:start + n_select]  # take middle n_select camera frames

    result = [selected[0]]
    for i in range(1, len(selected)):
        a, b = selected[i - 1], selected[i]
        for j in range(1, 4):
            result.append(int(round(a + j * (b - a) / 4)))
        result.append(b)
    return result


def load_video(video_path, target_height, target_width, frame_indices=None, max_frames=77):
    """Load specific frame_indices from video, return [C, F, H, W] in [-1, 1]."""
    if USE_DECORD:
        vr = decord.VideoReader(video_path)
        if frame_indices is None:
            frame_indices = list(range(min(max_frames, len(vr))))
        else:
            frame_indices = [i for i in frame_indices if i < len(vr)]
        frames = vr.get_batch(frame_indices)
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        video_tensor = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
    else:
        cap = cv2.VideoCapture(video_path)
        if frame_indices is None:
            frame_indices = list(range(max_frames))
        frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"No frames from {video_path}")
        video_tensor = torch.from_numpy(np.stack(frames)).float().permute(3, 0, 1, 2)

    video_tensor = F.interpolate(
        video_tensor.unsqueeze(0),
        size=(video_tensor.shape[1], target_height, target_width),
        mode='trilinear', align_corners=False
    ).squeeze(0)

    video_tensor = (video_tensor / 255.0 - 0.5) * 2.0
    return video_tensor


class VideoLatentExtractor:
    """Extract latents from videos and text using HunyuanVideo encoders."""

    def __init__(self, checkpoint_path, device, target_size=(480, 832)):
        self.device = device
        self.target_size = target_size
        self._load_models(checkpoint_path)

    def _load_models(self, checkpoint_path):
        print(f"Loading models from {checkpoint_path}...")

        self.vae = AutoencoderKLConv3D.from_pretrained(
            os.path.join(checkpoint_path, "vae"),
            torch_dtype=torch.float32
        ).to(self.device).eval()

        self.vision_encoder = VisionEncoder(
            vision_encoder_type="siglip",
            vision_encoder_precision="fp16",
            vision_encoder_path=os.path.join(checkpoint_path, "vision_encoder/siglip"),
            processor_type=None, processor_path=None,
            output_key=None, logger=None, device=self.device,
        )

        self.text_encoder = TextEncoder(
            text_encoder_type="llm", tokenizer_type="llm",
            text_encoder_path=os.path.join(checkpoint_path, "text_encoder/llm"),
            max_length=1000, text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            hidden_state_skip_layer=2, apply_final_norm=False,
            reproduce=False, logger=None, device=self.device,
        )
        self.text_len = self.text_encoder.max_length

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
        byt5_embeddings = torch.zeros((1, self.byt5_max_length, 1472), device=self.device)
        byt5_mask = torch.zeros((1, self.byt5_max_length), device=self.device, dtype=torch.int64)

        pattern = r'\"(.*?)\"|"(.*?)"'
        matches = re.findall(pattern, prompt_text)
        glyph_texts = [m[0] or m[1] for m in matches]
        glyph_texts = list(dict.fromkeys(glyph_texts)) if len(glyph_texts) > 1 else glyph_texts

        if glyph_texts:
            text_styles = [{"color": None, "font-family": None} for _ in glyph_texts]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)
            inputs = self.byt5_tokenizer(
                formatted_text, padding="max_length", max_length=self.byt5_max_length,
                truncation=True, add_special_tokens=True, return_tensors="pt",
            )
            text_ids = inputs.input_ids.to(self.device)
            text_mask = inputs.attention_mask.to(self.device)
            byt5_embeddings = self.byt5_model(text_ids, attention_mask=text_mask.float())[0]
            byt5_mask = text_mask
        return byt5_embeddings, byt5_mask

    @torch.no_grad()
    def extract(self, video_path, caption, camera_indices, max_frames=77):
        frame_indices = sample_frame_indices(camera_indices, max_frames)
        if frame_indices is None:
            return None  # not enough camera frames
        video_tensor = load_video(
            video_path, self.target_size[0], self.target_size[1], frame_indices=frame_indices
        ).to(self.device)

        video_input = video_tensor.unsqueeze(0)
        video_latents = self.vae.encode(video_input).latent_dist.mode()
        video_latents = video_latents * self.vae.config.scaling_factor

        image_cond = video_latents[:, :, 0:1, :, :]

        first_frame = video_tensor[:, 0, :, :]
        first_frame_uint8 = ((first_frame + 1) * 127.5).clamp(0, 255)
        first_frame_np = first_frame_uint8.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        vision_states = self.vision_encoder.encode_images(first_frame_np)
        vision_states = vision_states.last_hidden_state.to(device=self.device, dtype=torch.bfloat16)

        text_inputs = self.text_encoder.text2tokens(caption, data_type="video", max_length=self.text_len)
        prompt_outputs = self.text_encoder.encode(text_inputs, data_type="video", device=self.device)
        prompt_embeds = prompt_outputs.hidden_state.to(dtype=self.text_encoder.dtype, device=self.device)
        attention_mask = prompt_outputs.attention_mask.to(self.device) if prompt_outputs.attention_mask is not None else None

        byt5_embeddings, byt5_masks = self._process_byt5_prompt(caption)

        return {
            "latent": video_latents.to(torch.bfloat16),
            "image_cond": image_cond.to(torch.bfloat16),
            "prompt_embeds": prompt_embeds,
            "prompt_mask": attention_mask,
            "vision_states": vision_states,
            "byt5_text_states": byt5_embeddings,
            "byt5_text_mask": byt5_masks,
            "frame_indices": frame_indices,
        }


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=2),
        )
        global_rank = torch.distributed.get_rank()
    else:
        global_rank = 0

    device = torch.cuda.current_device()

    latents_dir = os.path.join(args.output_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)

    with open(args.input_json) as f:
        raw_list = json.load(f)

    # Input JSON is pre-filtered (from merge_filtered_for_preencode.py).
    # Just load captions from caption_path if available.
    data_list = []
    for m in raw_list:
        caption = ""
        if m.get("caption_path"):
            try:
                with open(m["caption_path"]) as f:
                    caption = json.load(f).get("SceneSummary", "")
            except Exception:
                pass
        data_list.append({**m, "caption": caption})

    if global_rank == 0:
        print(f"Loaded {len(data_list)} items from {args.input_json}")

    total_items = len(data_list)
    if world_size > 1:
        per_gpu = (total_items + world_size - 1) // world_size
        start_idx = global_rank * per_gpu
        end_idx = min(start_idx + per_gpu, total_items)
        data_list = data_list[start_idx:end_idx]
    else:
        start_idx = 0

    print(f"GPU {local_rank}: Processing {len(data_list)} items")

    extractor = VideoLatentExtractor(
        args.hunyuan_checkpoint_path, device,
        (args.target_height, args.target_width),
    )

    output_items = []
    for idx, item in enumerate(tqdm(data_list, desc=f"GPU {local_rank}")):
        video_path = item.get("video_path")
        caption = item.get("caption", "")

        if not video_path or not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        global_idx = start_idx + idx
        video_id = item.get("video_id", f"{global_idx:06d}")
        output_filename = f"{video_id}.pt"
        output_path = os.path.join(latents_dir, output_filename)
        output_path_abs = os.path.abspath(output_path)

        if args.skip_existing and os.path.exists(output_path):
            output_items.append({"latent_path": output_path_abs})
            continue

        try:
            latents = extractor.extract(video_path, caption, item["camera_indices"], args.max_frames)

            if latents is None:
                print(f"Skipped (not enough camera frames): {video_path}")
                continue

            # Add raw camera data (stored as-is for plucker construction in dataloader)
            intrinsics_raw = np.load(item["intrinsics_path"]).astype(np.float32)
            poses_raw = np.load(item["poses_path"]).astype(np.float32)
            latents["intrinsics"] = torch.from_numpy(intrinsics_raw[0])          # (4,)
            latents["poses"] = torch.from_numpy(poses_raw)                       # (N_cam, 7)
            latents["camera_indices"] = torch.tensor(item["camera_indices"])      # (N_cam,)
            latents["video_frame_indices"] = torch.tensor(latents.pop("frame_indices"), dtype=torch.long)  # (max_frames,)
            latents["num_video_frames"] = item["num_frames"]                     # int
            latents["video_id"] = item.get("video_id", output_filename)

            torch.save(latents, output_path)
            output_items.append({"latent_path": output_path_abs})

            if idx == 0 and local_rank == 0:
                print(f"\nFirst item shapes:")
                for key, value in latents.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {value}")

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            import traceback; traceback.print_exc()
            continue

    output_json = os.path.join(args.output_dir, f"train_index_rank{global_rank}.json")
    with open(output_json, "w") as f:
        json.dump(output_items, f, indent=2)
    print(f"GPU {global_rank}: Saved {len(output_items)} items to {output_json}")

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        if global_rank == 0:
            all_items = []
            for rank in range(world_size):
                rank_json = os.path.join(args.output_dir, f"train_index_rank{rank}.json")
                if os.path.exists(rank_json):
                    with open(rank_json) as f:
                        all_items.extend(json.load(f))
                    os.remove(rank_json)
            all_items.sort(key=lambda x: x["latent_path"])
            final_json = os.path.join(args.output_dir, "train_index.json")
            with open(final_json, "w") as f:
                json.dump(all_items, f, indent=2)
            print(f"Merged {len(all_items)} items to {final_json}")
        dist.barrier()
        dist.destroy_process_group()

    print(f"GPU {local_rank}: Done!")


if __name__ == "__main__":
    main()
