# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import os

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import loguru
import torch
import argparse
import einops
import imageio
# Removed: camera control imports
# import json
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# from PIL import Image, ImageDraw, ImageFont
# from moviepy.editor import VideoFileClip, VideoClip

from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state
# Removed: camera control import
# from hyvideo.generate_custom_trajectory import generate_camera_trajectory_local

parallel_dims = initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


# Removed: camera control (mapping and pose parsing functions)
# mapping = { ... }
# def one_hot_to_one_dimension(...)
# def parse_pose_string(...)
# def pose_string_to_json(...)
# def pose_to_input(...)


def save_video(video, path):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, "c f h w -> f h w c")
    imageio.mimwrite(path, vid, fps=24)


def rank0_log(message, level):
    if int(os.environ.get("RANK", "0")) == 0:
        loguru.logger.log(level, message)


def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True  # When --flag is provided without value, enable it
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")



def generate_video(args):
    assert (
        (args.video_length - 1) // 4 + 1
    ) % 4 == 0, "number of latents must be divisible by 4"
    initialize_infer_state(args)

    task = "i2v" if args.image_path else "t2v"

    enable_sr = args.sr

    # Build transformer_version based on flags
    transformer_version = f"{args.resolution}_{task}"
    assert transformer_version == "480p_i2v"

    if args.dtype == "bf16":
        transformer_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Must be 'bf16' or 'fp32'")

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        create_sr_pipeline=enable_sr,
        force_sparse_attn=False,
        transformer_dtype=transformer_dtype,
        action_ckpt=args.action_ckpt,
    )

    extra_kwargs = {}
    if task == "i2v":
        extra_kwargs["reference_image"] = args.image_path

    enable_rewrite = args.rewrite
    if not args.rewrite:
        rank0_log(
            "Warning: Prompt rewriting is disabled. This may affect the quality of generated videos.",
            "WARNING",
        )

    if task == "i2v":
        extra_kwargs["reference_image"] = args.image_path

    out = pipe(
        enable_sr=enable_sr,
        prompt=args.prompt,
        aspect_ratio=args.aspect_ratio,
        num_inference_steps=args.num_inference_steps,
        sr_num_inference_steps=None,
        video_length=args.video_length,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        output_type="pt",
        prompt_rewrite=enable_rewrite,
        return_pre_sr_video=args.save_pre_sr_video,
        few_step=args.few_step,
        chunk_latent_frames=4 if args.model_type == "ar" else 16,
        model_type=args.model_type,
        user_height=args.height,
        user_width=args.width,
        transformer_resident_ar_rollout=args.transformer_resident_ar_rollout,
        solver=args.solver,
        **extra_kwargs,
    )

    # save video
    if int(os.environ.get("RANK", "0")) == 0:
        output_path = args.output_path
        os.makedirs(output_path, exist_ok=True)

        save_video_path = os.path.join(output_path, "gen.mp4")
        save_video_sr_path = os.path.join(output_path, "gen_sr.mp4")

        # Determine which video to process for UI overlay
        video_to_process = None
        final_video_path = None

        if enable_sr and hasattr(out, "sr_videos"):
            save_video(out.sr_videos, save_video_sr_path)
            print(f"Saved SR video to: {save_video_sr_path}")
            video_to_process = save_video_sr_path
            final_video_path = save_video_sr_path

            if args.save_pre_sr_video:
                save_video(out.videos, save_video_path)
                print(f"Saved original video (before SR) to: {save_video_path}")
        else:
            save_video(out.videos, save_video_path)
            print(f"Saved video to: {save_video_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate video using HunyuanWorld-1.5"
    )

    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for video generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for video generation (default: empty string)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        required=True,
        choices=["480p", "720p"],
        help="Video resolution (480p or 720p)",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to pretrained model"
    )
    parser.add_argument(
        "--action_ckpt", type=str, help="Path to pretrained action model"
    )
    parser.add_argument(
        "--aspect_ratio", type=str, default="16:9", help="Aspect ratio (default: 16:9)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=127,
        help="Number of frames to generate (default: 127)",
    )
    parser.add_argument(
        "--sr",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable super resolution (default: true). "
        "Use --sr or --sr true/1 to enable, --sr false/0 to disable",
    )
    parser.add_argument(
        "--save_pre_sr_video",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Save original video before super resolution (default: false). "
        "Use --save_pre_sr_video or --save_pre_sr_video true/1 to enable, "
        "--save_pre_sr_video false/0 to disable",
    )
    parser.add_argument(
        "--rewrite",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable prompt rewriting (default: true). "
        "Use --rewrite or --rewrite true/1 to enable, --rewrite false/0 to disable",
    )
    parser.add_argument(
        "--offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable offloading (default: true). "
        "Use --offloading or --offloading true/1 to enable, "
        "--offloading false/0 to disable",
    )
    parser.add_argument(
        "--group_offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="Enable group offloading (default: None, automatically enabled if offloading is enabled). "
        "Use --group_offloading or --group_offloading true/1 to enable, "
        "--group_offloading false/0 to disable",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Data type for transformer (default: bf16). "
        "bf16: faster, lower memory; fp32: better quality, slower, higher memory",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (default: 123)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to reference image for i2v (if provided, uses i2v mode)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output file path for generated video (if not provided, saves to ./outputs/output.mp4)",
    )
    parser.add_argument(
        "--enable_torch_compile",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable torch compile for transformer (default: false). "
        "Use --enable_torch_compile or --enable_torch_compile true/1 to enable, "
        "--enable_torch_compile false/0 to disable",
    )
    parser.add_argument(
        "--few_step",
        type=str_to_bool,
        nargs="?",
        const=False,
        default=False,
        help="Enable super resolution (default: true). "
        "Use --few_step or --few_step true/1 to enable, --few_step false/0 to disable",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["bi", "ar"],
        help="inference bidirectional or autoregressive model. ",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="height for generation (recommended to set as 480)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="width for generation (recommended to set as 832)",
    )

    parser.add_argument(
        "--use_sageattn",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable sageattn (default: false). "
        "Use --use_sageattn or --use_sageattn true/1 to enable, "
        "--use_sageattn false/0 to disable",
    )
    parser.add_argument(
        "--sage_blocks_range",
        type=str,
        default="0-53",
        help="Sageattn blocks range (e.g., 0-5 or 0,1,2,3,4,5)",
    )
    parser.add_argument(
        "--use_vae_parallel",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable vae parallel (default: false). "
        "Use --use_vae_parallel or --use_vae_parallel true/1 to enable, "
        "--use_vae_parallel false/0 to disable",
    )
    # fp8 gemm related
    parser.add_argument(
        "--use_fp8_gemm",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable fp8 gemm for transformer (default: false). "
        "Use --use_fp8_gemm or --use_fp8_gemm true/1 to enable, "
        "--use_fp8_gemm false/0 to disable",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="fp8-per-block",
        help="Quantization type for fp8 gemm (e.g., fp8-per-tensor-weight-only, fp8-per-tensor, fp8-per-block)",
    )
    parser.add_argument(
        "--include_patterns",
        type=str,
        default="double_blocks",
        help="Include patterns for fp8 gemm (default: double_blocks)",
    )
    parser.add_argument(
        "--transformer_resident_ar_rollout",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Keep transformer on GPU for entire AR rollout instead of per-chunk offloading (default: false). "
        "Reduces inference time without increasing peak VRAM. Only affects AR model_type with offloading enabled. "
        "Use --transformer_resident_ar_rollout or --transformer_resident_ar_rollout true to enable.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="cm",
        choices=["euler", "cm"],
        help="Solver for AR denoising steps (default: cm). "
        "'euler': standard Euler ODE solver; 'cm': consistency model solver.",
    )

    args = parser.parse_args()

    assert args.image_path is not None

    generate_video(args)


if __name__ == "__main__":
    main()
