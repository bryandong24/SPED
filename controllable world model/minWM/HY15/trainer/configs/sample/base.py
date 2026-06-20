# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any

from trainer.logger import init_logger


logger = init_logger(__name__)


@dataclass
class SamplingParam:
    """
    Sampling parameters for video generation.
    """
    # All fields below are copied from ForwardBatch
    data_type: str = "video"

    # Image inputs
    image_path: str | None = None

    # Text inputs
    prompt: str | list[str] | None = None
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    prompt_path: str | None = None
    output_path: str = "outputs/"
    output_video_name: str | None = None

    # Batch info
    num_videos_per_prompt: int = 1
    seed: int = 1024

    # Original dimensions (before VAE scaling)
    num_frames: int = 125
    num_frames_round_down: bool = False  # Whether to round down num_frames if it's not divisible by num_gpus
    height: int = 720
    width: int = 1280
    fps: int = 24

    # Denoising parameters
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_rescale: float = 0.0

    # Misc
    save_video: bool = True
    return_frames: bool = False

    def __post_init__(self) -> None:
        self.data_type = "video" if self.num_frames > 1 else "image"

    def check_sampling_param(self):
        if self.prompt_path and not self.prompt_path.endswith(".txt"):
            raise ValueError("prompt_path must be a txt file")

    def update(self, source_dict: dict[str, Any]) -> None:
        for key, value in source_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.exception("%s has no attribute %s",
                                 type(self).__name__, key)

        self.__post_init__()
