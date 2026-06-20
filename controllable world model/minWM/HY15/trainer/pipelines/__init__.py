# SPDX-License-Identifier: Apache-2.0
"""
Diffusion pipelines for trainer.

This package contains diffusion pipelines for generating videos and images.
"""

from trainer.pipelines.composed_pipeline_base import ComposedPipelineBase
from trainer.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from trainer.pipelines.ar_hunyuan_training_pipeline import TrainingPipeline
from trainer.pipelines.ar_hunyuan_dmd_distill_pipeline import ARHunyuanDMDDistillationPipeline
from trainer.pipelines.ar_causal_cd_pipeline import ConsistencyDistillationPipeline

__all__ = [
    "ComposedPipelineBase",
    "ForwardBatch",
    "TrainingBatch",
    "TrainingPipeline",
    "ARHunyuanDMDDistillationPipeline",
    "ConsistencyDistillationPipeline",
]
