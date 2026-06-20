# SPDX-License-Identifier: Apache-2.0
"""
Consistency Distillation (CD) training entry point for causal AR models.

Thin shell entry point following the same pattern as:
    - ar_causal_ode_training_entry.py
    - ar_hunyuan_dmd_distill_entry.py

Usage:
    torchrun --nproc_per_node=8 \
        trainer/pipelines/ar_causal_cd_entry.py \
        --cls_name HunyuanTransformer3DARActionModel \
        --causal \
        ...
"""
from copy import deepcopy
import os
import sys
sys.path.append(os.path.abspath('.'))

from trainer.trainer_args import TrainerArgs, TrainingArgs
from trainer.logger import init_logger
from trainer.pipelines.ar_causal_cd_pipeline import ConsistencyDistillationPipeline

logger = init_logger(__name__)


class HunyuanConsistencyDistillationPipeline(ConsistencyDistillationPipeline):
    """
    Consistency Distillation pipeline for HunyuanVideo causal AR models.

    Thin subclass that only specifies which modules to load.
    All training logic lives in ConsistencyDistillationPipeline.
    """
    _required_config_modules = ["transformer", "vae"]

    def initialize_pipeline(self, trainer_args: TrainerArgs):
        pass

    def create_training_stages(self, training_args: TrainingArgs):
        """May be used in future refactors."""
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        pass


def main(args) -> None:
    logger.info("Starting Consistency Distillation training pipeline...")

    pipeline = HunyuanConsistencyDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Consistency Distillation training pipeline done")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    argv = sys.argv
    from trainer.trainer_args import TrainingArgs
    from trainer.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = TrainerArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
