# SPDX-License-Identifier: Apache-2.0
"""
Self-Forcing DMD distillation training entry point for HY-WorldPlay AR models.

This script provides the main entry point for training with the DMD (Diffusion Model Distillation)
pipeline. It follows the same pattern as ar_hunyuan_training_entry.py but uses
ARHunyuanDMDDistillationPipeline instead.
"""
from copy import deepcopy
import os
import sys
sys.path.append(os.path.abspath('.'))

from trainer.trainer_args import TrainerArgs, TrainingArgs
from trainer.logger import init_logger
from trainer.pipelines.ar_hunyuan_dmd_distill_pipeline import ARHunyuanDMDDistillationPipeline

logger = init_logger(__name__)


class HunyuanDMDDistillationPipeline(ARHunyuanDMDDistillationPipeline):
    """
    A DMD distillation training pipeline for Hunyuan with Self-Forcing.
    """
    _required_config_modules = ["transformer"]

    def initialize_pipeline(self, trainer_args: TrainerArgs):
        pass
        # self.modules["scheduler"] = FlowUniPCMultistepScheduler(
        #     shift=trainer_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        pass


def main(args) -> None:
    logger.info("Starting Self-Forcing DMD distillation training pipeline...")

    pipeline = HunyuanDMDDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("DMD distillation training pipeline done")


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
