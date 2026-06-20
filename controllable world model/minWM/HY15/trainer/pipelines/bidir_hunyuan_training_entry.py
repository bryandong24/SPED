# SPDX-License-Identifier: Apache-2.0
import os
import sys
sys.path.append(os.path.abspath('.'))

from trainer.trainer_args import TrainerArgs, TrainingArgs
from trainer.logger import init_logger
from trainer.pipelines.ar_hunyuan_training_entry import HunyuanTrainingPipeline

logger = init_logger(__name__)


class BidirHunyuanTrainingPipeline(HunyuanTrainingPipeline):
    """Bidirectional training pipeline - uses non-causal attention."""

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)
        # Set non-causal attention mode for bidirectional training
        self.transformer.set_attn_mode('flash')
        logger.info("Bidirectional training: set attn_mode='flash'")


def main(args) -> None:
    logger.info("Starting bidirectional training pipeline...")
    pipeline = BidirHunyuanTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    pipeline.train()
    logger.info("Bidirectional training pipeline done")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    from trainer.trainer_args import TrainingArgs
    from trainer.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = TrainerArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
