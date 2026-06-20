import argparse
import math
import os
from omegaconf import OmegaConf
import wandb

from wan_trainer import (
    DiffusionTrainer, BidirectionalDiffusionTrainer, ODETrainer,
    ScoreDistillationTrainer, ConsistencyDistillationTrainer,
    CameraDiffusionTrainer, CameraBidirectionalDiffusionTrainer,
    CameraODETrainer, CameraConsistencyDistillationTrainer,
    CameraScoreDistillationTrainer
)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--tf", action="store_true")
    parser.add_argument("--sp_size", type=int, default=1, help="Sequence parallel size (1=off)")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("Wan21/configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.tf = args.tf
    config.sp_size = args.sp_size
    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    # Check effective batch size >= 16
    total_gpus = int(os.environ.get("WORLD_SIZE", 1))
    dp = total_gpus // config.sp_size
    ga = getattr(config, "grad_accum_steps", 1)
    bsz = dp * ga
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"[BSZ Check] gpus={total_gpus}, dp={dp}, effective_bsz={bsz}")
        assert bsz >= 16, (
            f"effective_bsz={bsz} < 16. Suggest: "
            f"--sp_size 1 with grad_accum_steps >= {math.ceil(16 / total_gpus)}, "
            f"or keep --sp_size {config.sp_size} with grad_accum_steps >= {math.ceil(16 / dp)}"
        )

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "bidirectional_diffusion":
        trainer = BidirectionalDiffusionTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "consistency_distillation":
        trainer = ConsistencyDistillationTrainer(config)
    elif config.trainer == "camera_diffusion":
        trainer = CameraDiffusionTrainer(config)
    elif config.trainer == "camera_bidirectional_diffusion":
        trainer = CameraBidirectionalDiffusionTrainer(config)
    elif config.trainer == "camera_ode":
        trainer = CameraODETrainer(config)
    elif config.trainer == "camera_consistency_distillation":
        trainer = CameraConsistencyDistillationTrainer(config)
    elif config.trainer == "camera_score_distillation":
        trainer = CameraScoreDistillationTrainer(config)
    else:
        raise ValueError(f"Unknown trainer: {config.trainer}")
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
