# SPDX-License-Identifier: Apache-2.0
import os
from abc import ABC
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm


from trainer.distributed.parallel_state import (get_sp_parallel_rank,
                                                  get_sp_world_size)

from trainer.dataset_camera import build_inference_dataloader
from trainer.dataset_camera.dataloader.schema import pyarrow_schema_t2v
from trainer.distributed import (cleanup_dist_env_and_memory,
                                   get_local_torch_device, get_sp_group,
                                   get_world_group)
from trainer.trainer_args import TrainerArgs, TrainingArgs
from trainer.logger import init_logger
from trainer.pipelines import ComposedPipelineBase, TrainingBatch
from trainer.training.training_utils import normalize_dit_input
from trainer.utils import set_random_seed

logger = init_logger(__name__)


def _get_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_tensor_by_mask(tensor_1, tensor_2, mask, dim):
    assert tensor_1.shape == tensor_2.shape
    # Mask is a 0/1 vector. Choose tensor_2 when the value is 1; otherwise, tensor_1
    masked_indices = torch.nonzero(mask).squeeze(1)
    tmp = tensor_1.clone()
    if dim == 0:
        tmp[masked_indices] = tensor_2[masked_indices]
    elif dim == 1:
        tmp[:, masked_indices] = tensor_2[:, masked_indices]
    elif dim == 2:
        tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
    return tmp

class CausalODESamplingPipeline(ComposedPipelineBase, ABC):
    """
    A pipeline for training a model. All training pipelines should inherit from this class.
    All reusable components and code should be implemented in this class.
    """
    _required_config_modules = ["scheduler", "transformer"]
    validation_pipeline: ComposedPipelineBase
    train_dataloader: StatefulDataLoader
    train_loader_iter: Iterator[dict[str, Any]]

    def __init__(
            self,
            model_path: str,
            trainer_args: TrainingArgs,
            required_config_modules: list[str] | None = None,
            loaded_modules: dict[str, torch.nn.Module] | None = None) -> None:
        trainer_args.inference_mode = False
        self.lora_training = trainer_args.lora_training
        if self.lora_training and trainer_args.lora_rank is None:
            raise ValueError("lora rank must be set when using lora training")

        set_random_seed(trainer_args.seed)  # for lora param init
        super().__init__(model_path, trainer_args, required_config_modules,
                         loaded_modules)  # type: ignore
    
    def initialize_pipeline(self, trainer_args: TrainerArgs):
        pass

    def create_training_stages(self, training_args: TrainingArgs):
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        pass

    def create_pipeline_stages(self, trainer_args: TrainerArgs):
        raise RuntimeError(
            "create_pipeline_stages should not be called for training pipeline")

    def set_schemas(self) -> None:
        self.train_dataset_schema = pyarrow_schema_t2v

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing training pipeline...")
        self.device = get_local_torch_device()
        self.training_args = training_args
        world_group = get_world_group()
        self.world_size = world_group.world_size
        self.global_rank = world_group.rank
        self.sp_group = get_sp_group()
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size
        self.local_rank = world_group.local_rank
        self.transformer = self.get_module("transformer")
        self.seed = training_args.seed
        self.set_schemas()
        # self.action = training_args.action  # Removed: camera control
        # add the causal option
        self.causal = training_args.causal
        self.train_time_shift = training_args.train_time_shift

        # Set random seeds for deterministic training
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed)

        self.transformer.set_attn_mode("flex_tf")
        # hardcoded cfg
        self.sample_cfg = 5.0
        self.train_dataset, self.train_dataloader = build_inference_dataloader(
            json_path=training_args.json_path,
            causal=training_args.causal,
            window_frames=training_args.window_frames,
            batch_size=training_args.train_batch_size,
            num_data_workers=training_args.dataloader_num_workers,
            drop_last=True,
            drop_first_row=False,
            seed=self.seed,
            cfg_rate=training_args.training_cfg_rate,
            i2v_rate=training_args.i2v_rate,
        )

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        raise NotImplementedError(
            "Training pipelines must implement this method")

    def _prepare_training(self, training_batch: TrainingBatch) -> TrainingBatch:
        return training_batch

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            # ODE sampling: dataloader exhausted -> signal end-of-pass to caller.
            # (Training pipelines reset the iterator here; sampling must not, or it
            # would re-sample the same data and overwrite previous outputs.)
            training_batch.end_of_pass = True
            return training_batch

        latents = batch["latent"]
        prompt_embed = batch["prompt_embed"]

        # Removed: camera control (w2c, intrinsic, action) processing
        video_path = batch.get('video_path', batch.get('path'))
        image_cond = batch.get('image_cond')
        vision_states = batch.get('vision_states')
        prompt_mask = batch.get('prompt_mask')
        byt5_text_states = batch.get('byt5_text_states')
        byt5_text_mask = batch.get('byt5_text_mask')
        neg_prompt_embed = batch.get('neg_prompt_embed')
        neg_prompt_mask = batch.get('neg_prompt_mask')
        neg_byt5_text_states = batch.get('neg_byt5_text_states')
        neg_byt5_text_mask = batch.get('neg_byt5_text_mask')
        # add an indicator for memory training
        select_window_out_flag = batch.get('select_window_out_flag', 0)
        i2v_mask = batch.get('i2v_mask')
        viewmats = batch.get('viewmats')
        Ks = batch.get('Ks')

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.prompt_embed = prompt_embed.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.text_states = prompt_embed.to(
            get_local_torch_device(), dtype=torch.bfloat16)
        training_batch.video_path = video_path[0] if isinstance(video_path, list) else video_path

        # Removed: camera control (w2c, intrinsic, action)
        if image_cond is not None:
            training_batch.image_cond = image_cond.to(
                get_local_torch_device(), dtype=torch.bfloat16)
        if vision_states is not None:
            training_batch.vision_states = vision_states.to(
                get_local_torch_device(), dtype=torch.bfloat16)
        if prompt_mask is not None:
            training_batch.prompt_mask = prompt_mask.to(
                get_local_torch_device(), dtype=torch.bfloat16)
        if byt5_text_states is not None:
            training_batch.byt5_text_states = byt5_text_states.to(
                get_local_torch_device(), dtype=torch.bfloat16)
        if byt5_text_mask is not None:
            training_batch.byt5_text_mask = byt5_text_mask.to(
                get_local_torch_device(), dtype=torch.bfloat16)
        if neg_prompt_embed is not None:
            training_batch.neg_prompt_embed = neg_prompt_embed.to(
                get_local_torch_device(), dtype=torch.bfloat16)
        if neg_prompt_mask is not None:
            training_batch.neg_prompt_mask = neg_prompt_mask.to(
                get_local_torch_device(), dtype=torch.bfloat16)
        if neg_byt5_text_states is not None:
            training_batch.neg_byt5_text_states = neg_byt5_text_states.to(
                get_local_torch_device(), dtype=torch.bfloat16)
        if neg_byt5_text_mask is not None:
            training_batch.neg_byt5_text_mask = neg_byt5_text_mask.to(
                get_local_torch_device(), dtype=torch.bfloat16)
        # Set neg_text_states and neg_text_mask for CFG
        training_batch.neg_text_states = training_batch.neg_prompt_embed
        training_batch.neg_text_mask = training_batch.neg_prompt_mask
        if select_window_out_flag is not None:
            training_batch.select_window_out_flag = select_window_out_flag[0] if isinstance(select_window_out_flag, (list, torch.Tensor)) else select_window_out_flag
        if i2v_mask is not None:
            training_batch.i2v_mask = i2v_mask.to(
                get_local_torch_device(), dtype=torch.bfloat16)    # i2v mask only works for memory training
        if viewmats is not None:
            training_batch.viewmats = viewmats.to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.Ks = Ks.to(
                get_local_torch_device(), dtype=torch.bfloat16)

        return training_batch

    def _normalize_dit_input(self,
                             training_batch: TrainingBatch) -> TrainingBatch:
        # TODO(will): support other models
        training_batch.latents = normalize_dit_input('wan',
                                                     training_batch.latents,
                                                     self.get_module("vae"))
        return training_batch

    def timestep_transform(self, t, shift=1.0, num_timesteps=1000.0):
        t = t / num_timesteps
        t = shift * t / (1 + (shift - 1) * t)
        t = t * num_timesteps
        return t

    def _prepare_ar_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        latents = training_batch.latents

        training_batch.clean_x = training_batch.latents.clone()
    

        return training_batch

    def _build_attention_metadata(
            self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch.attn_metadata = None
        return training_batch

    def _build_rope_idx(self,
                        training_batch: TrainingBatch) -> TrainingBatch:
        rank_in_sp_group = get_sp_parallel_rank()
        per_sp_seq_length = training_batch.latents.shape[2] * training_batch.per_seq_length

        training_batch.current_start = per_sp_seq_length * rank_in_sp_group
        training_batch.current_end = per_sp_seq_length * rank_in_sp_group + per_sp_seq_length
        return training_batch

    def _prepare_cond_latents(self, task_type, cond_latents, latents, multitask_mask):
        """
        Prepare conditional latents and mask for multitask training.

        Args:
            task_type: Type of task ("i2v" or "t2v").
            cond_latents: Conditional latents tensor.
            latents: Main latents tensor.
            multitask_mask: Multitask mask tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - latents_concat: Concatenated conditional latents.
                - mask_concat: Concatenated mask tensor.
        """
        latents_concat = None
        mask_concat = None

        if cond_latents is not None and task_type == 'i2v':
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            latents_concat = torch.zeros(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3],
                                         latents.shape[4]).to(latents.device)

        mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_concat = merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(
            device=latents.device)

        cond_latents = torch.concat([latents_concat, mask_concat], dim=1)

        return cond_latents

    def get_task_mask(self, task_type, latent_target_length):
        if task_type == "t2v":
            mask = torch.zeros(latent_target_length)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported !")
        return mask

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        extra_kwargs = {
            "byt5_text_states": training_batch.byt5_text_states,
            "byt5_text_mask": training_batch.byt5_text_mask,
        }

        multitask_mask = self.get_task_mask("i2v", training_batch.latents.shape[2]).to(self.device)
        cond_latents = self._prepare_cond_latents(
            "i2v", training_batch.image_cond, training_batch.latents, multitask_mask
        )

        latents_concat = torch.concat([training_batch.latents, cond_latents], dim=1)

        training_batch.input_kwargs = {
            "latents":
            training_batch.latents,
            "cond_latents": cond_latents,
            "timestep_txt": torch.tensor(0).unsqueeze(0).to(get_local_torch_device(),
                                        dtype=torch.bfloat16), # for ar model, we set txt timestep to 0
            "text_states":
                training_batch.prompt_embed,
            "text_states_2": None,
            "encoder_attention_mask": training_batch.prompt_mask,
            "timestep_r": None,
            "vision_states": training_batch.vision_states,
            "mask_type": "i2v",
            "guidance": None,
            "extra_kwargs": extra_kwargs,

            # Removed: camera control (viewmats, Ks, action)
            "return_dict": False,
            # Teacher forcing - clean_x is prepared in _ode_generate
            "clean_x": None,
            "aug_timesteps": None,
            "neg_text_states": training_batch.neg_prompt_embed,
            "neg_text_mask": training_batch.neg_prompt_mask,
            "neg_extra_kwargs": {
                "byt5_text_states": training_batch.neg_byt5_text_states,
                "byt5_text_mask": training_batch.neg_byt5_text_mask,
            },

            # PRoPE camera control
            "viewmats": getattr(training_batch, 'viewmats', None),
            "Ks": getattr(training_batch, 'Ks', None),
            
        }
        return training_batch


    def _ode_generate(self, training_batch: TrainingBatch) -> TrainingBatch:
        latents = training_batch.latents
        cond_input = training_batch.input_kwargs['cond_latents']
        viewmats = getattr(training_batch, 'viewmats', None)
        Ks = getattr(training_batch, 'Ks', None)
        x = torch.randn_like(latents).to(latents.device)
        B, _, T, H, W = latents.shape
        self.noise_scheduler.set_timesteps(num_inference_steps=self.training_args.ode_sampling_steps)

        # clean_x for teacher forcing - needs cond_input concatenated the same way as noisy input
        clean_x = training_batch.clean_x
        multitask_mask = self.get_task_mask("i2v", clean_x.shape[2]).to(clean_x.device)
        clean_cond_input = self._prepare_cond_latents(
            "i2v", training_batch.image_cond, clean_x, multitask_mask
        )
        clean_hidden = torch.cat([clean_x, clean_cond_input], dim=1)

        trajectory = []
        for i, t in enumerate(tqdm(
            self.noise_scheduler.timesteps,
            desc=f"  ODE solve (rank{self.global_rank})",
            leave=False,
            disable=self.local_rank > 0,
        )):
            # print("current timestep:", t.item())

            timesteps_in = t.unsqueeze(0).expand(B * T).to(latents.device, dtype=torch.bfloat16)
            # aug_timesteps: 0 for clean (teacher forcing)
            timestep_txt = torch.tensor(0).unsqueeze(0).to(latents.device, dtype=torch.bfloat16)
            aug_timesteps = torch.zeros_like(timesteps_in)
            trajectory.append(x)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cond_pred = self.transformer(
                    hidden_states=torch.cat([x, cond_input], dim=1),
                    timestep=timesteps_in,
                    timestep_txt=timestep_txt,
                    text_states=training_batch.text_states,
                    text_states_2=None,
                    encoder_attention_mask=training_batch.prompt_mask,
                    timestep_r=None,
                    vision_states=training_batch.vision_states,
                    mask_type="i2v",
                    guidance=None,
                    extra_kwargs=training_batch.input_kwargs["extra_kwargs"],
                    return_dict=False,
                    clean_x=clean_hidden,
                    aug_timesteps=aug_timesteps,
                    viewmats=viewmats,
                    Ks=Ks,
                )[0]
     
                if self.sample_cfg > 1:
                    uncond_pred = self.transformer(
                            hidden_states=torch.cat([x, cond_input], dim=1),
                            timestep=timesteps_in,
                            timestep_txt=timestep_txt,
                            text_states=training_batch.neg_text_states,
                            text_states_2=None,
                            encoder_attention_mask=training_batch.neg_text_mask,
                            timestep_r=None,
                            vision_states=training_batch.vision_states,
                            mask_type="i2v",
                            guidance=None,
                            extra_kwargs=training_batch.input_kwargs["neg_extra_kwargs"],
                            return_dict=False,
                            clean_x=clean_hidden,
                            aug_timesteps=aug_timesteps,
                            viewmats=viewmats,
                            Ks=Ks,
                        )[0]
                    pred = uncond_pred + self.sample_cfg * (cond_pred - uncond_pred)
                else:
                    pred = cond_pred

            x = self.noise_scheduler.step(pred, t, x).prev_sample

        trajectory.append(x)
        trajectory.append(clean_x)

        trajectory = torch.stack(trajectory, dim=1)

        noisy_inputs = trajectory[:,[0,12,24,36,-2,-1]]

        # Build latents_dict matching test_flex_tf.py format
        latents_dict = {
            'latent': latents,                     # [B, 32, F_latent, H_latent, W_latent]
            'prompt_embeds': training_batch.prompt_embed,        # [B, 300, 3584]
            'prompt_mask': training_batch.prompt_mask,          # [B, 300]
            'image_cond': training_batch.image_cond,       # [B, 32, 1, H_latent, W_latent]
            'vision_states': training_batch.vision_states,    # [B, 729, 1152]
            'byt5_text_states': training_batch.byt5_text_states,      # [B, 256, 1472]
            'byt5_text_mask': training_batch.byt5_text_mask,      # [B, 256]
            'ode_trajectory': noisy_inputs,          # [num_steps+2, B, 32, T, H, W]
            'viewmats': viewmats,                    # [B, T, 4, 4]
            'Ks': Ks,                                # [B, T, 3, 3]
        }

        # Determine output directory
        if self.training_args.ode_output_path:
            # Use custom output path
            output_dir = self.training_args.ode_output_path
        else:
            # Use original path-based naming
            data_path = training_batch.video_path.split("/")
            output_dir = os.path.join(os.path.dirname(data_path[-2]), "ode_latents")

        os.makedirs(output_dir, exist_ok=True)
        # Use last 3 path components to avoid basename collisions across subdirectories
        data_name = "_".join(training_batch.video_path.split("/")[-3:])

        torch.save(
            latents_dict,
            os.path.join(output_dir, data_name)
        )
        dist.barrier()

    


    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        training_batch = self._prepare_training(training_batch)

        training_batch = self._get_next_batch(training_batch)
        if getattr(training_batch, "end_of_pass", False):
            return training_batch

        training_batch = self._prepare_ar_dit_inputs(training_batch)

        training_batch = self._build_input_kwargs(training_batch)

        training_batch = self._ode_generate(
            training_batch)
        
        return training_batch


    def train(self) -> None:
        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed + self.global_rank)
        logger.info('rank: %s: start training',
                    self.global_rank,
                    local_main_process_only=False)
        if not self.post_init_called:
            self.post_init()

        # Set random seeds for deterministic training
        self.noise_random_generator = torch.Generator(device="cpu").manual_seed(
            self.seed)
        self.noise_gen_cuda = torch.Generator(device="cuda").manual_seed(
            self.seed)
        self.validation_random_generator = torch.Generator(
            device="cpu").manual_seed(self.seed)
        logger.info("Initialized random seeds with seed: %s", self.seed)

        # self.noise_scheduler = ODEFlowMatchEulerDiscreteScheduler(shift=self.training_args.ode_shift)
        self.noise_scheduler = FlowMatchDiscreteScheduler(
            shift=self.training_args.ode_shift,
            reverse=True,
            solver="euler",
        )
        self.noise_scheduler.set_timesteps(self.training_args.ode_sampling_steps)

        # Set flex_tf mode for teacher forcing
        self.transformer.set_attn_mode("flex_tf")
        logger.info("Transformer set to flex_tf mode for teacher forcing")
        self.train_loader_iter = iter(self.train_dataloader)

        # ODE sampling: walk the dataloader exactly once. Total = number of
        # batches per SP group (each rank produces its own shard of latents).
        try:
            total_batches = len(self.train_dataloader)
        except TypeError:
            total_batches = None

        progress_bar = tqdm(
            total=total_batches,
            desc="ODE samples",
            disable=self.local_rank > 0,
        )

        step = 0
        while True:
            step += 1
            training_batch = TrainingBatch()
            training_batch.current_timestep = step
            training_batch.current_vsa_sparsity = 0.0
            training_batch = self.train_one_step(training_batch)
            if getattr(training_batch, "end_of_pass", False):
                break
            progress_bar.update(1)

        progress_bar.close()
        logger.info("ODE sampling done: %d batches processed on rank %s",
                    step - 1, self.global_rank)
        if get_sp_group():
            cleanup_dist_env_and_memory()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args) -> None:
    logger.info("Starting sampling pipeline...")

    pipeline = CausalODESamplingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    pipeline.train()
    logger.info("Sampling pipeline done")


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