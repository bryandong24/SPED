from wan_utils.wan_wrapper import WanDiffusionWrapper
from wan_utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache = None
        self.prope_kv_cache = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        # All ranks must execute torch.randint to keep CUDA RNG states in sync
        # across SP groups (SP ranks share the same seed). Only rank 0's values
        # are used via broadcast, but every rank must consume the same number of
        # RNG draws to avoid divergence.
        indices = torch.randint(
            low=0,
            high=num_denoising_steps,
            size=(num_blocks,),
            device=device
        )
        if self.last_step_only:
            indices = torch.ones_like(indices) * (num_denoising_steps - 1)

        dist.broadcast(indices, src=0)  # Broadcast rank 0's indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            clean_image_or_video: torch.Tensor = None, # same shape as noise
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        viewmats = conditional_dict.pop("viewmats", None)  # (B, total_F, 4, 4)
        Ks = conditional_dict.pop("Ks", None)              # (B, total_F, 3, 3)
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_prope_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )


        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None: # Never met
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        # In out training, self.independent_first_frame is False
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            vm_chunk = viewmats[:, current_start_frame:current_start_frame + current_num_frames] if viewmats is not None else None
            ks_chunk = Ks[:, current_start_frame:current_start_frame + current_num_frames] if Ks is not None else None

            if True:
                noisy_input = noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

                # Step 3.1: Spatial denoising loop
                # Such a loop corresponds to the truncated denoising algorithm:
                #    T -> \tau_1 -> \tau_2 ->...-> \tau —— enable grad ——> 0
                # For many-step model, we certainly cannot use this method, but for 4-step DMD,
                # we can inherit it for a fair comaprison. Note that as long as the conditions
                # are clean GT rather than self-generated frames, we can perform TF. So this
                # method does not conflict with TF in the frame- dimension.
                for index, current_timestep in enumerate(self.denoising_step_list):
                    # self.same_step_across_blocks is True
                    if self.same_step_across_blocks:
                        exit_flag = (index == exit_flags[0])
                    else:
                        exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if not exit_flag:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                viewmats=vm_chunk,
                                Ks=ks_chunk,
                                prope_kv_cache=self.prope_kv_cache,
                            )
                            next_timestep = self.denoising_step_list[index + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        # for getting real output
                        # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                        if current_start_frame < start_gradient_frame_index: # Always True as long as we train 21 latent frames
                            with torch.no_grad():
                                _, denoised_pred = self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=conditional_dict,
                                    timestep=timestep,
                                    kv_cache=self.kv_cache,
                                    crossattn_cache=self.crossattn_cache,
                                    current_start=current_start_frame * self.frame_seq_length,
                                    viewmats=vm_chunk,
                                    Ks=ks_chunk,
                                    prope_kv_cache=self.prope_kv_cache,
                                )
                        else: # enable grad
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                viewmats=vm_chunk,
                                Ks=ks_chunk,
                                prope_kv_cache=self.prope_kv_cache,
                            )
                        break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache (no context_noise)
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred.detach(),
                    conditional_dict=conditional_dict,
                    timestep=torch.zeros_like(timestep),
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache,
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks: # Useless, never met
            denoised_timestep_from, denoised_timestep_to = None, None
        # T -> \tau_1 -> \tau_2 ->...-> \tau —— enable grad ——> 0
        # denoised_timestep_from = \tau
        # denoised_timestep_to = next timestep smaller than \tau
        # These are just engineering tricks
        # to align DMD timestep sampling with the actual denoising range used by the generator
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            # corner case when \tau is the smallest non-zero timestep
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step: # False
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        Under SP, KV cache is stored in head-parallel domain (post all-to-all),
        so each rank only stores num_heads // sp_size heads.
        """
        num_heads = self._get_sp_num_heads(12)
        kv_cache = []

        for _ in range(self.num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, num_heads, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache = kv_cache  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.

        NOTE: Cross-attention does NOT use SP all-to-all (q comes from
        seq-parallel x, k/v from text context), so cache keeps full num_heads.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _initialize_prope_kv_cache(self, batch_size, dtype, device):
        num_heads = self._get_sp_num_heads(12)
        prope_kv_cache = []
        for _ in range(self.num_transformer_blocks):
            prope_kv_cache.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, num_heads, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        self.prope_kv_cache = prope_kv_cache

    @staticmethod
    def _get_sp_num_heads(full_num_heads):
        """Return per-rank num_heads under SP (head-parallel domain)."""
        try:
            from sp.parallel_states import get_parallel_state
            ps = get_parallel_state()
            if ps.sp_enabled:
                return full_num_heads // ps.sp
        except (ImportError, AttributeError):
            pass
        return full_num_heads
