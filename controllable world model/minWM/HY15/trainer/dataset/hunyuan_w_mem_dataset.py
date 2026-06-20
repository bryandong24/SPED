import json
import os
import sys

sys.path.append(os.path.abspath('.'))
import torch
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Dict
import torch.multiprocessing as mp

from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from trainer.distributed import (
        get_local_torch_device,
        )
from trainer.distributed import (get_sp_world_size,
                                  get_world_rank,
                                  get_world_size)

from trainer.logger import init_logger

logger = init_logger(__name__)


class DP_SP_BatchSampler(Sampler[list[int]]):
    """
    A simple sequential batch sampler that yields batches of indices.
    """

    def __init__(
            self,
            batch_size: int,
            dataset_size: int,
            num_sp_groups: int,
            sp_world_size: int,
            global_rank: int,
            drop_last: bool = True,
            drop_first_row: bool = False,
            seed: int = 0,
    ):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.drop_last = drop_last
        self.seed = seed
        self.num_sp_groups = num_sp_groups
        self.global_rank = global_rank
        self.sp_world_size = sp_world_size

        # ── epoch-level RNG ────────────────────────────────────────────────
        rng = torch.Generator().manual_seed(self.seed)
        # Create a random permutation of all indices
        global_indices = torch.randperm(self.dataset_size, generator=rng)

        if drop_first_row:
            # drop 0 in global_indices
            global_indices = global_indices[global_indices != 0]
            self.dataset_size = self.dataset_size - 1

        if self.drop_last:
            # For drop_last=True, we:
            # 1. Ensure total samples is divisible by (batch_size * num_sp_groups)
            # 2. This guarantees each SP group gets same number of complete batches
            # 3. Prevents uneven batch sizes across SP groups at end of epoch
            num_batches = self.dataset_size // self.batch_size
            num_global_batches = num_batches // self.num_sp_groups
            global_indices = global_indices[:num_global_batches *
                                             self.num_sp_groups *
                                             self.batch_size]
        else:
            if self.dataset_size % (self.num_sp_groups * self.batch_size) != 0:
                # add more indices to make it divisible by (batch_size * num_sp_groups)
                padding_size = self.num_sp_groups * self.batch_size - (
                        self.dataset_size % (self.num_sp_groups * self.batch_size))
                logger.info("Padding the dataset from %d to %d",
                            self.dataset_size, self.dataset_size + padding_size)
                global_indices = torch.cat(
                    [global_indices, global_indices[:padding_size]])

        # shard the indices to each sp group
        ith_sp_group = self.global_rank // self.sp_world_size
        sp_group_local_indices = global_indices[ith_sp_group::self.
        num_sp_groups]
        self.sp_group_local_indices = sp_group_local_indices
        logger.info("Dataset size for each sp group: %d",
                    len(sp_group_local_indices))

    def __iter__(self):
        indices = self.sp_group_local_indices
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield batch_indices.tolist()

    def __len__(self):
        return len(self.sp_group_local_indices) // self.batch_size


class JsonWMemDataset(Dataset):
    def __init__(self, json_path, causal, window_frames, batch_size, training_cfg_rate, i2v_rate, task_type, drop_last, drop_first_row,
                 seed, device, shared_state):
        self.json_data = json.load(open(json_path, 'r'))
        self.all_length = len(self.json_data)
        self.causal = causal
        self.window_frames = window_frames
        self.training_cfg_rate = training_cfg_rate
        self.rng = random.Random(seed)
        self.i2v_rate = i2v_rate
        self.task_type = task_type
        self.device = device
        self.shared_state = shared_state

        self.sampler = DP_SP_BatchSampler(
            batch_size=batch_size,
            dataset_size=self.all_length,
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(),
            drop_last=drop_last,
            drop_first_row=drop_first_row,
            seed=seed,
        )

        # Get the directory containing the json_path to find negative prompt files
        neg_prompt_path = os.environ.get("NEG_PROMPT_PT", os.path.join(os.path.dirname(json_path), "hunyuan_neg_prompt.pt"))
        neg_byt5_path = os.environ.get("NEG_BYT5_PT", os.path.join(os.path.dirname(json_path), "hunyuan_neg_byt5_prompt.pt"))

        self.neg_prompt_pt = torch.load(
            neg_prompt_path,
            map_location="cpu",
            weights_only=True,
        )

        self.neg_byt5_pt = torch.load(
            neg_byt5_path,
            map_location="cpu",
            weights_only=True,
        )

    def __len__(self):
        return self.all_length

    def update_max_frames(self, training_step):
        # [TODO] update_max_frames: current strategy needs review
        if training_step < 500:
            self.shared_state["max_frames"] = 32
        elif training_step < 1000:
            self.shared_state["max_frames"] = 64
        elif training_step < 2000:
            self.shared_state["max_frames"] = 96
        elif training_step < 3000:
            self.shared_state["max_frames"] = 128
        else:
            self.shared_state["max_frames"] = 160

    def __getitem__(self, idx):
        while True:
            try:
                json_data = self.json_data[idx]
                latent_pt_path = json_data['latent_path']

                latent_pt = torch.load(
                    os.path.join(latent_pt_path),
                    map_location="cpu",
                    weights_only=True,
                )
                latent = latent_pt['latent'][0]
                latent_length = latent.shape[1]

                # Check if latent is long enough
                if latent_length < self.window_frames:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue
                
                # Apply max_frames limit
                max_frames = int(self.shared_state["max_frames"]) // 4 * 4
                max_length = min(max_frames, latent_length // 4 * 4)

                max_length = min(max_length, self.window_frames)
                latent = latent[:, :max_length, ...]

                # Load text embeddings
                prompt_embed = latent_pt['prompt_embeds'][0]
                prompt_mask = latent_pt['prompt_mask'][0]
                byt5_text_states = latent_pt['byt5_text_states'][0]
                byt5_text_mask = latent_pt['byt5_text_mask'][0]

                # Apply CFG (classifier-free guidance)
                if self.rng.random() < self.training_cfg_rate:
                    prompt_embed = self.neg_prompt_pt['negative_prompt_embeds'][0]
                    prompt_mask = self.neg_prompt_pt['negative_prompt_mask'][0]
                    byt5_text_states = self.neg_byt5_pt['byt5_text_states'][0]
                    byt5_text_mask = self.neg_byt5_pt['byt5_text_mask'][0]

                # Load image conditioning based on task type
                if self.task_type == "i2v":
                    image_cond = latent_pt['image_cond'][0]
                    vision_states = latent_pt['vision_states'][0]
                else:
                    # For t2v, use zero tensors
                    image_cond = torch.zeros(32, 1, latent.shape[-2], latent.shape[-1])
                    vision_states = torch.zeros(1, 4096, 3584)

                # Create mask based on task type
                i2v_mask = torch.ones_like(latent)

                batch = {
                    "i2v_mask": i2v_mask,
                    "latent": latent,
                    "prompt_embed": prompt_embed,
                    "prompt_mask": prompt_mask,
                    "byt5_text_states": byt5_text_states,
                    "byt5_text_mask": byt5_text_mask,
                    "image_cond": image_cond,
                    "vision_states": vision_states,
                    "video_path": latent_pt_path,  # for logging
                    "select_window_out_flag": 0,  # no memory training
                }
                break
            except Exception as e:
                print('error:', e, latent_pt_path, flush=True)
                idx = self.rng.randint(0, self.all_length - 1)
        return batch


def cycle(dl):
    while True:
        for data in dl:
            yield data


def latent_collate_function(batch):
    latent = torch.stack([b["latent"] for b in batch], dim=0)
    prompt_embed = torch.stack([b["prompt_embed"] for b in batch], dim=0)
    i2v_mask = torch.stack([b["i2v_mask"] for b in batch], dim=0)

    image_cond = torch.stack([b["image_cond"] for b in batch], dim=0)
    vision_states = torch.stack([b["vision_states"] for b in batch], dim=0)
    prompt_mask = torch.stack([b["prompt_mask"] for b in batch], dim=0)
    byt5_text_states = torch.stack([b["byt5_text_states"] for b in batch], dim=0)
    byt5_text_mask = torch.stack([b["byt5_text_mask"] for b in batch], dim=0)

    video_path = [b["video_path"] for b in batch]
    select_window_out_flag = [b["select_window_out_flag"] for b in batch]

    return {
        "i2v_mask": i2v_mask,
        "latent": latent,
        "prompt_embed": prompt_embed,
        "prompt_mask": prompt_mask,
        "byt5_text_states": byt5_text_states,
        "byt5_text_mask": byt5_text_mask,
        "image_cond": image_cond,
        "vision_states": vision_states,
        "video_path": video_path,
        "select_window_out_flag": select_window_out_flag,
    }


def build_hunyuan_w_mem_dataloader(
        json_path,
        causal,
        window_frames,
        batch_size,
        num_data_workers,
        drop_last,
        drop_first_row,
        seed,
        training_cfg_rate,
        i2v_rate,
        task_type, ) -> tuple[JsonWMemDataset, StatefulDataLoader]:
    manager = mp.Manager()
    shared_state = manager.dict()
    shared_state["max_frames"] = window_frames

    dataset = JsonWMemDataset(json_path, causal, window_frames, batch_size, training_cfg_rate, i2v_rate, task_type,
                              drop_last=drop_last, drop_first_row=drop_first_row, seed=seed,
                              device=get_local_torch_device(), shared_state=shared_state)

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=latent_collate_function,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
