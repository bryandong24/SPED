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

from trainer.dataset_camera.ar_camera_plucker_dataset import build_viewmats_and_Ks, normalize_translations
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


class InferenceDataset(Dataset):
    def __init__(self, json_path, causal, window_frames, batch_size, cfg_rate, i2v_rate, drop_last, drop_first_row,
                 seed, device, shared_state):
        self.json_data = json.load(open(json_path, 'r'))
        self.all_length = len(self.json_data)
        self.causal = causal
        self.window_frames = window_frames
        self.cfg_rate = cfg_rate
        self.rng = random.Random(seed)
        self.i2v_rate = i2v_rate
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

        neg_prompt_path = os.environ.get("NEG_PROMPT_PT", "/your_path/to/hunyuan_neg_prompt.pt")
        neg_byt5_path = os.environ.get("NEG_BYT5_PT", "/your_path/to/hunyuan_neg_byt5_prompt.pt")
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
                latent = latent[:, :max_length, ...]

                # Load text embeddings
                prompt_embed = latent_pt['prompt_embeds'][0]
                prompt_mask = latent_pt['prompt_mask'][0]
                byt5_text_states = latent_pt['byt5_text_states'][0]
                byt5_text_mask = latent_pt['byt5_text_mask'][0]

                # Apply CFG (classifier-free guidance)
                # if self.rng.random() < self.cfg_rate:
                neg_prompt_embed = self.neg_prompt_pt['negative_prompt_embeds'][0]
                neg_prompt_mask = self.neg_prompt_pt['negative_prompt_mask'][0]
                neg_byt5_text_states = self.neg_byt5_pt['byt5_text_states'][0]
                neg_byt5_text_mask = self.neg_byt5_pt['byt5_text_mask'][0]

                # Load image conditioning
                image_cond = latent_pt['image_cond'][0]
                vision_states = latent_pt['vision_states'][0]

                # Simple random window selection
                if latent.shape[1] > self.window_frames:
                    max_start = latent.shape[1] - self.window_frames
                    start_idx = self.rng.randint(0, max_start)
                    start_idx = 0 # hardcode to avoid condition mismatch
                    latent = latent[:, start_idx:start_idx + self.window_frames, ...]
                else:
                    latent = latent[:, :self.window_frames, ...]

                T_lat = latent.shape[1]

                # ── Camera matrices for PRoPE (aligned with ar_camera_plucker_dataset) ──
                intrinsics = latent_pt['intrinsics'].numpy()        # (4,)
                if intrinsics[0] <= 0 or intrinsics[1] <= 0:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue
                poses_raw = latent_pt['poses'].numpy()              # (N_cam, 7)
                poses_lat = poses_raw[:T_lat]
                viewmats, Ks = build_viewmats_and_Ks(intrinsics, poses_lat)
                viewmats, discard = normalize_translations(viewmats)
                if discard:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue
                viewmats = torch.from_numpy(viewmats).float()  # (T_lat, 4, 4)
                Ks = torch.from_numpy(Ks).float()              # (T_lat, 3, 3)

                # Create i2v mask
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
                    "neg_prompt_embed": neg_prompt_embed,
                    "neg_prompt_mask": neg_prompt_mask,
                    "neg_byt5_text_states": neg_byt5_text_states,
                    "neg_byt5_text_mask": neg_byt5_text_mask,
                    "viewmats": viewmats,
                    "Ks": Ks,
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
    neg_prompt_embed = torch.stack([b["neg_prompt_embed"] for b in batch], dim=0)
    neg_prompt_mask = torch.stack([b["neg_prompt_mask"] for b in batch], dim=0)
    neg_byt5_text_states = torch.stack([b["neg_byt5_text_states"] for b in batch], dim=0)
    neg_byt5_text_mask = torch.stack([b["neg_byt5_text_mask"] for b in batch], dim=0)
    viewmats = torch.stack([b["viewmats"] for b in batch], dim=0)
    Ks = torch.stack([b["Ks"] for b in batch], dim=0)

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
        "neg_prompt_embed": neg_prompt_embed,
        "neg_prompt_mask": neg_prompt_mask,
        "neg_byt5_text_states": neg_byt5_text_states,
        "neg_byt5_text_mask": neg_byt5_text_mask,
        "viewmats": viewmats,
        "Ks": Ks,
    }


def build_inference_dataloader(
        json_path,
        causal,
        window_frames,
        batch_size,
        num_data_workers,
        drop_last,
        drop_first_row,
        seed,
        cfg_rate,
        i2v_rate, ) -> tuple[InferenceDataset, StatefulDataLoader]:
    manager = mp.Manager()
    shared_state = manager.dict()
    shared_state["max_frames"] = window_frames

    dataset = InferenceDataset(json_path, causal, window_frames, batch_size, cfg_rate, i2v_rate,
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