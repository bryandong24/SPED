from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }





class LatentLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1]
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }



class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }



class CameraODERegressionLMDBDataset(ODERegressionLMDBDataset):
    """ODERegressionLMDBDataset extended with per-frame camera data for PRoPE.

    LMDB keys:
      - ``latents``:  float16 (N, S, F, C, H, W) — ODE trajectory
      - ``prompts``:  string
      - ``viewmats``: float32 (N, F, 4, 4) — w2c view matrices
      - ``Ks``:       float32 (N, F, 3, 3) — intrinsic matrices
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        super().__init__(data_path, max_pair)
        self.viewmats_shape = get_array_shape_from_lmdb(self.env, 'viewmats')
        self.Ks_shape = get_array_shape_from_lmdb(self.env, 'Ks')

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: String
            - ode_latent: (S, F, C, H, W) float32 — ODE trajectory from noise to clean
            - viewmats: (F, 4, 4) float32 — w2c view matrices
            - Ks: (F, 3, 3) float32 — intrinsic matrices
        """
        latents = retrieve_row_from_lmdb(
            self.env, "latents", np.float16, idx, shape=self.latents_shape[1:]
        )
        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(self.env, "prompts", str, idx)

        viewmats = retrieve_row_from_lmdb(
            self.env, "viewmats", np.float32, idx, shape=self.viewmats_shape[1:]
        )
        Ks = retrieve_row_from_lmdb(
            self.env, "Ks", np.float32, idx, shape=self.Ks_shape[1:]
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
            "viewmats": torch.tensor(viewmats, dtype=torch.float32),
            "Ks": torch.tensor(Ks, dtype=torch.float32),
        }


class CameraLatentLMDBDataset(LatentLMDBDataset):
    """LatentLMDBDataset extended with per-frame camera data for PRoPE.

    Expects the LMDB to contain raw camera parameters:
      - ``intrinsics``: float32 array of shape ``(N, 4)`` — [fx, fy, cx, cy] normalized
      - ``poses``:      float32 array of shape ``(N, F, 7)`` — [tx,ty,tz, qx,qy,qz,qw] w2c

    viewmats (F, 4, 4) and Ks (F, 3, 3) are built on-the-fly via
    ``build_viewmats_and_Ks()``, which normalizes poses to the first frame.

    ``data_path`` can be either:
      - a single LMDB directory (has ``data.mdb`` inside), or
      - a parent directory containing multiple LMDB subdirectories (sharding).
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        # Detect sharding: if data_path contains data.mdb, it's a single LMDB;
        # otherwise treat each subdirectory as a shard.
        if os.path.isfile(os.path.join(data_path, "data.mdb")):
            self._sharded = False
            super().__init__(data_path, max_pair)
            self.intrinsics_shape = get_array_shape_from_lmdb(
                self.env, 'intrinsics')
            self.poses_shape = get_array_shape_from_lmdb(self.env, 'poses')
        else:
            self._sharded = True
            self.envs = []
            self.index = []  # list of (shard_id, local_idx)
            self._latents_shapes = []
            self._intrinsics_shapes = []
            self._poses_shapes = []
            for fname in sorted(os.listdir(data_path)):
                sub = os.path.join(data_path, fname)
                if not os.path.isdir(sub):
                    continue
                if not os.path.isfile(os.path.join(sub, "data.mdb")):
                    continue
                env = lmdb.open(sub, readonly=True, lock=False,
                                readahead=False, meminit=False)
                sid = len(self.envs)
                self.envs.append(env)
                ls = get_array_shape_from_lmdb(env, 'latents')
                self._latents_shapes.append(ls)
                self._intrinsics_shapes.append(
                    get_array_shape_from_lmdb(env, 'intrinsics'))
                self._poses_shapes.append(
                    get_array_shape_from_lmdb(env, 'poses'))
                for j in range(ls[0]):
                    self.index.append((sid, j))
            self.max_pair = max_pair

    def __len__(self):
        if self._sharded:
            return min(len(self.index), self.max_pair)
        return super().__len__()

    def __getitem__(self, idx):
        if self._sharded:
            sid, local_idx = self.index[idx]
            env = self.envs[sid]
            ls = self._latents_shapes[sid]
            latents = retrieve_row_from_lmdb(
                env, "latents", np.float16, local_idx, shape=ls[1:])
            if len(latents.shape) == 4:
                latents = latents[None, ...]
            prompts = retrieve_row_from_lmdb(env, "prompts", str, local_idx)
            intrinsics = retrieve_row_from_lmdb(
                env, "intrinsics", np.float32, local_idx,
                shape=self._intrinsics_shapes[sid][1:])
            poses = retrieve_row_from_lmdb(
                env, "poses", np.float32, local_idx,
                shape=self._poses_shapes[sid][1:])
        else:
            # Single LMDB path — original behavior
            latents = retrieve_row_from_lmdb(
                self.env, "latents", np.float16, idx,
                shape=self.latents_shape[1:])
            if len(latents.shape) == 4:
                latents = latents[None, ...]
            prompts = retrieve_row_from_lmdb(self.env, "prompts", str, idx)
            intrinsics = retrieve_row_from_lmdb(
                self.env, "intrinsics", np.float32, idx,
                shape=self.intrinsics_shape[1:])
            poses = retrieve_row_from_lmdb(
                self.env, "poses", np.float32, idx,
                shape=self.poses_shape[1:])

        viewmats, Ks = build_viewmats_and_Ks(intrinsics, poses)
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1],
            "viewmats": torch.tensor(viewmats, dtype=torch.float32),
            "Ks": torch.tensor(Ks, dtype=torch.float32),
        }


def build_viewmats_and_Ks(intrinsics, poses):
    """Build 4x4 w2c view matrices and 3x3 intrinsics from raw poses.

    Called at dataset load time (in ``CameraLatentLMDBDataset.__getitem__``).

    Args:
        intrinsics: (4,) ndarray [fx, fy, cx, cy] (normalized)
        poses:      (T, 7) ndarray [tx, ty, tz, qx, qy, qz, qw] w2c OpenCV

    Returns:
        viewmats: (T, 4, 4) float32 — w2c SE3, normalized to first frame
        Ks:       (T, 3, 3) float32 — intrinsics
    """
    T = len(poses)
    fx, fy, cx, cy = intrinsics

    viewmats = np.zeros((T, 4, 4), dtype=np.float32)
    for i in range(T):
        tx, ty, tz, qx, qy, qz, qw = poses[i]
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        viewmats[i, :3, :3] = R
        viewmats[i, :3, 3] = [tx, ty, tz]
        viewmats[i, 3, 3] = 1.0

    # Normalize: align all poses to first frame
    c2w = np.linalg.inv(viewmats)
    C0_inv = np.linalg.inv(c2w[0])
    c2w_aligned = np.array([C0_inv @ C for C in c2w])
    viewmats = np.linalg.inv(c2w_aligned).astype(np.float32)

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    Ks = np.tile(K, (T, 1, 1))

    return viewmats, Ks


def cycle(dl):
    while True:
        for data in dl:
            yield data
