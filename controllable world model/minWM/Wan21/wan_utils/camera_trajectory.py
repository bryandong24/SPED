"""Camera trajectory generators for inference evaluation.

All poses are w2c (world-to-camera) OpenCV convention, matching the training
pipeline (build_worldplaygen_lmdb.py: c2w -> w2c via np.linalg.inv).

Trajectory string format (each step = 0.08 unit translation or 3° rotation):
  w*N  -- move forward  (+Z in camera local frame)
  s*N  -- move backward (-Z)
  a*N  -- move left     (-X)
  d*N  -- move right    (+X)
  u*N  -- move up       (-Y, OpenCV Y-down)
  dn*N -- move down     (+Y)
  j*N  -- yaw left      (rotate around Y, positive)
  l*N  -- yaw right     (rotate around Y, negative)
  i*N  -- pitch up      (rotate around X, negative)
  k*N  -- pitch down    (rotate around X, positive)

Example: "w*19" -> 20-frame forward dolly (1 identity + 19 steps).
Chain segments: "w*10,d*9", "w*9,j*10"
"""

import re
import numpy as np
import torch

_STEP = 0.08
_ROT_STEP = np.radians(3.0)  # 3.0 degrees per latent frame


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# Direction -> per-step motion dict (same convention as HY-WorldPlay)
_MOTIONS = {
    "w":  {"forward":  _STEP},
    "s":  {"forward": -_STEP},
    "d":  {"right":    _STEP},
    "a":  {"right":   -_STEP},
    "u":  {"up":       _STEP},
    "dn": {"up":      -_STEP},
    "j":  {"yaw":     -_ROT_STEP},   # yaw left
    "l":  {"yaw":      _ROT_STEP},   # yaw right
    "i":  {"pitch":    _ROT_STEP},   # pitch up
    "k":  {"pitch":   -_ROT_STEP},   # pitch down
}


def _generate_c2w_trajectory(motions):
    """Build c2w 4x4 matrices from motion dicts.

    Exact equivalent of HY-WorldPlay/hyvideo/generate_custom_trajectory.py
    to ensure poses match the training pipeline.
    """
    T = np.eye(4)
    poses = [T.copy()]
    for move in motions:
        if "yaw" in move:
            T[:3, :3] = T[:3, :3] @ _rot_y(move["yaw"])
        if "pitch" in move:
            T[:3, :3] = T[:3, :3] @ _rot_x(move["pitch"])
        forward = move.get("forward", 0.0)
        if forward:
            T[:3, 3] += T[:3, :3] @ np.array([0, 0, forward])
        right = move.get("right", 0.0)
        if right:
            T[:3, 3] += T[:3, :3] @ np.array([right, 0, 0])
        up = move.get("up", 0.0)
        if up:
            # up in camera frame = -Y (OpenCV Y-down)
            T[:3, 3] += T[:3, :3] @ np.array([0, -up, 0])
        poses.append(T.copy())
    return poses


def parse_trajectory(traj_str: str) -> np.ndarray:
    """Parse trajectory string into (T, 4, 4) w2c viewmats.

    Builds c2w via _generate_c2w_trajectory (matching training pipeline),
    then inverts to w2c via np.linalg.inv. First frame is always identity.
    """
    segments = traj_str.strip().split(",")
    motions = []
    for seg in segments:
        seg = seg.strip()
        m = re.fullmatch(r"([a-z]+)\*(\d+)", seg)
        if m is None:
            raise ValueError(f"Cannot parse trajectory segment: '{seg}'. Expected 'w*19'.")
        key, n = m.group(1), int(m.group(2))
        if key not in _MOTIONS:
            raise ValueError(f"Unknown direction '{key}'. Valid: {list(_MOTIONS.keys())}")
        motions.extend([_MOTIONS[key]] * n)

    c2w_list = _generate_c2w_trajectory(motions)
    T = len(c2w_list)
    viewmats = np.zeros((T, 4, 4), dtype=np.float32)
    for i, c2w in enumerate(c2w_list):
        viewmats[i] = np.linalg.inv(c2w)
    return viewmats


def make_camera_tensors(traj_str: str,
                        fx: float = 0.5050505, fy: float = 0.89786756,
                        cx: float = 0.5, cy: float = 0.5,
                        device="cpu", dtype=torch.float32):
    """Build (1, T, 4, 4) viewmats and (1, T, 3, 3) Ks tensors.

    Args:
        traj_str: trajectory string, e.g. "w*19"
        fx, fy, cx, cy: normalized intrinsics
            (defaults match WorldPlayGen: 1920x1080, f~969.7)

    Returns:
        viewmats: (1, T, 4, 4) w2c matrices
        Ks:       (1, T, 3, 3)
    """
    viewmats_np = parse_trajectory(traj_str)
    T = len(viewmats_np)

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    Ks_np = np.tile(K, (T, 1, 1))

    viewmats = torch.tensor(viewmats_np, dtype=dtype, device=device).unsqueeze(0)
    Ks = torch.tensor(Ks_np, dtype=dtype, device=device).unsqueeze(0)
    return viewmats, Ks
