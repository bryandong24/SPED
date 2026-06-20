#!/usr/bin/env python3
"""
CameraTrajectory: unified class for generating, converting, and visualizing
DL3DV-style camera trajectories, with direct pipeline integration.

Usage:
    from hyvideo.utils.camera_utils import CameraTrajectory

    cam = CameraTrajectory(num_frames=77, seed=42)

    # 1. Generate random action string (length = num_frames - 1)
    action_str = cam.generate()

    # 2. Action string -> w2c matrices (num_frames, 4, 4)
    w2c = cam.to_w2c(action_str)

    # 3. Subsample to latent-aligned w2c
    w2c_latent = cam.to_latent_w2c(w2c)

    # 4. Get pipeline-ready tensors: (viewmats, Ks, action_label)
    viewmats, Ks, action = cam.to_pipeline_input(action_str)

    # 5. Visualize
    cam.visualize(action_str, "traj.png")
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# ── Constants ────────────────────────────────────────────────────────────

TRANS_STEP = 0.02   # 0.08 per latent / 4 pixel-frames, aligned with parse_pose_string
ROT_STEP = 0.75    # 3.0° per latent / 4 pixel-frames, aligned with parse_pose_string

ACTIONS = ['W', 'S', 'A', 'D', 'J', 'L', 'I', 'K']
TRANS_ACTIONS = {'W', 'S', 'A', 'D'}
YAW_ACTIONS = {'J', 'L'}
PITCH_ACTIONS = {'I', 'K'}
ACTION_WEIGHT = {a: 1.2 if a in TRANS_ACTIONS else 1.0 if a in YAW_ACTIONS else 0.8 for a in ACTIONS}
FORBIDDEN = {'W': 'S', 'S': 'W'}

# Default intrinsics (DL3DV convention, 1920x1080)
DEFAULT_FX = DEFAULT_FY = 969.6969696969696
DEFAULT_CX = 960.0
DEFAULT_CY = 540.0

# Action label encoding — same mapping as generate.py
_ACTION_MAPPING = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1, (0, 1, 0, 0): 2,  # forward / backward
    (0, 0, 1, 0): 3, (0, 0, 0, 1): 4,  # right / left
    (1, 0, 1, 0): 5, (1, 0, 0, 1): 6,  # forward+right / forward+left
    (0, 1, 1, 0): 7, (0, 1, 0, 1): 8,  # backward+right / backward+left
}

ACTION_LABELS = {
    'W': 'Forward(W)', 'S': 'Back(S)', 'A': 'Left(A)', 'D': 'Right(D)',
    'J': 'YawL(J)', 'L': 'YawR(L)', 'I': 'PitchU(I)', 'K': 'PitchD(K)',
}
ACTION_COLORS = {
    'W': '#2ecc71', 'S': '#e74c3c', 'A': '#3498db', 'D': '#e67e22',
    'J': '#9b59b6', 'L': '#8e44ad', 'I': '#1abc9c', 'K': '#16a085',
}


def _rot_x(deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


_LOCAL_TRANS = {
    'W': np.array([0, 0, TRANS_STEP]),
    'S': np.array([0, 0, -TRANS_STEP]),
    'D': np.array([TRANS_STEP, 0, 0]),
    'A': np.array([-TRANS_STEP, 0, 0]),
}
_ROT_FUNC = {
    'L': lambda: _rot_y(ROT_STEP),
    'J': lambda: _rot_y(-ROT_STEP),
    'K': lambda: _rot_x(ROT_STEP),
    'I': lambda: _rot_x(-ROT_STEP),
}


def _one_hot_to_label(one_hot):
    """Convert (N, 4) one-hot tensor to (N,) label tensor via _ACTION_MAPPING."""
    return torch.tensor([_ACTION_MAPPING[tuple(row.tolist())] for row in one_hot])


# ── Class ────────────────────────────────────────────────────────────────

class CameraTrajectory:
    """
    Generate and convert camera trajectories.

    Coordinate convention:
      - Camera looks along -Z (OpenCV)
      - W/S = translate along Z, A/D = translate along X
      - J/L = yaw (rotate around Y), I/K = pitch (rotate around X)

    Temporal mapping (VAE 4x compression):
      - num_frames video frames -> num_frames - 1 action chars
      - (num_frames - 1) / 4 = num_latents latent frames
      - Latent frame i corresponds to video frame (i+1)*4
    """

    def __init__(self, num_frames=77, seed=None):
        self.num_frames = num_frames
        self.num_actions = num_frames - 1
        self.num_latents = self.num_actions // 4
        self.rng = np.random.default_rng(seed)

    # ── 1. Generate action string ────────────────────────────────────────

    def generate(self) -> str:
        """
        Generate a random action string.
        Returns str of length num_frames - 1, each char in {W,S,A,D,J,L,I,K}.
        """
        blocks = self._generate_blocks()
        chars = []
        for action, size in blocks:
            chars.extend([action] * size)
        return ''.join(chars)

    def _generate_blocks(self):
        MIN_BLOCK = 16
        blocks = []
        remaining = self.num_actions

        w = np.array([ACTION_WEIGHT[a] for a in ACTIONS])
        w /= w.sum()
        action = ACTIONS[self.rng.choice(len(ACTIONS), p=w)]

        while remaining > 0:
            size = self._block_size(action)
            if size <= remaining:
                blocks.append((action, size))
                remaining -= size
                if remaining >= MIN_BLOCK:
                    action = self._next_action(action)
                elif remaining > 0:
                    blocks[-1] = (blocks[-1][0], blocks[-1][1] + remaining)
                    remaining = 0
            else:
                if remaining >= MIN_BLOCK:
                    blocks.append((action, remaining))
                else:
                    if blocks:
                        blocks[-1] = (blocks[-1][0], blocks[-1][1] + remaining)
                    else:
                        blocks.append((action, remaining))
                remaining = 0
        return blocks

    def _next_action(self, prev):
        candidates, weights = [], []
        for a in ACTIONS:
            if a == prev:
                continue  # no action can repeat consecutively
            if FORBIDDEN.get(prev) == a:
                continue
            candidates.append(a)
            weights.append(ACTION_WEIGHT[a])
        weights = np.array(weights)
        weights /= weights.sum()
        return candidates[self.rng.choice(len(candidates), p=weights)]

    @staticmethod
    def _block_size(action):
        return 16 if action in ('W', 'S', 'I', 'K') else 32

    # ── 2. Action string -> w2c matrices ─────────────────────────────────

    @staticmethod
    def to_w2c(action_str: str) -> np.ndarray:
        """
        Convert action string to w2c matrices.
        Returns np.ndarray of shape (len(action_str)+1, 4, 4).
        Frame 0 is identity.
        """
        c2w = np.eye(4)
        c2ws = [c2w.copy()]
        for ch in action_str:
            if ch in _LOCAL_TRANS:
                c2w[:3, 3] += c2w[:3, :3] @ _LOCAL_TRANS[ch]
            elif ch in _ROT_FUNC:
                c2w[:3, :3] = c2w[:3, :3] @ _ROT_FUNC[ch]()
            else:
                raise ValueError(f"Unknown action: {ch}")
            c2ws.append(c2w.copy())
        c2ws = np.array(c2ws)
        w2cs = np.linalg.inv(c2ws)
        return w2cs

    # ── 3. Subsample to latent-aligned poses ────────────────────────────

    @staticmethod
    def to_latent_c2w(action_str: str) -> np.ndarray:
        """
        Get latent-aligned c2w matrices including frame 0 (identity).
        Returns (latent_num, 4, 4) where latent_num = (len(action_str))//4 + 1.

        Indices: [0, 4, 8, ..., (latent_num-1)*4]
        This matches the format expected by pose_to_input() in generate.py.
        """
        # Build c2w at frame level
        c2w = np.eye(4)
        c2ws = [c2w.copy()]
        for ch in action_str:
            if ch in _LOCAL_TRANS:
                c2w[:3, 3] += c2w[:3, :3] @ _LOCAL_TRANS[ch]
            elif ch in _ROT_FUNC:
                c2w[:3, :3] = c2w[:3, :3] @ _ROT_FUNC[ch]()
            else:
                raise ValueError(f"Unknown action: {ch}")
            c2ws.append(c2w.copy())
        c2ws = np.array(c2ws)

        num_frames = len(c2ws)
        num_latents = (num_frames - 1) // 4
        # frame 0 + every 4th frame
        indices = [0] + [(i + 1) * 4 for i in range(num_latents)]
        return c2ws[indices].copy()

    @staticmethod
    def to_latent_w2c(w2c: np.ndarray) -> np.ndarray:
        """
        Subsample w2c to latent temporal resolution (without frame 0).
        Takes the last frame of each 4-frame group:
          latent k -> frame (k+1)*4
        Returns (num_latents, 4, 4).
        """
        num_frames = w2c.shape[0]
        num_latents = (num_frames - 1) // 4
        indices = [(i + 1) * 4 for i in range(num_latents)]
        return w2c[indices].copy()

    # ── 4. Normalized intrinsics (3x3 K matrix) ─────────────────────────

    @staticmethod
    def get_normalized_intrinsics(fx=DEFAULT_FX, fy=DEFAULT_FY,
                                  cx=DEFAULT_CX, cy=DEFAULT_CY) -> np.ndarray:
        """
        Get normalized 3x3 intrinsics matrix, matching pose_to_input() convention.

        Normalization: K[0,0] = fx / (cx * 2), K[1,1] = fy / (cy * 2),
                       K[0,2] = 0.5, K[1,2] = 0.5

        Default: fx=fy=969.697, cx=960, cy=540 (DL3DV 1920x1080)
        -> [[0.5051, 0, 0.5], [0, 0.8979, 0.5], [0, 0, 1]]

        Returns:
            np.ndarray of shape (3, 3), dtype float64
        """
        K = np.array([
            [fx / (cx * 2), 0.0,           0.5],
            [0.0,           fy / (cy * 2), 0.5],
            [0.0,           0.0,           1.0],
        ])
        return K

    # ── 5. Pipeline-ready tensors ────────────────────────────────────────

    def to_pipeline_input(self, action_str: str):
        """
        Convert action string to pipeline-ready tensors.
        Matches the output format of pose_to_input() in generate.py.

        Args:
            action_str: str of length num_frames - 1

        Returns:
            (viewmats, Ks, action_label):
              viewmats:     torch.Tensor (T_lat, 4, 4)  — latent-aligned w2c
              Ks:           torch.Tensor (T_lat, 3, 3)  — normalized intrinsics
              action_label: torch.Tensor (T_lat,)        — action labels (0-80)
        """
        # Full w2c -> latent-aligned w2c
        w2c_full = self.to_w2c(action_str)
        w2c_lat = self.to_latent_w2c(w2c_full)  # (T_lat, 4, 4)
        T_lat = w2c_lat.shape[0]

        # Intrinsics: same normalized K for all frames
        K = self.get_normalized_intrinsics()
        Ks = np.tile(K, (T_lat, 1, 1))  # (T_lat, 3, 3)

        # Action labels: compute from relative c2w (same logic as pose_to_input)
        c2ws = np.linalg.inv(w2c_lat)
        C_inv = np.linalg.inv(c2ws[:-1])
        relative_c2w = np.zeros_like(c2ws)
        relative_c2w[0] = c2ws[0]
        relative_c2w[1:] = C_inv @ c2ws[1:]

        trans_one_hot = np.zeros((T_lat, 4), dtype=np.int32)
        rotate_one_hot = np.zeros((T_lat, 4), dtype=np.int32)
        move_norm_valid = 0.0001

        for i in range(1, T_lat):
            move_dirs = relative_c2w[i, :3, 3]
            move_norms = np.linalg.norm(move_dirs)

            if move_norms > move_norm_valid:
                move_norm_dirs = move_dirs / move_norms
                angles_rad = np.arccos(np.clip(move_norm_dirs, -1.0, 1.0))
                trans_angles_deg = np.degrees(angles_rad)
            else:
                trans_angles_deg = np.zeros(3)

            R_rel = relative_c2w[i, :3, :3]
            rot_angles_deg = Rotation.from_matrix(R_rel).as_euler("xyz", degrees=True)

            # Translation labels
            if move_norms > move_norm_valid:
                if trans_angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1   # forward
                elif trans_angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1   # backward
                if trans_angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1   # right
                elif trans_angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1   # left

            # Rotation labels
            if rot_angles_deg[1] > 5e-2:
                rotate_one_hot[i, 0] = 1      # yaw right
            elif rot_angles_deg[1] < -5e-2:
                rotate_one_hot[i, 1] = 1      # yaw left
            if rot_angles_deg[0] > 5e-2:
                rotate_one_hot[i, 2] = 1      # pitch up
            elif rot_angles_deg[0] < -5e-2:
                rotate_one_hot[i, 3] = 1      # pitch down

        trans_label = _one_hot_to_label(torch.tensor(trans_one_hot))
        rotate_label = _one_hot_to_label(torch.tensor(rotate_one_hot))
        action_label = trans_label * 9 + rotate_label

        return (
            torch.as_tensor(w2c_lat).float(),
            torch.as_tensor(Ks).float(),
            action_label,
        )

    # ── 6. Visualize ─────────────────────────────────────────────────────

    @staticmethod
    def visualize(action_str: str, output_path: str = "camera_traj.png"):
        """Visualize trajectory: [Top-down XZ] [Side YZ] [Action timeline]."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        c2w = np.eye(4)
        c2ws = [c2w.copy()]
        for ch in action_str:
            if ch in _LOCAL_TRANS:
                c2w[:3, 3] += c2w[:3, :3] @ _LOCAL_TRANS[ch]
            elif ch in _ROT_FUNC:
                c2w[:3, :3] = c2w[:3, :3] @ _ROT_FUNC[ch]()
            c2ws.append(c2w.copy())
        c2ws = np.array(c2ws)
        pos = c2ws[:, :3, 3]
        T = len(pos)

        blocks = []
        if len(action_str) > 0:
            cur, cnt = action_str[0], 1
            for ch in action_str[1:]:
                if ch == cur:
                    cnt += 1
                else:
                    blocks.append((cur, cnt))
                    cur, cnt = ch, 1
            blocks.append((cur, cnt))

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        for ax, (xi, zi, xlabel, zlabel, title) in zip(axes[:2], [
            (0, 2, 'X', 'Z', 'Top-down (XZ)'),
            (1, 2, 'Y', 'Z', 'Side view (YZ)'),
        ]):
            off = 0
            for action, size in blocks:
                seg = slice(off, off + size + 1)
                ax.plot(pos[seg, xi], pos[seg, zi],
                        color=ACTION_COLORS.get(action, '#95a5a6'), lw=2, alpha=0.7)
                off += size
            ax.scatter(pos[:, xi], pos[:, zi], c=np.arange(T), cmap='coolwarm', s=15, zorder=5)
            ax.scatter(pos[0, xi], pos[0, zi], c='lime', s=150, marker='*', zorder=10, label='Start')
            ax.scatter(pos[-1, xi], pos[-1, zi], c='red', s=100, marker='s', zorder=10, label='End')
            ax.set_xlabel(xlabel); ax.set_ylabel(zlabel); ax.set_title(title)
            ax.set_aspect('equal'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[2]
        off = 0
        for action, size in blocks:
            ax.barh(0, size, left=off, height=0.8,
                    color=ACTION_COLORS.get(action, '#95a5a6'), edgecolor='white', lw=0.5)
            if size > 8:
                ax.text(off + size / 2, 0, action, ha='center', va='center',
                        fontsize=8, fontweight='bold')
            off += size
        ax.set_xlabel('Frame'); ax.set_title('Action Timeline')
        ax.set_yticks([]); ax.set_xlim(0, off)
        used = {a for a, _ in blocks}
        ax.legend(handles=[Patch(facecolor=ACTION_COLORS[a], label=ACTION_LABELS[a])
                           for a in ACTIONS if a in used],
                  loc='upper right', fontsize=7, ncol=2)

        block_str = " ".join(f"{a}x{s}" for a, s in blocks)
        fig.suptitle(f"{block_str}  |  {T} frames", fontsize=10, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    # ── 7. JSON export (compatible with pose_to_input) ─────────────────

    @staticmethod
    def action_str_to_blocks(action_str: str) -> list:
        """Parse action string into [(action, count), ...] blocks."""
        if not action_str:
            return []
        blocks = []
        cur, cnt = action_str[0], 1
        for ch in action_str[1:]:
            if ch == cur:
                cnt += 1
            else:
                blocks.append((cur, cnt))
                cur, cnt = ch, 1
        blocks.append((cur, cnt))
        return blocks

    @staticmethod
    def blocks_to_summary(blocks: list) -> str:
        """Convert blocks to human-readable summary like 'W*16,D*32,W*28'."""
        return ','.join(f'{a}*{c}' for a, c in blocks)

    @staticmethod
    def blocks_to_pose_str(blocks: list) -> str:
        """Convert pixel-frame blocks to pose_string format for pose_to_input().

        Pose string format: "w-4, left-8, d-7"
        where the number is the duration in latent frames (pixel_frames / 4).

        Mapping: W->w, S->s, A->a, D->d, J->left, L->right, I->up, K->down
        """
        _MAP = {
            'W': 'w', 'S': 's', 'A': 'a', 'D': 'd',
            'J': 'left', 'L': 'right', 'I': 'up', 'K': 'down',
        }
        parts = []
        for action, pixel_count in blocks:
            latent_count = pixel_count // 4
            if latent_count > 0:
                parts.append(f"{_MAP[action]}-{latent_count}")
        return ', '.join(parts)

    @staticmethod
    def blocks_to_pose_str(blocks: list) -> str:
        """Convert pixel-frame blocks to pose_string format for pose_to_input().

        Pose string format: "w-4, left-8, d-7"
        where the number is the duration in *latent* frames (pixel_frames / 4).
        """
        _MAP = {
            'W': 'w', 'S': 's', 'A': 'a', 'D': 'd',
            'J': 'left', 'L': 'right', 'I': 'up', 'K': 'down',
        }
        parts = []
        for action, pixel_count in blocks:
            latent_count = pixel_count // 4
            if latent_count > 0:
                parts.append(f"{_MAP[action]}-{latent_count}")
        return ", ".join(parts)

    @staticmethod
    def to_json(c2ws: np.ndarray, metadata: dict = None) -> dict:
        """
        Convert c2w matrices to JSON format compatible with pose_to_input().

        Args:
            c2ws: (N, 4, 4) c2w matrices (first is identity)
            metadata: optional dict with extra info

        Returns:
            dict with keys "0", "1", ..., each containing "extrinsic" and "K".
            If metadata is provided, it's stored under "_metadata" key.
        """
        intrinsic = [
            [DEFAULT_FX, 0.0, DEFAULT_CX],
            [0.0, DEFAULT_FY, DEFAULT_CY],
            [0.0, 0.0, 1.0],
        ]
        result = {}
        if metadata:
            result["_metadata"] = metadata
        for i, c2w in enumerate(c2ws):
            result[str(i)] = {
                "extrinsic": c2w.tolist(),
                "K": intrinsic,
            }
        return result

    def generate_json(self, seed: int = None, output_path: str = None) -> dict:
        """
        Generate a random camera trajectory and export as JSON.

        Uses the frame-level generation (generate() -> to_latent_c2w())
        to produce 20 latent c2w poses for 77 frames.

        Args:
            seed: optional seed override (uses instance seed if None)
            output_path: optional path to save JSON

        Returns:
            dict compatible with pose_to_input()
        """
        import json as _json

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        action_str = self.generate()
        c2ws = self.to_latent_c2w(action_str)
        blocks = self.action_str_to_blocks(action_str)
        summary = self.blocks_to_summary(blocks)

        metadata = {
            "motion": summary,
            "action_str": action_str,
            "num_latents": len(c2ws),
            "num_frames": self.num_frames,
        }
        result = self.to_json(c2ws, metadata=metadata)

        if output_path:
            with open(output_path, 'w') as f:
                _json.dump(result, f, indent=2)
            print(f"Saved: {output_path}  motion={summary}  "
                  f"({len(c2ws)} latents, {self.num_frames} frames)")

        return result

    # ── Convenience ──────────────────────────────────────────────────────

    @staticmethod
    def w2c_to_pose7(w2c: np.ndarray) -> np.ndarray:
        """Convert w2c (N, 4, 4) to (N, 7) poses [tx,ty,tz,qx,qy,qz,qw]."""
        N = w2c.shape[0]
        poses = np.zeros((N, 7), dtype=np.float32)
        for i in range(N):
            poses[i, :3] = w2c[i, :3, 3]
            poses[i, 3:] = Rotation.from_matrix(w2c[i, :3, :3]).as_quat()
        return poses


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, json as _json, os

    parser = argparse.ArgumentParser(description="Generate camera trajectory JSON files")
    parser.add_argument('-o', '--output_dir', type=str, default='./assets/pose/generated')
    parser.add_argument('-n', type=int, default=20, help='Number of trajectories')
    parser.add_argument('--num_frames', type=int, default=77)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.n):
        cam = CameraTrajectory(num_frames=args.num_frames, seed=args.seed + i)
        action_str = cam.generate()
        blocks = cam.action_str_to_blocks(action_str)
        summary = cam.blocks_to_summary(blocks)
        safe_name = summary.replace(',', '_').replace('*', 'x')
        out_path = os.path.join(args.output_dir, f"pose_{i:03d}_{safe_name}.json")
        cam.generate_json(seed=args.seed + i, output_path=out_path)
