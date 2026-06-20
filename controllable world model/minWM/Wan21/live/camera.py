"""Live camera/action controller for the minWM streaming worker.

Turns a continuous velocity state {forward, strafe, turn, pitch, up, speed} into
per-frame camera poses (viewmats/Ks) for the camera-conditioned Wan Action2V model.

We reuse `_generate_c2w_trajectory` from `wan_utils.camera_trajectory` — it already
integrates a list of per-frame motion dicts (each may carry arbitrary float
forward/right/up/yaw/pitch) into c2w poses, matching the training pipeline. The stock
inference path builds these from a fixed symbolic string (e.g. "a*4,w*8,s*7"); here we
build them frame-by-frame from a live velocity state instead.

Conventions (from camera_trajectory.py):
  forward = +Z (camera local),  right = +X,  up = -Y (OpenCV Y-down)
  yaw  > 0 == 'l' (turn right),  pitch > 0 == 'i' (look up)
  per-step training increment: 0.08 unit translation, 3 deg rotation.
Intrinsics match the working demo (wan_inference.py): fx=fy=cx=cy=0.5 (normalized).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WAN21 = os.path.dirname(_HERE)
for _p in (_HERE, _WAN21):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from wan_utils.camera_trajectory import _generate_c2w_trajectory, _rot_x, _rot_y, _STEP, _ROT_STEP

# Velocity components the controller understands (all in [-1, 1] except speed).
_AXES = ("forward", "strafe", "turn", "pitch", "up")


class CameraController:
    """Maintains a global per-frame motion history and emits windowed viewmats/Ks.

    Frame/motion bookkeeping: a trajectory of N frames is integrated from N-1 motion
    dicts (frame 0 is identity, each motion transitions frame i-1 -> i). We keep the
    *global* motion list and, every step, slice the last (W+K-1) motions so the window
    is re-zeroed to identity at its first frame — what PRoPE's relative geometry needs.
    """

    def __init__(self, fx=0.5, fy=0.5, cx=0.5, cy=0.5,
                 step=_STEP, rot_step=_ROT_STEP, max_scale=2.0, ema=0.2,
                 device="cuda", dtype=torch.bfloat16):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.step, self.rot_step = step, rot_step
        self.max_scale = max_scale          # clamp |per-frame| to max_scale * increment
        self.ema = ema                      # weight of the *new* command (0..1)
        self.device = torch.device(device)
        self.dtype = dtype
        self.motions = []                   # global per-frame motion dicts
        self._smoothed = self._zero_state()

    @staticmethod
    def _zero_state():
        return {"forward": 0.0, "strafe": 0.0, "turn": 0.0, "pitch": 0.0, "up": 0.0, "speed": 1.0}

    def _K_np(self, T):
        K = np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0, 0, 1]], dtype=np.float32)
        return np.tile(K, (T, 1, 1))

    def _tensors_from(self, motions, T):
        """motions: list of T-1 dicts -> (viewmats[1,T,4,4], Ks[1,T,3,3]) on device."""
        c2w = _generate_c2w_trajectory(motions)            # len == len(motions)+1 == T
        assert len(c2w) == T, f"expected {T} poses, got {len(c2w)}"
        viewmats_np = np.stack([np.linalg.inv(m) for m in c2w]).astype(np.float32)  # w2c
        Ks_np = self._K_np(T)
        vm = torch.from_numpy(viewmats_np).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        ks = torch.from_numpy(Ks_np).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        return vm, ks

    # --- public API ---------------------------------------------------------
    def bootstrap_tensors(self, n):
        """Reset history to a static (identity) trajectory of n frames."""
        self.motions = [dict() for _ in range(n - 1)]
        self._smoothed = self._zero_state()
        return self._tensors_from(self.motions, n)

    def velocity_to_motions(self, cs, K):
        """Map a velocity state dict to K per-frame motion dicts (EMA-smoothed)."""
        s = self._smoothed
        for key in _AXES:
            s[key] = (1.0 - self.ema) * s[key] + self.ema * float(cs.get(key, 0.0))
        s["speed"] = float(cs.get("speed", 1.0))
        sp = s["speed"]

        def cl(x):
            return max(-self.max_scale, min(self.max_scale, x))

        move = {
            "forward": cl(s["forward"] * sp) * self.step,    # +Z local
            "right":   cl(s["strafe"] * sp) * self.step,     # +X local
            "up":      cl(s["up"] * sp) * self.step,         # -Y handled in _generate
            "yaw":     cl(s["turn"] * sp) * self.rot_step,   # +yaw == turn right ('l')
            "pitch":   cl(s["pitch"] * sp) * self.rot_step,  # +pitch == look up ('i')
        }
        return [dict(move) for _ in range(K)]   # constant velocity across the chunk

    def extend(self, motions):
        self.motions.extend(motions)

    def window_tensors(self, W, K):
        """viewmats/Ks for the [W context + K new] window, re-zeroed to identity.

        Used by the recache-every-chunk worker (MinWMWorker, Option A)."""
        win = self.motions[-(W + K - 1):]       # last W+K-1 motions -> W+K poses
        return self._tensors_from(win, W + K)

    # --- global continuous trajectory (for the persistent-cache streaming worker) ---
    # The persistent KV/PRoPE cache holds history in ABSOLUTE poses, so each new chunk
    # must continue the single global trajectory (NOT re-zeroed). We track the running
    # global c2w pose incrementally and only emit the new frames' w2c.
    def reset_global(self):
        self.motions = []
        self._pose = np.eye(4, dtype=np.float64)   # running global c2w; frame 0 = identity
        self._poses = [self._pose.copy()]          # all global c2w poses so far
        self._smoothed = self._zero_state()

    @staticmethod
    def _integrate(move, T):
        """One per-frame motion onto global c2w T (same convention as
        _generate_c2w_trajectory)."""
        T = T.copy()
        if "yaw" in move:
            T[:3, :3] = T[:3, :3] @ _rot_y(move["yaw"])
        if "pitch" in move:
            T[:3, :3] = T[:3, :3] @ _rot_x(move["pitch"])
        if move.get("forward", 0.0):
            T[:3, 3] += T[:3, :3] @ np.array([0, 0, move["forward"]])
        if move.get("right", 0.0):
            T[:3, 3] += T[:3, :3] @ np.array([move["right"], 0, 0])
        if move.get("up", 0.0):
            T[:3, 3] += T[:3, :3] @ np.array([0, -move["up"], 0])
        return T

    def extend_global(self, motions):
        for m in motions:
            self._pose = self._integrate(m, self._pose)
            self._poses.append(self._pose)
        self.motions.extend(motions)

    def global_new_tensors(self, K):
        """viewmats/Ks for the LAST K global frames (the new chunk), absolute poses."""
        new = self._poses[-K:]
        viewmats_np = np.stack([np.linalg.inv(p) for p in new]).astype(np.float32)
        vm = torch.from_numpy(viewmats_np).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        ks = torch.from_numpy(self._K_np(K)).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        return vm, ks
