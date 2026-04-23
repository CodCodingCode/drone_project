"""Standalone reimplementation of the frozen waypoint policy.

The original lives inside HierarchicalVLAActor (vla/vla_policy.py:_waypoint_policy_forward),
but that class pulls in PaliGemma + LoRA + LSTM + cross-attention. For scan +
navigate we only need the 3-layer MLP + obs normalizer. This file loads
`model_2998_waypoint.pt` directly and mirrors the forward pass verbatim.

Input:  flight_state (9,) + target_body (3,) + pos_error_w (3,)  →  obs_15
Output: action (4,)  = [thrust, roll_m, pitch_m, yaw_m]  in [-1, 1]-ish
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, torch.Tensor]


class WaypointController:
    def __init__(
        self,
        ckpt_path: str = "/home/ubuntu/drone_project/model_2998_waypoint.pt",
        device: str | torch.device = "cuda",
    ):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Waypoint checkpoint not found: {ckpt_path}\n"
                f"Run the waypoint-nav training stage first, or download the checkpoint."
            )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["actor_state_dict"]

        self.device = torch.device(device)

        # 3-layer MLP: 15 -> 256 -> 256 -> 4
        self.w0 = state["mlp.0.weight"].to(self.device)
        self.b0 = state["mlp.0.bias"].to(self.device)
        self.w1 = state["mlp.2.weight"].to(self.device)
        self.b1 = state["mlp.2.bias"].to(self.device)
        self.w2 = state["mlp.4.weight"].to(self.device)
        self.b2 = state["mlp.4.bias"].to(self.device)

        # Obs normalizer (mean/std shape (1, 15))
        self.obs_mean = state["obs_normalizer._mean"].to(self.device)
        self.obs_std = state["obs_normalizer._std"].to(self.device)

    # --------------------------------------------------------------
    def _to_tensor(self, x: ArrayLike) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(self.device)
        return x.to(self.device).float()

    @torch.no_grad()
    def act(
        self,
        flight_state: ArrayLike,  # (9,) or (B, 9)
        target_body: ArrayLike,   # (3,) or (B, 3)
        pos_error_w: ArrayLike,   # (3,) or (B, 3)
    ) -> torch.Tensor:
        """Returns action tensor of shape (4,) for single input, else (B, 4)."""
        fs = self._to_tensor(flight_state)
        tb = self._to_tensor(target_body)
        pe = self._to_tensor(pos_error_w)

        single = (fs.dim() == 1)
        if single:
            fs = fs.unsqueeze(0)
            tb = tb.unsqueeze(0)
            pe = pe.unsqueeze(0)

        obs = torch.cat([fs, tb, pe], dim=-1)  # (B, 15)
        x = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        x = F.elu(F.linear(x, self.w0, self.b0))
        x = F.elu(F.linear(x, self.w1, self.b1))
        action = F.linear(x, self.w2, self.b2)

        return action.squeeze(0) if single else action
