"""ADEPT Appendix H vision and visuo-tactile student architecture."""

from __future__ import annotations

import math

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class SharedResNetCrossAttentionVision(nn.Module):
    """Shared ResNet-18-style backbone and two-view cross-attention fuser."""

    def __init__(self, latent_dim: int = 256, tokens_per_view: int = 16):
        super().__init__()
        side = math.isqrt(tokens_per_view)
        if side * side != tokens_per_view:
            raise ValueError("tokens_per_view must be a perfect square")
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d((side, side)),
        )
        self.left_to_center = nn.MultiheadAttention(256, 4, batch_first=True)
        self.center_to_left = nn.MultiheadAttention(256, 4, batch_first=True)
        self.projection = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, latent_dim), nn.ELU()
        )

    def forward(self, left: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        if left.shape != center.shape or left.ndim != 4 or left.shape[1] != 3:
            raise ValueError("left and center images must both have shape (B, 3, H, W)")
        batch = left.shape[0]
        features = self.backbone(torch.cat((left, center), dim=0))
        tokens = features.flatten(2).transpose(1, 2)
        left_tokens, center_tokens = tokens[:batch], tokens[batch:]
        left_cross, _ = self.left_to_center(left_tokens, center_tokens, center_tokens)
        center_cross, _ = self.center_to_left(center_tokens, left_tokens, left_tokens)
        return self.projection(
            torch.cat((left_cross.mean(1), center_cross.mean(1)), dim=-1)
        )


class TactileEncoder(nn.Module):
    """Shared two-channel per-finger encoder specified in Appendix H."""

    def __init__(self, feature_dim: int = 32, contact_threshold: float = 1.0 / 255.0):
        super().__init__()
        self.contact_threshold = contact_threshold
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, 2, 1),
            nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ELU(),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim != 4:
            raise ValueError("per-finger tactile depth must have shape (B, 1, H, W)")
        contact = (depth >= self.contact_threshold).to(depth.dtype)
        return self.cnn(torch.cat((depth, contact), dim=1))


def fourier_position(position: torch.Tensor, bands: int = 4) -> torch.Tensor:
    """Gamma(x) = [x, sin(2^b pi x), cos(2^b pi x)] for b=0..B-1."""

    encodings = [position]
    for band in range(bands):
        phase = (2**band) * math.pi * position
        encodings.extend((phase.sin(), phase.cos()))
    return torch.cat(encodings, dim=-1)


class SpatiallyAnchoredTactile(nn.Module):
    def __init__(
        self,
        num_fingers: int = 5,
        feature_dim: int = 32,
        fourier_bands: int = 4,
        modulation_scale: float = 0.1,
    ):
        super().__init__()
        self.num_fingers = num_fingers
        self.feature_dim = feature_dim
        self.fourier_bands = fourier_bands
        self.modulation_scale = modulation_scale
        self.encoder = TactileEncoder(feature_dim)
        position_dim = 3 * (1 + 2 * fourier_bands)
        self.film = nn.Sequential(
            nn.Linear(position_dim, 128),
            nn.ELU(),
            nn.Linear(128, 2 * feature_dim),
        )

    @property
    def output_dim(self) -> int:
        return self.num_fingers * self.feature_dim

    def forward(self, depth: torch.Tensor, fingertip_position: torch.Tensor) -> torch.Tensor:
        if depth.ndim != 5 or depth.shape[1] != self.num_fingers or depth.shape[2] != 1:
            raise ValueError("tactile input must have shape (B, fingers, 1, H, W)")
        if fingertip_position.shape != (depth.shape[0], self.num_fingers, 3):
            raise ValueError("fingertip positions must have shape (B, fingers, 3)")
        batch = depth.shape[0]
        feature = self.encoder(depth.flatten(0, 1)).view(batch, self.num_fingers, -1)
        gamma, beta = self.film(fourier_position(fingertip_position, self.fourier_bands)).chunk(2, -1)
        anchored = (1.0 + self.modulation_scale * gamma) * feature
        anchored = anchored + self.modulation_scale * beta
        return anchored.flatten(1)


class AdeptStudentPolicy(nn.Module):
    """Two-camera student; tactile is enabled for the Flexiv-Sharpa variant."""

    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        *,
        use_tactile: bool,
        fixed_log_std: float = -2.0,
        vision_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.use_tactile = use_tactile
        self.vision = vision_encoder or SharedResNetCrossAttentionVision(256)
        self.tactile = SpatiallyAnchoredTactile() if use_tactile else None
        tactile_dim = self.tactile.output_dim if self.tactile is not None else 0
        self.fusion = nn.Sequential(
            nn.Linear(256 + tactile_dim + proprio_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
        )
        self.lstm = nn.LSTM(512, 1024, batch_first=True)
        self.lstm_norm = nn.LayerNorm(1024)
        self.action_mean = nn.Linear(1024, action_dim)
        self.keypoints = nn.Linear(256, 8 * 3)
        self.register_buffer("fixed_log_std", torch.full((action_dim,), fixed_log_std))

    def forward(
        self,
        left_rgb: torch.Tensor,
        center_rgb: torch.Tensor,
        proprioception: torch.Tensor,
        tactile_depth: torch.Tensor | None = None,
        fingertip_position: torch.Tensor | None = None,
        recurrent_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        vision = self.vision(left_rgb, center_rgb)
        latents = [vision, proprioception]
        if self.tactile is not None:
            if tactile_depth is None or fingertip_position is None:
                raise ValueError("tactile policy requires depth maps and fingertip positions")
            latents.insert(1, self.tactile(tactile_depth, fingertip_position))
        elif tactile_depth is not None or fingertip_position is not None:
            raise ValueError("vision-only policy does not accept tactile inputs")
        fused = self.fusion(torch.cat(latents, dim=-1)).unsqueeze(1)
        recurrent, state = self.lstm(fused, recurrent_state)
        recurrent = self.lstm_norm(recurrent[:, 0])
        return {
            "mean": self.action_mean(recurrent),
            "log_std": self.fixed_log_std.expand(recurrent.shape[0], -1),
            "keypoints": self.keypoints(vision).view(-1, 8, 3),
            "recurrent_state": state,
        }


def make_kuka_vision_student() -> AdeptStudentPolicy:
    return AdeptStudentPolicy(proprio_dim=206, action_dim=23, use_tactile=False)


def make_flexiv_sharpa_visuotactile_student() -> AdeptStudentPolicy:
    return AdeptStudentPolicy(proprio_dim=196, action_dim=29, use_tactile=True)

