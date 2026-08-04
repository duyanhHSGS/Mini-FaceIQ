"""Lean MobileNetV3-Small model with exactly one landmark heatmap."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class DiscreteLandmarkModel(nn.Module):
    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_small(weights=weights).features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder((images - self.mean) / self.std))


def soft_argmax_2d(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, height, width = logits.shape
    probabilities = torch.softmax(logits.reshape(batch, channels, -1), dim=-1).reshape_as(logits)
    x_grid = torch.linspace(0, 1, width, device=logits.device, dtype=logits.dtype).view(1, 1, 1, width)
    y_grid = torch.linspace(0, 1, height, device=logits.device, dtype=logits.dtype).view(1, 1, height, 1)
    coordinates = torch.stack(
        ((probabilities * x_grid).sum((2, 3)), (probabilities * y_grid).sum((2, 3))),
        dim=-1,
    )
    confidence = torch.sigmoid(logits).amax((2, 3))
    return coordinates, confidence


def landmark_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    sigma: float,
    map_weight: float,
    coordinate_weight: float,
) -> dict[str, torch.Tensor]:
    predicted, _ = soft_argmax_2d(logits)
    size = logits.shape[-1]
    grid = torch.arange(size, device=logits.device, dtype=logits.dtype)
    grid_y, grid_x = torch.meshgrid(grid, grid, indexing="ij")
    center_x = targets[..., 0, None, None] * (size - 1)
    center_y = targets[..., 1, None, None] * (size - 1)
    heatmaps = torch.exp(-((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / (2 * sigma**2))
    map_loss = F.mse_loss(torch.sigmoid(logits), heatmaps)
    coordinate_loss = F.smooth_l1_loss(predicted, targets, beta=0.01)
    return {
        "total": map_weight * map_loss + coordinate_weight * coordinate_loss,
        "map": map_loss,
        "coordinate": coordinate_loss,
        "predicted": predicted,
    }
