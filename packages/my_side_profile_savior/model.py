"""MobileNetV3-Large profile-landmark heatmap network and masked losses."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


LANDMARK_COUNT = 31
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ProfileLandmarkModel(nn.Module):
    """Predict one heatmap for each Mini-FaceIQ side-landmark slot."""

    def __init__(
        self,
        *,
        landmark_count: int = LANDMARK_COUNT,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.encoder = backbone.features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(960, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            nn.Conv2d(64, landmark_count, kernel_size=1),
        )
        self.landmark_count = landmark_count
        self.register_buffer(
            "image_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized = (images - self.image_mean) / self.image_std
        return self.decoder(self.encoder(normalized))


def soft_argmax_2d(
    heatmap_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized x/y coordinates and peak sigmoid confidence."""

    if heatmap_logits.ndim != 4:
        raise ValueError("heatmap_logits must have shape [batch, landmarks, h, w]")
    batch, landmarks, height, width = heatmap_logits.shape
    probabilities = torch.softmax(
        heatmap_logits.reshape(batch, landmarks, -1) / temperature,
        dim=-1,
    ).reshape(batch, landmarks, height, width)

    x_grid = torch.linspace(
        0.0,
        1.0,
        width,
        device=heatmap_logits.device,
        dtype=heatmap_logits.dtype,
    ).view(1, 1, 1, width)
    y_grid = torch.linspace(
        0.0,
        1.0,
        height,
        device=heatmap_logits.device,
        dtype=heatmap_logits.dtype,
    ).view(1, 1, height, 1)
    x = (probabilities * x_grid).sum(dim=(2, 3))
    y = (probabilities * y_grid).sum(dim=(2, 3))
    coordinates = torch.stack((x, y), dim=-1)
    confidence = torch.sigmoid(heatmap_logits).amax(dim=(2, 3))
    return coordinates, confidence


def gaussian_heatmap_targets(
    coordinates: torch.Tensor,
    *,
    heatmap_size: int,
    sigma: float,
) -> torch.Tensor:
    """Create Gaussian maps centered on normalized landmark coordinates."""

    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape [batch, landmarks, 2]")
    dtype = coordinates.dtype
    device = coordinates.device
    grid = torch.arange(heatmap_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(grid, grid, indexing="ij")
    center_x = coordinates[..., 0].unsqueeze(-1).unsqueeze(-1) * (heatmap_size - 1)
    center_y = coordinates[..., 1].unsqueeze(-1).unsqueeze(-1) * (heatmap_size - 1)
    squared_distance = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    return torch.exp(-squared_distance / (2.0 * sigma**2))


def masked_landmark_loss(
    heatmap_logits: torch.Tensor,
    target_coordinates: torch.Tensor,
    active_mask: torch.Tensor,
    visibility_mask: torch.Tensor,
    *,
    sigma: float = 1.5,
    map_weight: float = 1.0,
    coordinate_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Train only human-confirmed, visible landmark slots."""

    predicted_coordinates, _ = soft_argmax_2d(heatmap_logits)
    targets = gaussian_heatmap_targets(
        target_coordinates,
        heatmap_size=heatmap_logits.shape[-1],
        sigma=sigma,
    )
    point_mask = visibility_mask.to(heatmap_logits.dtype) * active_mask.to(
        heatmap_logits.dtype
    ).view(1, -1)
    denominator = point_mask.sum().clamp_min(1.0)

    map_per_point = F.mse_loss(
        torch.sigmoid(heatmap_logits),
        targets,
        reduction="none",
    ).mean(dim=(2, 3))
    coordinate_per_point = F.smooth_l1_loss(
        predicted_coordinates,
        target_coordinates,
        reduction="none",
        beta=0.01,
    ).mean(dim=-1)

    map_loss = (map_per_point * point_mask).sum() / denominator
    coordinate_loss = (coordinate_per_point * point_mask).sum() / denominator
    total = map_weight * map_loss + coordinate_weight * coordinate_loss
    return {
        "total": total,
        "map": map_loss,
        "coordinate": coordinate_loss,
        "predicted_coordinates": predicted_coordinates,
    }
