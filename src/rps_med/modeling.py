"""PyTorch models for image-to-burden and pair-to-spectrum learning."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import resnet18  # type: ignore[import-untyped]


class BurdenEncoder(nn.Module):
    """ResNet-18 encoder with burden and lesion-presence supervision heads."""

    embedding_dim = 512

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.burden_head = nn.Linear(self.embedding_dim, 4)
        self.presence_head = nn.Linear(self.embedding_dim, 4)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        embedding = self.backbone(images)
        return embedding, self.burden_head(embedding), self.presence_head(embedding)


class PairSpectrumHead(nn.Module):
    """Orientation-aware head for a 17-scale relation spectrum."""

    def __init__(
        self,
        *,
        embedding_dim: int = 512,
        scale_count: int = 17,
        relation_count: int = 5,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.scale_count = scale_count
        self.relation_count = relation_count
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim * 3, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spectrum = nn.Linear(256, scale_count * relation_count)
        self.transition = nn.Linear(256, 1)
        self.critical_log2 = nn.Linear(256, 1)

    def forward(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = torch.cat((left - right, torch.abs(left - right), left * right), dim=1)
        hidden = self.trunk(features)
        spectrum = self.spectrum(hidden).reshape(
            -1, self.scale_count, self.relation_count
        )
        return (
            spectrum,
            self.transition(hidden).squeeze(1),
            self.critical_log2(hidden).squeeze(1),
        )


def load_resnet18_imagenet_v1(
    model: BurdenEncoder, weights_path: str | Path
) -> None:
    """Load the official torchvision ResNet-18 V1 backbone state."""

    state: Mapping[str, Any] = torch.load(
        Path(weights_path), map_location="cpu", weights_only=True
    )
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    if tuple(missing) != () or set(unexpected) != {"fc.bias", "fc.weight"}:
        raise ValueError("ResNet-18 pretrained state does not match the frozen backbone")


__all__ = ["BurdenEncoder", "PairSpectrumHead", "load_resnet18_imagenet_v1"]
