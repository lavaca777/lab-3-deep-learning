"""Exercise placeholder for ResNet transfer learning strategies."""

from __future__ import annotations

import torch

from src.models.base import BaseMultiTaskModel


class MultiTaskResNet(BaseMultiTaskModel):
    """TODO(alumno): implement E4 and E5 with torchvision.models.resnet18.

    The implementation should support a frozen backbone and fine-tuning. It
    must replace the original classification layer with separate gender and age
    heads, and expose the number of unfrozen blocks for ablation studies.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "E4/E5 MultiTaskResNet no ha sido implementado. "
            "Complete src/models/resnet_todo.py."
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO(alumno): implementar forward de MultiTaskResNet.")
