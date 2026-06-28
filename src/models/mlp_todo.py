"""Exercise placeholder for the multitask MLP strategy."""

from __future__ import annotations

import torch

from src.models.base import BaseMultiTaskModel


class MultiTaskMLP(BaseMultiTaskModel):
    """TODO(alumno): implement E2 using PyTorch.

    Suggested steps:
    1. Flatten the RGB image.
    2. Build a shared fully connected representation.
    3. Add one head with two gender logits and one scalar age head.
    4. Expose dropout as a constructor argument so it can be ablated.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "E2 MultiTaskMLP no ha sido implementado. Complete src/models/mlp_todo.py."
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO(alumno): implementar forward de MultiTaskMLP.")
