"""Multitask Multi-Layer Perceptron implemented with PyTorch."""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseMultiTaskModel


class MLPMultitask(BaseMultiTaskModel):
    """Perceptrón Multicapa Multitarea para predecir género y edad."""

    def __init__(self, input_shape: tuple[int, int, int] = (3, 224, 224), hidden_dim: int = 256, dropout_prob: float = 0.3, use_dropout: bool = True) -> None:
        super().__init__()
        
        # Tamaño del vector aplanado (3 * 224 * 224 = 150528)
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        
        # Extractor de características común (capas compartidas)
        modules = [
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU()
        ]
        if use_dropout:
            modules.append(nn.Dropout(p=dropout_prob))
            
        modules.extend([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        ])
        if use_dropout:
            modules.append(nn.Dropout(p=dropout_prob))
            
        self.shared = nn.Sequential(*modules)
        
        # Cabezales finales
        self.gender_head = nn.Linear(hidden_dim // 2, 2)
        self.age_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.shared(images)
        gender_logits = self.gender_head(representation)
        age_predictions = self.age_head(representation).squeeze(1)
        return gender_logits, age_predictions


class MLPMultitaskNoDropout(MLPMultitask):
    """Variante de ablación: MLP sin Dropout para evaluar sobreajuste."""
    def __init__(self, input_shape: tuple[int, int, int] = (3, 224, 224), hidden_dim: int = 256) -> None:
        super().__init__(input_shape=input_shape, hidden_dim=hidden_dim, use_dropout=False)