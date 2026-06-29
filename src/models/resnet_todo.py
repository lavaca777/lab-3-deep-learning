"""Exercise placeholder for ResNet transfer learning strategies."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from src.models.base import BaseMultiTaskModel


class MultiTaskResNet(BaseMultiTaskModel):
    """Implementación de E4 y E5 con torchvision.models.resnet18."""

    # Se agregó 'dropout: float = 0.0' a los parámetros
    def __init__(self, num_unfrozen_blocks: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        
        # 1. Cargar el modelo base preentrenado
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 2. Congelar todo el backbone inicialmente (Útil para E4)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 3. Descongelar bloques específicos de atrás hacia adelante para fine-tuning (E5)
        # ResNet18 está compuesta por layer1, layer2, layer3 y layer4.
        blocks_to_unfreeze = []
        if num_unfrozen_blocks >= 1:
            blocks_to_unfreeze.append(self.backbone.layer4)
        if num_unfrozen_blocks >= 2:
            blocks_to_unfreeze.append(self.backbone.layer3)
        if num_unfrozen_blocks >= 3:
            blocks_to_unfreeze.append(self.backbone.layer2)
        if num_unfrozen_blocks >= 4:
            blocks_to_unfreeze.append(self.backbone.layer1)

        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True

        # 4. Extraer la cantidad de características de salida y reemplazar la capa original (fc)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 5. Definir la capa de dropout y los nuevos cabezales independientes
        self.dropout = nn.Dropout(p=dropout)
        self.gender_head = nn.Linear(in_features, 2)
        self.age_head = nn.Linear(in_features, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Extraer características visuales compartidas
        features = self.backbone(images)
        
        # Aplicar regularización Dropout
        features = self.dropout(features)
        
        # Generar salidas de clasificación (género) y regresión (edad)
        gender_logits = self.gender_head(features)
        age_predictions = self.age_head(features).squeeze(1)
        
        return gender_logits, age_predictions