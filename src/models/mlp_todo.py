import torch
import torch.nn as nn
from src.models.base import BaseMultiTaskModel  # <-- Importamos la clase base

class MLPMultitask(BaseMultiTaskModel):  # <-- Cambiamos nn.Module por BaseMultiTaskModel
    def __init__(self, input_shape=(3, 224, 224), hidden_dim=256, dropout_prob=0.3, use_dropout=True):
        super(MLPMultitask, self).__init__()
        
        # Tamaño del vector aplanado (3 * 224 * 224 = 150528)
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        
        # Bloque extractor común de características
        modules = [
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU()
        ]
        if use_dropout:
            modules.append(nn.Dropout(dropout_prob))
            
        modules.extend([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        ])
        if use_dropout:
            modules.append(nn.Dropout(dropout_prob))
            
        self.shared_features = nn.Sequential(*modules)
        
        # Cabezal de clasificación de género (Salida de tamaño 2)
        self.gender_head = nn.Linear(hidden_dim // 2, 2)
        
        # Cabezal de regresión de edad (Salida de tamaño 1)
        self.age_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # <-- Usamos la firma del profe
        features = self.shared_features(images)
        gender_logits = self.gender_head(features)
        age_pred = self.age_head(features).squeeze(-1)  # Ajustar a dimensión (batch_size,)
        return gender_logits, age_pred


class MLPMultitaskNoDropout(MLPMultitask):
    """Variante de ablación: Evaluar el impacto de remover la regularización Dropout"""
    def __init__(self, input_shape=(3, 224, 224), hidden_dim=256):
        super().__init__(input_shape=input_shape, hidden_dim=hidden_dim, use_dropout=False)