import torch
import torch.nn as nn
from src.models.base import BaseMultiTaskModel

class MLPMultitask(BaseMultiTaskModel):
    def __init__(self, input_shape=(3, 224, 224), hidden_dim=256, dropout_prob=0.3, use_dropout=True):
        super(MLPMultitask, self).__init__()
        
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        
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
        self.gender_head = nn.Linear(hidden_dim // 2, 2)
        self.age_head = nn.Linear(hidden_dim // 2, 1)

        # === SOLUCIÓN AL ERROR ===
        # Si el runner del profe llama al método con guion bajo, lo redirigimos al método correcto
        self._count_trainable_parameters = self.count_trainable_parameters

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_features(images)
        gender_logits = self.gender_head(images)
        age_pred = self.age_head(features).squeeze(-1)
        return gender_logits, age_pred


class MLPMultitaskNoDropout(MLPMultitask):
    """Variante de ablación: Evaluar el impacto de remover la regularización Dropout"""
    def __init__(self, input_shape=(3, 224, 224), hidden_dim=256):
        super().__init__(input_shape=input_shape, hidden_dim=hidden_dim, use_dropout=False)