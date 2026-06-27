import torch
import torch.nn as nn

class MLPMultitask(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), hidden_dim=256, dropout_prob=0.3, use_dropout=True):
        super(MLPMultitask, self).__init__()
        
        # Calcular el tamaño del vector aplanado (3 * 224 * 224 = 150528)
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        
        # Extractor de características común (capas compartidas)
        modules = [
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU()
        ]
        if use_dropout:
            modules.append(nn.Dropout(dropout_prob))
            
        # Podemos agregar una segunda capa oculta para darle más capacidad
        modules.extend([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        ])
        if use_dropout:
            modules.append(nn.Dropout(dropout_prob))
            
        self.shared_features = nn.Sequential(*modules)
        
        # Cabezal de clasificación de género (Salida: 2 clases)
        self.gender_head = nn.Linear(hidden_dim // 2, 2)
        
        # Cabezal de regresión de edad (Salida: 1 valor continuo)
        self.age_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        # Pasar por las capas compartidas
        features = self.shared_features(x)
        
        # Bifurcación en las dos tareas
        gender_logits = self.gender_head(features)
        age_pred = self.age_head(features).squeeze(-1) # Remover dimensión extra para que quede (batch_size,)
        
        return gender_logits, age_pred


# Clases para las variantes de ablación obligatorias (E2_ablacion_1 y E2_ablacion_2)

class MLPMultitaskNoDropout(MLPMultitask):
    """Variante de ablación 1: MLP sin Dropout para evaluar sobreajuste"""
    def __init__(self, input_shape=(3, 224, 224), hidden_dim=256):
        super().__init__(input_shape=input_shape, hidden_dim=hidden_dim, use_dropout=False)


class MLPMultitaskWide(MLPMultitask):
    """Variante de ablación 2: MLP más ancha (mayor número de neuronas ocultas)"""
    def __init__(self, input_shape=(3, 224, 224), hidden_dim=512, dropout_prob=0.3):
        super().__init__(input_shape=input_shape, hidden_dim=hidden_dim, dropout_prob=dropout_prob, use_dropout=True)