"""
model.py
Arquitecturas de modelos.

Contiene:
    - LandmarkCNN: CNN diseñada desde cero para Fase 2
    - get_transfer_model(): ResNet50 adaptada para Fase 3
"""

import torch
import torch.nn as nn
from torchvision import models


class LandmarkCNN(nn.Module):
    """
    CNN custom con 4 bloques convolucionales + clasificador.

    Diseño:
    - Input: (3, 224, 224)
    - 4 bloques Conv→BatchNorm→ReLU→MaxPool
    - Cada bloque duplica los canales y reduce el tamaño a la mitad
    - Dropout agresivo para combatir overfitting (dataset chico)
    - Salida: 50 logits

    Flujo de dimensiones:
        (3, 224, 224)
        → Bloque1 → (32, 112, 112)
        → Bloque2 → (64, 56, 56)
        → Bloque3 → (128, 28, 28)
        → Bloque4 → (256, 14, 14)
        → GlobalAvgPool → (256)
        → FC + Dropout → (512)
        → FC salida → (50)

    Justificación del diseño (responder en el notebook):
    - 4 capas convolucionales (mínimo 3 exigido): balance entre capacidad
      y riesgo de overfitting con 100 imgs/clase.
    - BatchNorm después de cada conv: acelera convergencia y estabiliza.
    - Dropout 0.5: agresivo a propósito, dataset chico es propenso a overfit.
    - GlobalAvgPool en vez de Flatten masivo: reduce parámetros drásticamente
      (de ~50M a ~130K solo en la transición), mejor generalización.
    """

    def __init__(self, num_classes=50, dropout_rate=0.5):
        super().__init__()

        # Bloque 1: 3 → 32 canales
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloque 2: 32 → 64 canales
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloque 3: 64 → 128 canales
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloque 4: 128 → 256 canales
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global Average Pooling: promedia espacialmente cada feature map.
        # Reduce (256, 14, 14) a (256,) sin necesidad de flatten masivo.
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Clasificador: 256 → 512 → num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
            # Nota: NO ponemos Softmax acá. CrossEntropyLoss lo aplica internamente.
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x


def get_transfer_model(num_classes=50, freeze_backbone=True):
    """
    Retorna una ResNet50 preentrenada en ImageNet, con la cabeza
    adaptada para num_classes clases.

    Args:
        num_classes: número de clases de salida
        freeze_backbone: si True, congela todas las capas excepto la cabeza
                        (para Fase 3 primera pasada)

    Returns:
        model: nn.Module listo para entrenar
    """
    # Cargamos ResNet50 con pesos preentrenados en ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Congelamos todas las capas si corresponde
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Reemplazamos la capa final (fc) por una nueva adaptada a 50 clases
    # La nueva capa tiene requires_grad=True por default
    num_features = model.fc.in_features  # 2048 en ResNet50
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, num_classes)
    )

    return model


def count_parameters(model):
    """Utilidad: cuenta parámetros totales y entrenables."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
