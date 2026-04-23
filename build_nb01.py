"""Construye notebooks/01_cnn_from_scratch.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Fase 2: CNN desde Cero (From Scratch)

## Objetivo
Diseñar, entrenar y evaluar una CNN propia (sin transfer learning) que alcance al menos **40% de test accuracy** en clasificación de 50 landmarks.

## Justificación de la arquitectura

Nuestra CNN tiene 4 bloques convolucionales, cada uno con:
- `Conv2D` (kernel 3×3, padding 1): extrae features manteniendo el tamaño
- `BatchNorm2D`: estabiliza y acelera el entrenamiento, reduce sensibilidad al learning rate
- `ReLU`: no-linealidad, computacionalmente barata y sin problema de vanishing gradient
- `MaxPool2D (2×2)`: reduce dimensión a la mitad y da cierta invariancia a traslaciones

**Progresión de canales:** 3 → 32 → 64 → 128 → 256. Duplicar canales al reducir resolución es un patrón estándar que mantiene capacidad de representación.

**Global Average Pooling** en vez de Flatten gigante: reduce drásticamente parámetros (de ~50M a ~130K en la transición a fully-connected), mejorando generalización en datasets chicos.

**Dropout 0.5** agresivo: con solo 100 imgs/clase, el riesgo de overfitting es alto. Dropout agresivo fuerza a la red a no depender de neuronas específicas.

**Total de parámetros:** ~1.2M (compacto vs ResNet50 de 25M, pero adecuado para este dataset)."""))

cells.append(nbf.v4.new_code_cell('''import sys
sys.path.append(\'..\')  # Para importar desde src/

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.data import get_dataloaders
from src.model import LandmarkCNN, count_parameters
from src.train import train_model, validate

# Reproducibilidad
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
'''))

cells.append(nbf.v4.new_code_cell('''train_loader, val_loader, test_loader, classes = get_dataloaders(
    data_dir="../data",
    batch_size=32,
    val_split=0.15,
    num_workers=2,
    seed=42
)

NUM_CLASES = len(classes)
print(f"Clases: {NUM_CLASES}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}")
'''))

cells.append(nbf.v4.new_code_cell('''model = LandmarkCNN(num_classes=NUM_CLASES, dropout_rate=0.5).to(device)

total, trainable = count_parameters(model)
print(f"Parámetros totales: {total:,}")
print(f"Parámetros entrenables: {trainable:,}")
print(f"\\nArquitectura:\\n{model}")
'''))

cells.append(nbf.v4.new_code_cell('''# Configuración del entrenamiento
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

# Loss y optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Scheduler: reduce LR cuando val_loss se estanca
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=\'min\', factor=0.5, patience=3)

# Entrenar
SAVE_PATH = "../models/cnn_scratch_best.pt"
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    device=device,
    save_path=SAVE_PATH,
    scheduler=scheduler
)
'''))

cells.append(nbf.v4.new_code_cell('''def plot_history(history, title="Training History", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history[\'train_loss\'], label=\'Train\', linewidth=2)
    axes[0].plot(history[\'val_loss\'], label=\'Val\', linewidth=2)
    axes[0].set_xlabel(\'Época\')
    axes[0].set_ylabel(\'Loss\')
    axes[0].set_title(f\'{title} - Loss\')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history[\'train_acc\'], label=\'Train\', linewidth=2)
    axes[1].plot(history[\'val_acc\'], label=\'Val\', linewidth=2)
    axes[1].set_xlabel(\'Época\')
    axes[1].set_ylabel(\'Accuracy (%)\')
    axes[1].set_title(f\'{title} - Accuracy\')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches=\'tight\')
    plt.show()

plot_history(history, title="CNN From Scratch", save_path="../report/curvas_cnn_scratch.png")
'''))

cells.append(nbf.v4.new_code_cell('''# Cargamos el MEJOR modelo (menor val_loss)
model.load_state_dict(torch.load(SAVE_PATH))
model.to(device)

# Evaluamos en test
test_loss, test_acc = validate(model, test_loader, criterion, device)

print(f"\\n{\'=\'*50}")
print(f"RESULTADOS FINALES - CNN FROM SCRATCH")
print(f"{\'=\'*50}")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Umbral mínimo: 40.00%")
print(f"{\'✅ APROBADO\' if test_acc >= 40 else \'❌ NO APROBADO\'}")
print(f"{\'=\'*50}")
'''))

cells.append(nbf.v4.new_code_cell('''# Torch Script: formato optimizado para producción, no depende de la clase Python
# Usamos torch.jit.script que preserva la lógica del forward

model.eval()
example_input = torch.randn(1, 3, 224, 224).to(device)

# Trace: graba el grafo ejecutando un input de ejemplo
traced_model = torch.jit.trace(model, example_input)
TORCHSCRIPT_PATH = "../models/cnn_scratch_torchscript.pt"
traced_model.save(TORCHSCRIPT_PATH)

print(f"Modelo TorchScript exportado a: {TORCHSCRIPT_PATH}")

# Verificación: cargar y probar
loaded_model = torch.jit.load(TORCHSCRIPT_PATH)
loaded_model.eval()
output = loaded_model(example_input)
print(f"Verificación OK - output shape: {output.shape}")
'''))

cells.append(nbf.v4.new_markdown_cell("""## Análisis de resultados

### Rendimiento
- Test accuracy obtenido vs umbral mínimo (40%)
- Observaciones sobre las curvas:
  - ¿Hay overfitting? (train acc >> val acc)
  - ¿Convergió o podría mejorar con más épocas?
  - ¿El scheduler redujo el LR?

### Limitaciones de la CNN from-scratch
1. **Dataset chico (100 imgs/clase):** insuficiente para aprender features visuales genéricas desde cero.
2. **Arquitectura limitada:** aunque 4 capas es razonable, redes modernas usan 50+ capas con conexiones residuales.
3. **Sin conocimiento previo:** la red arranca con pesos aleatorios, mientras ImageNet ofrece millones de imágenes de preentrenamiento.

Estas limitaciones motivan Transfer Learning (Fase 3)."""))

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python (landmark-classifier)", "language": "python", "name": "landmark-classifier"},
    "language_info": {"name": "python", "version": "3.11.9"}
}

out = "notebooks/01_cnn_from_scratch.ipynb"
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Notebook escrito: {out} ({len(cells)} celdas)")
