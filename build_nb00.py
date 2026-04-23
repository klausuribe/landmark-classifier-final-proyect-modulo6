"""Construye notebooks/00_exploracion_y_preproceso.ipynb desde PROYECTO.md (Fase 1)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Celda 1: markdown titulo y contexto
cells.append(nbf.v4.new_markdown_cell("""# Fase 1: Exploración y Preprocesamiento de Datos

## Objetivo
Explorar el dataset de landmarks, visualizar muestras, analizar la distribución de clases e implementar el pipeline de preprocesamiento (transforms + DataLoaders) que se usará en las fases posteriores.

## Dataset
- **Fuente:** Subconjunto del Google Landmarks Dataset v2
- **Clases:** 50 landmarks
- **Train:** 100 imágenes por clase (5.000 total)
- **Test:** 25 imágenes por clase (1.250 total)

## Estructura esperada
```
data/
├── train/
│   ├── <clase_01>/
│   ├── <clase_02>/
│   └── ... (50 carpetas)
└── test/
    ├── <clase_01>/
    └── ... (50 carpetas)
```"""))

# Celda 2: imports y config
cells.append(nbf.v4.new_code_cell('''"""
Imports principales.
- torch y torchvision son el core de PyTorch y visión.
- ImageFolder carga datasets organizados por carpetas (estructura estándar).
- DataLoader es el "iterador" que maneja batches, shuffle y paralelismo.
- transforms define las transformaciones aplicadas a las imágenes.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
from pathlib import Path

# Configuración visual
sns.set_style("whitegrid")
plt.rcParams[\'figure.figsize\'] = (12, 6)

# Reproducibilidad: fijamos semillas para que los resultados sean repetibles.
# Esto es crítico para poder comparar experimentos de manera justa.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Dispositivo: usamos GPU si está disponible, si no CPU.
# En este proyecto debería ser CUDA (RTX 2050).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
'''))

# Celda 3: markdown exploración
cells.append(nbf.v4.new_markdown_cell("""## 1. Exploración estructural del dataset

Antes de definir transformaciones, necesitamos entender:
- Cuántas clases hay
- Cuántas imágenes hay por clase
- Si el dataset está balanceado
- Qué tamaños y formatos tienen las imágenes"""))

# Celda 4: explorar estructura
cells.append(nbf.v4.new_code_cell('''# Rutas del dataset
DATA_DIR = Path("../data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Verificación de existencia
assert TRAIN_DIR.exists(), f"No se encontró {TRAIN_DIR}"
assert TEST_DIR.exists(), f"No se encontró {TEST_DIR}"

# Listar clases (subcarpetas de train)
clases = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
NUM_CLASES = len(clases)
print(f"Número de clases: {NUM_CLASES}")
print(f"Primeras 10 clases: {clases[:10]}")

# Contar imágenes por clase
train_counts = {c: len(list((TRAIN_DIR / c).glob("*.*"))) for c in clases}
test_counts = {c: len(list((TEST_DIR / c).glob("*.*"))) for c in clases}

print(f"\\nTotal imágenes train: {sum(train_counts.values())}")
print(f"Total imágenes test:  {sum(test_counts.values())}")
print(f"Promedio imágenes/clase (train): {np.mean(list(train_counts.values())):.1f}")
print(f"Promedio imágenes/clase (test):  {np.mean(list(test_counts.values())):.1f}")
'''))

# Celda 5: markdown distribución
cells.append(nbf.v4.new_markdown_cell("""## 2. Distribución de clases

Visualizamos la distribución de imágenes por clase para verificar balance. En este dataset esperamos balance perfecto (100/clase en train, 25/clase en test), pero esta verificación es una buena práctica estándar."""))

# Celda 6: grafico distribución
cells.append(nbf.v4.new_code_cell('''# Gráfico de distribución de clases
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Train
axes[0].bar(range(NUM_CLASES), list(train_counts.values()), color=\'steelblue\')
axes[0].set_title("Distribución de clases - TRAIN", fontsize=14, fontweight=\'bold\')
axes[0].set_xlabel("Clase (índice)")
axes[0].set_ylabel("Cantidad de imágenes")
axes[0].axhline(y=np.mean(list(train_counts.values())), color=\'red\', linestyle=\'--\', label=f\'Media: {np.mean(list(train_counts.values())):.0f}\')
axes[0].legend()

# Test
axes[1].bar(range(NUM_CLASES), list(test_counts.values()), color=\'coral\')
axes[1].set_title("Distribución de clases - TEST", fontsize=14, fontweight=\'bold\')
axes[1].set_xlabel("Clase (índice)")
axes[1].set_ylabel("Cantidad de imágenes")
axes[1].axhline(y=np.mean(list(test_counts.values())), color=\'red\', linestyle=\'--\', label=f\'Media: {np.mean(list(test_counts.values())):.0f}\')
axes[1].legend()

plt.tight_layout()
plt.savefig("../report/distribucion_clases.png", dpi=100, bbox_inches=\'tight\')
plt.show()

# Análisis del balance
train_std = np.std(list(train_counts.values()))
print(f"Desviación estándar train: {train_std:.2f}")
if train_std < 5:
    print("Dataset BALANCEADO: no se requieren técnicas de balanceo (class weights, oversampling).")
else:
    print("Dataset DESBALANCEADO: considerar class_weights en CrossEntropyLoss.")
'''))

# Celda 7: markdown visualización
cells.append(nbf.v4.new_markdown_cell("""## 3. Visualización de muestras

Mostramos imágenes de ejemplo de diferentes clases para tener intuición visual del dataset. Esto ayuda a detectar problemas temprano:
- Imágenes corruptas
- Etiquetas inconsistentes
- Diversidad de ángulos, iluminación, escalas"""))

# Celda 8: mostrar muestras
cells.append(nbf.v4.new_code_cell('''# Mostrar 12 imágenes de diferentes clases (más que las 5 mínimas del proyecto para mejor exploración)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

# Seleccionamos 12 clases al azar
clases_muestra = random.sample(clases, 12)

for i, clase in enumerate(clases_muestra):
    # Tomamos la primera imagen de la clase
    imagenes_clase = list((TRAIN_DIR / clase).glob("*.*"))
    if imagenes_clase:
        img_path = imagenes_clase[0]
        img = Image.open(img_path).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(f"Clase: {clase}\\nTamaño: {img.size}", fontsize=10)
        axes[i].axis("off")

plt.suptitle("Muestras del dataset (una por clase aleatoria)", fontsize=14, fontweight=\'bold\')
plt.tight_layout()
plt.savefig("../report/muestras_dataset.png", dpi=100, bbox_inches=\'tight\')
plt.show()
'''))

# Celda 9: markdown pipeline
cells.append(nbf.v4.new_markdown_cell("""## 4. Pipeline de transformaciones

### ¿Por qué hay dos pipelines (train vs val/test)?

**Train:** aplicamos **data augmentation** (rotaciones, flips, color jitter) para aumentar artificialmente la variedad del dataset. Esto ayuda a combatir overfitting — crítico con solo 100 imágenes por clase.

**Validation/Test:** NO aplicamos augmentation. Solo las transformaciones determinísticas necesarias para que la red pueda procesar la imagen (resize, crop central, normalización). Esto asegura que la evaluación sea reproducible y consistente.

### ¿Por qué resize a 256 y luego crop central a 224?

Es el pipeline estándar de ImageNet. Hacer resize directo a 224 distorsiona proporciones; primero se redimensiona al lado más corto a 256, y luego se toma un crop cuadrado de 224×224 del centro. Esto preserva aspect ratio y entrega exactamente el tamaño que la red espera.

### ¿Por qué normalizamos con medias y desvíos de ImageNet?

Porque en Fase 3 vamos a usar ResNet50 preentrenada en ImageNet. Esa red fue entrenada esperando imágenes normalizadas con esos valores específicos. Para mantener consistencia entre fases, usamos la misma normalización tanto en la CNN from-scratch como en Transfer Learning."""))

# Celda 10: definir transforms
cells.append(nbf.v4.new_code_cell('''# Valores de normalización de ImageNet (estándar de facto para transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Pipeline de TRAIN con data augmentation
# Cada transformación sirve para simular variabilidad realista:
# - RandomResizedCrop: simula diferentes encuadres/zoom
# - RandomHorizontalFlip: un landmark fotografiado desde el "otro lado"
# - RandomRotation(15°): pequeñas inclinaciones de cámara
# - ColorJitter: diferentes iluminaciones (mañana, tarde, nublado)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),  # Convierte PIL Image a Tensor [0,1] con shape (C,H,W)
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Pipeline de VALIDATION/TEST: determinístico, sin augmentation
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

print("Transforms definidos correctamente:")
print(f"  Train transform: {len(train_transform.transforms)} operaciones (incluye augmentation)")
print(f"  Val/Test transform: {len(val_test_transform.transforms)} operaciones (determinístico)")
'''))

# Celda 11: markdown datasets y DataLoaders
cells.append(nbf.v4.new_markdown_cell("""## 5. Creación de datasets y DataLoaders

### Estrategia de splits

El proyecto provee carpetas `train/` y `test/`. Para poder monitorear overfitting durante el entrenamiento necesitamos un **validation set**, que creamos tomando un **15% del train** (split 85/15).

- **Train (85% del train original):** ~4.250 imágenes → la red entrena aquí.
- **Validation (15% del train original):** ~750 imágenes → medimos generalización durante el entrenamiento (guardar mejor modelo).
- **Test (intacto):** 1.250 imágenes → evaluación final honesta.

El test set NUNCA se usa durante entrenamiento ni para elegir hiperparámetros."""))

# Celda 12: datasets
cells.append(nbf.v4.new_code_cell('''# Cargamos el dataset completo de train con el transform de TRAIN
full_train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_transform)

# Creamos un dataset "gemelo" de train pero con el transform de VAL
# Esto es necesario porque random_split divide el dataset, pero queremos
# que las imágenes de validación usen el transform sin augmentation.
full_train_dataset_for_val = datasets.ImageFolder(root=str(TRAIN_DIR), transform=val_test_transform)

# Generamos índices del split 85/15
n_total = len(full_train_dataset)
n_val = int(0.15 * n_total)
n_train = n_total - n_val

# Usamos generator con seed para reproducibilidad
generator = torch.Generator().manual_seed(SEED)
train_indices, val_indices = random_split(
    range(n_total), [n_train, n_val], generator=generator
)

# Creamos los subsets usando Subset de PyTorch
from torch.utils.data import Subset
train_subset = Subset(full_train_dataset, train_indices.indices)
val_subset = Subset(full_train_dataset_for_val, val_indices.indices)

# Dataset de test
test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=val_test_transform)

print(f"Train: {len(train_subset)} imágenes (con augmentation)")
print(f"Val:   {len(val_subset)} imágenes (sin augmentation)")
print(f"Test:  {len(test_dataset)} imágenes (sin augmentation)")
print(f"Clases: {len(full_train_dataset.classes)}")
'''))

# Celda 13: DataLoaders
cells.append(nbf.v4.new_code_cell('''# DataLoaders: manejan batching, shuffling y paralelismo
# - batch_size=32: cabe cómodo en 4GB de VRAM para CNN custom
# - shuffle=True en train: clave para que la red no memorice el orden
# - num_workers=2: procesos paralelos para cargar imágenes
#   (Windows tiene problemas con num_workers alto, 2 es conservador)
# - pin_memory=True: acelera transferencia CPU→GPU

BATCH_SIZE = 32
NUM_WORKERS = 2

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # No mezclamos validación para resultados consistentes
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"Batches por época: {len(train_loader)} (train), {len(val_loader)} (val), {len(test_loader)} (test)")
'''))

# Celda 14: validación DataLoaders
cells.append(nbf.v4.new_code_cell('''# Verificamos que los DataLoaders funcionan correctamente
# Obtenemos un batch de muestra y verificamos shapes y tipos
images, labels = next(iter(train_loader))

print(f"Shape del batch de imágenes: {images.shape}")
print(f"  → (batch_size, canales, altura, ancho) = ({BATCH_SIZE}, 3, 224, 224) esperado")
print(f"Shape de labels: {labels.shape}")
print(f"Tipo de datos imágenes: {images.dtype}")
print(f"Rango de valores: [{images.min():.3f}, {images.max():.3f}]")
print(f"  → Tras normalización, el rango suele ser aprox [-2, +2]")

assert images.shape == (BATCH_SIZE, 3, 224, 224), "Shape incorrecta del batch"
assert labels.shape == (BATCH_SIZE,), "Shape incorrecta de labels"
print("\\n✅ DataLoaders validados correctamente")
'''))

# Celda 15: visualizar batch aug
cells.append(nbf.v4.new_code_cell('''# Función helper para "des-normalizar" imágenes para visualización
def denormalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Convierte un tensor normalizado a rango [0,1] para mostrar."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)

# Mostramos 8 imágenes del batch con sus labels (muestra el efecto del augmentation)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(8):
    img = denormalize(images[i])
    img_np = img.permute(1, 2, 0).numpy()  # (C,H,W) → (H,W,C) para matplotlib
    axes[i].imshow(img_np)
    axes[i].set_title(f"Clase: {full_train_dataset.classes[labels[i]]}", fontsize=9)
    axes[i].axis("off")

plt.suptitle("Batch de entrenamiento (con data augmentation aplicada)", fontsize=14, fontweight=\'bold\')
plt.tight_layout()
plt.savefig("../report/batch_augmentation.png", dpi=100, bbox_inches=\'tight\')
plt.show()
'''))

# Celda 16: conclusiones markdown
cells.append(nbf.v4.new_markdown_cell("""## Conclusiones de Fase 1

✅ **Dataset verificado:**
- 50 clases, 100 imágenes/clase en train y 25/clase en test
- Balance perfecto (no se requieren class weights)

✅ **Pipeline de preprocesamiento definido:**
- Train con augmentation (crítico para dataset chico)
- Val/Test sin augmentation (evaluación consistente)
- Normalización ImageNet (compatibilidad con ResNet50 en Fase 3)

✅ **Split 85/15 aplicado:**
- Train: ~4.250 imágenes (con augmentation)
- Val: ~750 imágenes (sin augmentation)
- Test: 1.250 imágenes (sin augmentation, intactas)

✅ **DataLoaders validados:**
- Batches de shape (32, 3, 224, 224)
- Shuffle activo en train
- 2 workers paralelos (seguro para Windows)

**Siguiente paso:** Fase 2 - Diseñar y entrenar una CNN desde cero."""))

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python (landmark-classifier)", "language": "python", "name": "landmark-classifier"},
    "language_info": {"name": "python", "version": "3.11.9"}
}

out = "notebooks/00_exploracion_y_preproceso.ipynb"
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Notebook escrito: {out} ({len(cells)} celdas)")
